import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from datetime import timedelta

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import caption.val as validate
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import GenericLogger
from utils.model_utils import DET_LAYERS, CAP_LAYERS, find_layer
from utils.plots import plot_evolve, plot_labels
from utils.caption.dataloaders import create_dataloader
from utils.caption.loss_tal_dual import ComputeLossLH as ComputeLoss
from utils.caption.loss_tal import ComputeLoss as ComputeLossVal
from utils.caption.loss_tal_dual import ComputeCaptionLoss
from utils.caption.metrics import KEYS, fitness
from utils.caption.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None#check_git_info()


def custom_optimizer(model, hyp, name='Adam'):
    model_key = de_parallel(model).model
    caption_idx = find_layer(model_key, CAP_LAYERS)

    lr, momentum, decay = hyp['lr0'], hyp['momentum'], hyp['weight_decay']

    g = [], [], [], [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for k, v in model.named_modules():
        # Grit layer
        if f'model.{caption_idx}.' in k:
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g[5].append(v.bias)  # biases
            if isinstance(v, bn):
                g[4].append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g[3].append(v.weight)  # apply decay
            continue

        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g[2].append(v.bias)  # biases
        if isinstance(v, bn):
            g[1].append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g[0].append(v.weight)  # apply decay

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        optimizer2 = torch.optim.Adam(g[5], lr=lr/2., betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), eps = 1e-4, weight_decay = 0.0)
        optimizer2 = torch.optim.AdamW(g[5], lr=lr/2., betas=(momentum, 0.999), eps = 1e-4, weight_decay = 0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        optimizer2 = torch.optim.RMSprop(g[5], lr=lr/2., momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        optimizer2 = torch.optim.SGD(g[5], lr=lr/2., momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer2.add_param_group({'params': g[4], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    optimizer2.add_param_group({'params': g[3], 'weight_decay': decay})  # add g0 with weight_decay

    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
                f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer, optimizer2


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, mask_ratio = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.mask_ratio

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        logger = GenericLogger(opt=opt, console_logger=LOGGER)

    # Config
    plots = not evolve and not opt.noplots  # create plots
    overlap = not opt.no_overlap
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')  # COCO dataset

    # Define all tasks
    all_tasks = {'vision': ['detection', 'semantic'], 'language': ['caption']} if ('all_tasks' not in hyp) else hyp['all_tasks']

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = SegmentationModel(cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = SegmentationModel(cfg, ch=3, nc=nc).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        logger.update_params({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer, optimizer2 = custom_optimizer(model, hyp, name=opt.optimizer)

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        lf2 = one_cycle(1, (hyp['lrf']*2.), epochs)  # cosine 1->hyp['lrf']
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
        lf2 = one_flat_cycle(1, (hyp['lrf']*2.), epochs)  # cosine 1->hyp['lrf']
    elif opt.fixed_lr:
        lf = lambda x: 1.0
        lf2 = lambda x: 1.0
    elif opt.poly_lr:
        power = 0.9
        lf = lambda x: ((1 - (x / epochs)) ** power) * (1.0 - hyp['lrf']) + hyp['lrf']
        lf2 = lambda x: ((1 - (x / epochs)) ** power) * (1.0 - (hyp['lrf']*2.)) + (hyp['lrf']*2.)
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        lf2 = lambda x: (1 - x / epochs) * (1.0 - (hyp['lrf']*2.)) + (hyp['lrf']*2.)  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler2 = lr_scheduler.LambdaLR(optimizer2, lr_lambda=lf2)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, optimizer2, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp = hyp,
        augment = True,
        cache = None if opt.cache == 'val' else opt.cache,
        rect = opt.rect,
        rank = LOCAL_RANK,
        workers = workers,
        image_weights = opt.image_weights,
        close_mosaic = opt.close_mosaic != 0,
        quad = opt.quad,
        prefix = colorstr('train: '),
        shuffle = True,
        mask_downsample_ratio = mask_ratio,
        overlap_mask = overlap,
        is_train = True,
        pin_memory = cuda,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp = hyp,
            cache = None if noval else opt.cache,
            rect = False,
            rank = -1,
            workers = workers * 2,
            pad = 0.5,
            mask_downsample_ratio = mask_ratio,
            overlap_mask = overlap,
            prefix = colorstr('val: '),
            is_train = False,
            pin_memory = cuda,
        )[0]

        if not resume:
            model.half().float()  # pre-reduce anchor precision

            if plots:
                plot_labels(labels, names, save_dir)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model, find_unused_parameters = True)

    # Model attributes
    model_key = de_parallel(model).model
    det_layer = find_layer(model_key, DET_LAYERS)
    m = model_key[det_layer]  # Detect() module
    nl = m.nl  # number of detection layers (to scale hyps)
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    caption_idx = find_layer(model_key, CAP_LAYERS)

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model, overlap=overlap)  # init loss class
    compute_loss_val = ComputeLossVal(model, overlap=overlap)  # init loss class
    compute_caption_loss = ComputeCaptionLoss(model)
    # callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
        if epoch == (epochs - opt.close_mosaic):
            LOGGER.info("Closing dataloader mosaic")
            dataset.mosaic = False

        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        pbar_desc = ('Epoch', 'GPU_mem', 'box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'fcl_loss', 'dic_loss', 'cap_loss', 'Instances', 'Size')
        num_detection_loss, num_semantic_loss, num_caption_loss = 4, 2, 1

        # mloss = torch.zeros(6, device=device)  # mean losses
        mloss = torch.zeros((num_detection_loss + num_semantic_loss + num_caption_loss), device = device)  # mean losses

        LOGGER.info(('\n' + '%11s' * len(pbar_desc)) % pbar_desc)
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        optimizer2.zero_grad()
        for i, (imgs, ori_imgs, targets, paths, _, masks, semasks, src_masks, caps, cap_masks, _) in pbar:  # batch ------------------------------------------------------

            if caption_idx is not None:
                src_masks = src_masks.to(device)
                caps = caps.to(device)
                cap_masks = cap_masks.to(device)
                model_key[caption_idx].set_params(src_masks, caps, cap_masks)

            ni = i + nb * epoch  # number integrated batches (since train start)

            for i_task in all_tasks:
                # uint8 to float32, 0-255 to 0.0-1.0
                imgs = imgs.to(device, non_blocking=True).float() / 255 \
                    if ('vision' == i_task) else ori_imgs.to(device, non_blocking = True).float() / 255

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    if 'vision' == i_task:
                        pred_detect = pred['detect']
                        loss, loss_items = compute_loss(pred_detect, targets.to(device), masks=masks.to(device).float(),
                                                    semasks=semasks.to(device).float())
                    else: # 'language' == i_task
                        pred_cap = pred['captions']
                        loss, loss_items = compute_caption_loss(pred_cap.float(), caps)

                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                # Backward
                torch.use_deterministic_algorithms(False)
                scaler.scale(loss).backward()

                if 'vision' == i_task:
                    scaler.unscale_(optimizer)  # unscale gradients
                else:
                    scaler.unscale_(optimizer2)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(
                    [v for k, v in model.named_parameters() if ('.' + str(caption_idx) + '.' not in k)],
                    max_norm = 10.0
                )  # clip gradients

                torch.nn.utils.clip_grad_norm_(
                    [v for k, v in model.named_parameters() if ('.' + str(caption_idx) + '.' in k)],
                    max_norm = hyp['caption_clip_max_norm']
                )  # cap. head

                if 'vision' == i_task:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    scaler.step(optimizer2)  # optimizer.step
                    scaler.update()
                    optimizer2.zero_grad()
                if ema:
                    ema.update(model)
                # last_opt_step = ni

                # Log
                if RANK in {-1, 0}:
                    if 'vision' == i_task:
                        loss_items = torch.cat((loss_items, torch.zeros(num_caption_loss, device = device)), dim = 0)
                    else:  # 'language' == i_task
                        loss_items = torch.cat((torch.zeros((num_detection_loss + num_semantic_loss), device = device), loss_items), dim = 0)
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%11s' * 2 + '%11.4g' * (len(pbar_desc) - 2)) %
                                        (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

                    # Mosaic plots
                    if ('vision' == i_task) and plots:
                        if ni < 10:
                            plot_images_and_masks(imgs, targets, masks, semasks, paths, save_dir / f"train_batch{ni}.jpg")
                        if ni == 10:
                            files = sorted(save_dir.glob('train*.jpg'))
                            logger.log_images(files, "Mosaics", epoch)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()
        scheduler2.step()

        if RANK in {-1, 0}:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                if (opt.save_period > 0 and epoch % opt.save_period == 0) or (epoch > (epochs - 2 * opt.close_mosaic)):
                    results, maps, _ = validate.run(
                        data_dict,
                        batch_size = batch_size // WORLD_SIZE * 2,
                        imgsz = imgsz,
                        half = False,
                        model = ema.ema,
                        single_cls = single_cls,
                        dataloader = val_loader,
                        save_dir = save_dir,
                        plots = False,
                        callbacks = callbacks,
                        compute_loss = compute_loss_val,
                        compute_caption_loss = compute_caption_loss,
                        mask_downsample_ratio = mask_ratio,
                        overlap = overlap,
                    )

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            metrics_dict = dict(zip(KEYS, log_vals))
            logger.log_metrics(metrics_dict, epoch)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'optimizer2': optimizer2.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                    logger.log_model(w / f'epoch{epoch}.pt')
                del ckpt
                # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size = batch_size // WORLD_SIZE * 2,
                        imgsz = imgsz,
                        half = False,
                        model = attempt_load(f, device).half(),
                        iou_thres = 0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls = single_cls,
                        dataloader = val_loader,
                        save_dir = save_dir,
                        save_json = is_coco,
                        # verbose = True,
                        plots = plots,
                        callbacks = callbacks,
                        compute_loss = compute_loss_val,
                        compute_caption_loss = compute_caption_loss,
                        mask_downsample_ratio = mask_ratio,
                        overlap = overlap
                    )  # val best model with plots
                    if is_coco:
                        metrics_dict = dict(zip(KEYS, list(mloss) + list(results) + lr))
                        logger.log_metrics(metrics_dict, epoch)

        logger.log_metrics(dict(zip(KEYS[7:25], results)), epochs)
        if not opt.evolve:
            logger.log_model(best, epoch)
        if plots:
            plot_results_with_masks(file=save_dir / 'results.csv')  # save results.png
            files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
            files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
            logger.log_images(files, "Results", epoch + 1)
            logger.log_images(sorted(save_dir.glob('val*.jpg')), "Validation", epoch + 1)
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolo-cap.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'LION'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train-cap', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--flat-cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--fixed-lr', action='store_true', help='fixed LR scheduler')
    parser.add_argument('--poly-lr', action='store_true', help='fixed LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--close-mosaic', type=int, default=0, help='Experimental')

    # Instance Segmentation Args
    parser.add_argument('--mask-ratio', type=int, default=4, help='Downsample the truth masks to saving memory')
    parser.add_argument('--no-overlap', action='store_true', help='Overlap masks train faster at slightly less mAP')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # Resume
    if opt.resume and not opt.evolve:  # resume from specified or most recent last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        
        ## waue add for wandb
        opt.project = Path(opt.project).name  ## waue add for wandb 

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLO Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend = "nccl" if dist.is_nccl_available() else "gloo", timeout = timedelta(seconds = 86400))

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(KEYS, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
