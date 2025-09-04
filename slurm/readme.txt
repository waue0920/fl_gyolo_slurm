### ###


* train

```
torchrun --nproc_per_node 8 --master_port 9527 caption/train.py --data coco.yaml --epochs 60 --batch 64 --img 640 --cfg models/caption/gyolo.yaml --name gyolo --weights '' --hyp data/hyps/hyp.cap.scratch.yaml --device 0,1,2,3,4,5,6,7 --optimizer AdamW --flat-cos-lr --no-overlap --close-mosaic 2 --save-period 1 --noval --noplots

```

* Cider Optimization for Improving Image Captioning
```
torchrun --nproc_per_node 8 --master_port 9527 caption/tune.py --weights 'gyolo.pt' --cfg models/caption/gyolo.yaml --name gyolo-co --data data/coco.yaml --hyp data/hyps/hyp.scratch-cap.yaml --epochs 10 --batch-size 64 --img 640 --device 0,1,2,3,4,5,6,7 --optimizer AdamW --save-period 1
```

