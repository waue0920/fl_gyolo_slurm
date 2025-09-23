
import argparse
import torch
import glob
import sys
import os
from collections import OrderedDict
from pathlib import Path
from aggregation import AGGREGATORS
import pandas as pd
import time

# Add gyolo directory to Python path for model imports
if 'WROOT' in os.environ:
    gyolo_path = os.path.join(os.environ['WROOT'], 'gyolo')
    if gyolo_path not in sys.path:
        sys.path.insert(0, gyolo_path)

def calculate_alg_complexity(algorithm=None, template_model=None, expected_clients=None, start_time=None):
    BYTE=8
    elapsed = None
    if start_time is not None:
        elapsed = time.time() - start_time
    num_params = sum(p.numel() for p in template_model.parameters()) if template_model is not None else 0
    param_bits = sum(p.numel() * p.element_size() * BYTE for p in template_model.parameters()) if template_model is not None else 0

    print(f"[SUMMARY] Aggregation time (s): {elapsed:.3f}")    

    if algorithm == 'fedopt':
        total_bits = param_bits * 3
        print(f"[SUMMARY] SpaceComplexity <3P> (bit): {total_bits:,}")
    elif algorithm == 'fedavgm':
        total_bits = param_bits * 2
        print(f"[SUMMARY] SpaceComplexity <2P> (bit): {total_bits:,}")
    elif algorithm == 'fedprox':
        total_bits = param_bits
        print(f"[SUMMARY] SpaceComplexity <1P> (bit): {total_bits:,}")
    elif algorithm == 'fednova':
        total_bits = param_bits + expected_clients * 32
        print(f"[SUMMARY] SpaceComplexity <1P+N> (bit): {total_bits:,}")
    elif algorithm == 'fedawa':
        total_bits = expected_clients * param_bits + expected_clients**2 * 32
        print(f"[SUMMARY] SpaceComplexity <(N^2 + NP)> (bit): {total_bits:,}")
    else:
        print(f"[SUMMARY] SpaceComplexity <1P> (bit): {param_bits:,}")

    print(f"[SUMMARY] TemplateModelParams (count): {num_params:,}")


def federated_aggregate(input_dir: Path, output_file: Path, expected_clients: int, round_num: int, algorithm: str, **agg_kwargs):
    print("================================================================================")
    print(f"Starting Federated Aggregation: {algorithm}")
    print(f">> Input directory:  {input_dir}")
    print(f">> Output file:      {output_file}")
    print(f">> Expected clients: {expected_clients}")
    print(f">> Target round:     {round_num}")
    print(f">> Algorithm:        {algorithm}")
    print("================================================================================")

    # 1. Find all client weights for the specific round
    client_weights_pattern = str(input_dir / f"r{round_num}_c*" / "weights" / "best.pt")
    client_weights_paths = sorted(glob.glob(client_weights_pattern))
    print(f"Searching for pattern: r{round_num}_c*/weights/best.pt")
    num_found = len(client_weights_paths)
    if num_found != expected_clients:
        print(f"Error: Mismatch in number of clients!")
        print(f"  - Expected: {expected_clients}")
        print(f"  - Found:    {num_found} ({client_weights_paths})")
        print("  - Please check the training logs for failed clients before proceeding.")
        exit(1)
    if num_found == 0:
        print(f"Error: No client weights found in '{input_dir}'")
        exit(1)
    print(f"Found {num_found} client models for aggregation:")
    for path in client_weights_paths:
        print(f"  - {path}")

    # 2. Load all client model state_dicts and get template model
    all_state_dicts = []
    template_model = None
    template_ckpt = None
    # TODO: This needs to be implemented to get client sizes for weighted average
    client_sizes = [1] * num_found # Placeholder: assume equal sizes for now
    print(f"[INFO] Using placeholder for client sizes: {client_sizes}")

    for i, path in enumerate(client_weights_paths):
        try:
            ckpt = torch.load(path, map_location='cpu')
            # gyolo best.pt 可能直接是 state_dict 或 dict 需根據實際格式調整
            if 'model' in ckpt:
                all_state_dicts.append(ckpt['model'].state_dict())
                if i == 0:
                    template_model = ckpt['model']
                    template_ckpt = ckpt
            elif 'state_dict' in ckpt:
                all_state_dicts.append(ckpt['state_dict'])
                if i == 0:
                    template_model = None # This case needs careful handling
                    template_ckpt = ckpt
            else:
                # Assuming the ckpt is the state_dict itself
                all_state_dicts.append(ckpt)
                if i == 0:
                    # This case is tricky as we don't have a model structure
                    template_ckpt = {'model_state_dict': ckpt}
        except Exception as e:
            print(f"\nError: Failed to load weight file: {path}")
            print(f"  - Reason: {e}")
            exit(1)

    if template_model is None and 'model' not in template_ckpt:
        # If we only have state_dicts, we can't easily create a template model.
        # This requires the user to provide a model config or load a base model.
        # For now, we assume the first client's checkpoint contains a 'model' object.
        print("Error: Could not load template model structure from client weights.")
        print("       Ensure the checkpoint contains the 'model' object.")
        exit(1)

    # 3. Perform aggregation

    agg_fn = AGGREGATORS.get(algorithm)
    if agg_fn is None:
        print(f"Error: Unknown aggregation algorithm: {algorithm}")
        exit(1)
    print(f"\nAggregating weights using {algorithm}...")

    start_time = time.time()
    if algorithm == 'fedopt':
        aggregated, optimizer_state = agg_fn(all_state_dicts, **agg_kwargs)
        print("Aggregation complete.")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print("Loading aggregated weights into template model...")
        template_model.load_state_dict(aggregated)
        # 計算複雜度
        calculate_alg_complexity(
            algorithm=algorithm, template_model=template_model, expected_clients=expected_clients, start_time=start_time)
        model_to_save = {
            'model': template_model,
            'optimizer': template_ckpt.get('optimizer', None),
            'epoch': template_ckpt.get('epoch', -1),
            'training_results': template_ckpt.get('training_results', None)
        }
        torch.save(model_to_save, output_file)
        # 保存 optimizer_state
        opt_state_path = input_dir / f"fedopt_state.pt"
        torch.save(optimizer_state, opt_state_path)
        print(f"\nSuccessfully saved aggregated model to:")
        print(f"  -> {output_file}")
        print(f"Optimizer state saved to: {opt_state_path}")
        print("================================================================================")
    elif algorithm == 'fedavgm':
        aggregated, new_momentum = agg_fn(all_state_dicts, **agg_kwargs)
        print("Aggregation complete.")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print("Loading aggregated weights into template model...")
        template_model.load_state_dict(aggregated)
        # 計算複雜度
        calculate_alg_complexity(
            algorithm=algorithm, template_model=template_model, expected_clients=expected_clients, start_time=start_time)
        model_to_save = {
            'model': template_model,
            'optimizer': template_ckpt.get('optimizer', None),
            'epoch': template_ckpt.get('epoch', -1),
            'training_results': template_ckpt.get('training_results', None),
            'server_momentum': new_momentum
        }
        torch.save(model_to_save, output_file)
        print(f"\nSuccessfully saved aggregated model to:")
        print(f"  -> {output_file}")
        print("================================================================================")
    else:
        aggregated = agg_fn(all_state_dicts, **agg_kwargs)
        print("Aggregation complete.")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print("Loading aggregated weights into template model...")
        template_model.load_state_dict(aggregated)
        # 計算複雜度
        calculate_alg_complexity(
            algorithm=algorithm, template_model=template_model, expected_clients=expected_clients, start_time=start_time)        
        model_to_save = {
            'model': template_model,
            'optimizer': template_ckpt.get('optimizer', None),
            'epoch': template_ckpt.get('epoch', -1),
            'training_results': template_ckpt.get('training_results', None)
        }
        torch.save(model_to_save, output_file)
        print(f"\nSuccessfully saved aggregated model to:")
        print(f"  -> {output_file}")
        print("================================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Federated Aggregation Script for YOLOv9 models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input-dir', type=Path, required=True, help="Directory containing the client training outputs for a single round.")
    parser.add_argument('--output-file', type=Path, required=True, help="Full path for the output aggregated model.")
    parser.add_argument('--expected-clients', type=int, required=True, help="The number of client models expected to be in the input directory.")
    parser.add_argument('--round', type=int, required=True, help="Round number to filter client weights (e.g., 1, 2, 3, ...).")
    parser.add_argument('--algorithm', type=str, default='fedavg', choices=AGGREGATORS.keys(), help="Aggregation algorithm to use.")
    # 可擴充：parser.add_argument('--mu', type=float, default=0.01, help="FedProx mu value")
    args = parser.parse_args()

    server_fedprox_mu_env = os.environ.get('SERVER_FEDPROX_MU')

    client_weights_pattern = str(args.input_dir / f"r{args.round}_c*" / "weights" / "best.pt")
    client_weights_paths = sorted(glob.glob(client_weights_pattern))

    # 統一參數準備
    agg_kwargs = {}
    if args.algorithm == 'fedprox':
        pass  # 不再傳遞 mu，FedProx 內部自取 os.environ
            
    # fedopt 參數準備
    if args.algorithm == 'fedopt':
        if args.round == 1:
            # 第一輪直接載入 INITIAL_WEIGHTS (gyolo.pt)
            gyolo_pt_path = os.environ.get('INITIAL_WEIGHTS')
            if not gyolo_pt_path or not os.path.exists(gyolo_pt_path):
                print(f"Error: INITIAL_WEIGHTS not set or file not found: {gyolo_pt_path}")
                exit(1)
            ckpt = torch.load(gyolo_pt_path, map_location='cpu')
            if 'model' in ckpt:
                template_model = ckpt['model']
            else:
                print("Error: INITIAL_WEIGHTS checkpoint must contain 'model' object.")
                exit(1)
            agg_kwargs['global_weights'] = template_model.state_dict()
        else:
            # 第二輪以後載入上一輪聚合結果
            prev_agg_path = args.output_file.parent / f"w_s_r{args.round-1}.pt"
            if not os.path.exists(prev_agg_path):
                print(f"Error: Previous round weights not found: {prev_agg_path}")
                exit(1)
            ckpt = torch.load(prev_agg_path, map_location='cpu')
            if 'model' in ckpt:
                template_model = ckpt['model']
            else:
                print("Error: Previous round checkpoint must contain 'model' object.")
                exit(1)
            agg_kwargs['global_weights'] = template_model.state_dict()
        # 不再傳遞 lr, beta1, beta2, eps，FedOpt 內部自取 os.environ

    # fednova 參數準備
    if args.algorithm == 'fednova':
        # 先載入 client_weights 以取得 template_model
        client_weights_pattern = str(args.input_dir / f"r{args.round}_c*" / "weights" / "best.pt")
        client_weights_paths = sorted(glob.glob(client_weights_pattern))
        if not client_weights_paths:
            print(f"Error: No client weights found for fednova parameter preparation.")
            exit(1)
        ckpt = torch.load(client_weights_paths[0], map_location='cpu')
        if 'model' in ckpt:
            template_model = ckpt['model']
        elif 'state_dict' in ckpt:
            print("Error: fednova requires a model object in checkpoint.")
            exit(1)
        else:
            print("Error: fednova requires a model object in checkpoint.")
            exit(1)
        agg_kwargs['server_weights'] = template_model.state_dict()
        agg_kwargs['client_steps'] = [1] * args.expected_clients  # 可改為真實步數
        # 不再傳遞 mu, lr，FedNova 內部自取 os.environ

    # fedawa 參數準備
    if args.algorithm == 'fedawa':
        client_results_pattern = str(args.input_dir / f"r{args.round}_c*" / "results.csv")
        client_results_paths = sorted(glob.glob(client_results_pattern))
        if not client_weights_paths:
            print(f"Error: No client weights found for fedawa parameter preparation.")
            exit(1)
        if not client_results_paths:
            print(f"Error: No client results found for fedawa parameter preparation.")
            exit(1)
        ckpt = torch.load(client_weights_paths[0], map_location='cpu')
        if 'model' in ckpt:
            template_model = ckpt['model']
        elif 'state_dict' in ckpt:
            print("Error: fedawa requires a model object in checkpoint.")
            exit(1)
        else:
            print("Error: fedawa requires a model object in checkpoint.")
            exit(1)
        agg_kwargs['global_weights'] = template_model.state_dict()
        # 取得 client_vectors，包含 weights_history 與 results_history
        client_vectors = []
        for w_path, r_path in zip(client_weights_paths, client_results_paths):
            weights_history = [torch.load(w_path, map_location='cpu')]
            try:
                df = pd.read_csv(r_path)
                results_history = df.to_dict(orient='records')
            except Exception as e:
                print(f"Warning: Failed to read {r_path}: {e}")
                results_history = []
            client_vectors.append({
                'weights_history': weights_history,
                'results': results_history,
                'history': []
            })
        agg_kwargs['client_vectors'] = client_vectors

    if args.algorithm == 'fedavgm':
        # 準備 global_weights
        ckpt = torch.load(client_weights_paths[0], map_location='cpu')
        if 'model' in ckpt:
            template_model = ckpt['model']
        elif 'state_dict' in ckpt:
            template_model = None
        else:
            template_model = None
        
        all_state_dicts = []
        for path in client_weights_paths:
            try:
                ckpt = torch.load(path, map_location='cpu')
                if 'model' in ckpt:
                    all_state_dicts.append(ckpt['model'].state_dict())
                elif 'state_dict' in ckpt:
                    all_state_dicts.append(ckpt['state_dict'])
                else:
                    all_state_dicts.append(ckpt)
            except Exception as e:
                print(f"\nError: Failed to load weight file: {path}")
                print(f"  - Reason: {e}")
                exit(1)
        
        agg_kwargs['global_weights'] = template_model.state_dict() if template_model else all_state_dicts[0]
        agg_kwargs['client_sizes'] = [1] * int(args.expected_clients)
        # 準備 server_momentum（可用零向量或前一輪結果）
        agg_kwargs['server_momentum'] = {k: torch.zeros_like(v) for k, v in agg_kwargs['global_weights'].items()}

    federated_aggregate(args.input_dir, args.output_file, args.expected_clients, args.round, args.algorithm, **agg_kwargs)
