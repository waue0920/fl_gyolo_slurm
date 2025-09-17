import argparse
import os
import random
import yaml
import shutil
import json
from pathlib import Path

def prepare_data(
    dataset_name: str,
    num_clients: int,
    seed: int,
    project_root: Path,
    source_dir: Path = None,
    output_dir: Path = None,
):
    """
    Splits a gyolo-compatible dataset into multiple partitions for federated learning.

    This function creates symlinks to the original data to save space.

    Args:
        dataset_name: The name of the dataset (e.g., 'coco').
        num_clients: The number of clients to partition the data for.
        seed: The random seed for shuffling to ensure reproducibility.
        project_root: The absolute path to the project's root directory.
        source_dir: Optional. The path to the source dataset. 
                    Defaults to '{project_root}/datasets/{dataset_name}'.
        output_dir: Optional. The path to the output directory for federated data.
                    Defaults to '{project_root}/federated_data/{dataset_name}'.
    """
    print(f"--- Starting data preparation for gyolo dataset: {dataset_name} ---")
    print(f"--- Number of clients: {num_clients} ---")

    # If paths are not provided, construct them based on project_root
    # 支援 coco 資料集特殊路徑
    if dataset_name == "coco":
        source_dir = project_root / "coco"
        source_images_dir = source_dir / "images" / "train2017"
        source_labels_dir = source_dir / "labels" / "train2017"
        source_yaml_path = project_root / "gyolo" / "data" / f"{dataset_name}.yaml"
    else:
        ### TBD: 支援其他資料集格式
        if source_dir is None:
            source_dir = project_root / "datasets" / dataset_name
        source_images_dir = source_dir / "images" / "train"
        source_labels_dir = source_dir / "labels" / "train"
        source_yaml_path = project_root / "data" / f"{dataset_name}.yaml"
        
    if output_dir is None:
        output_dir = project_root / "federated_data" / f"{dataset_name}_{num_clients}"

    # --- 1. Validation ---
    if not source_images_dir.is_dir():
        print(f"Error: Source image directory not found at: {source_images_dir}")
        exit(1)
    if not source_labels_dir.is_dir():
        print(f"Error: Source label directory not found at: {source_labels_dir}")
        exit(1)
    if not source_yaml_path.is_file():
        print(f"Error: Source YAML file not found at: {source_yaml_path}")
        exit(1)

    # --- Anti-fool mechanism: Check if data is already partitioned ---
    if output_dir.exists() and any(output_dir.iterdir()):
         print(f"Output directory '{output_dir}' already exists and is not empty. Skipping preparation.")
         return

    print(f"Reading images from: {source_images_dir}")
    image_files = sorted([f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print(f"Error: No images found in {source_images_dir}")
        exit(1)

    print(f"Found {len(image_files)} total images.")

    # --- 2. Shuffle and Split ---
    random.seed(seed)
    random.shuffle(image_files)
    file_chunks = [image_files[i::num_clients] for i in range(num_clients)]
    print(f"Successfully shuffled and split data into {num_clients} chunks.")

    # --- 3. Create Output Directories and Symlinks ---
    print(f"Creating base output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(source_yaml_path, 'r') as f:
        original_yaml_data = yaml.safe_load(f)
        # Resolve the original validation path relative to the source dataset directory
        original_val_path = source_dir / original_yaml_data['val']

    for i in range(1, num_clients + 1):
        ### gyolo 要處理 annotations, images, labels, stuff 四個資料夾，且 train , val, test 取名為 train2017, val2017, test2017
        client_id = f"c{i}"
        client_dir = output_dir / client_id
        print(f"Processing {client_id}...")

        ### images 要取名為 train2017
        client_images_dir = client_dir / "images" / "train2017"
        client_images_dir.mkdir(parents=True, exist_ok=True)        
        ### labels 要取名為 train2017
        client_labels_dir = client_dir / "labels" / "train2017"
        client_labels_dir.mkdir(parents=True, exist_ok=True)
        ### stuff 
        source_stuff_labels_dir = project_root / "coco" / "stuff" / "train2017"
        client_stuff_labels_dir = client_dir / "stuff" / "train2017"
        if source_stuff_labels_dir.is_dir():
            client_stuff_labels_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  - No source stuff labels found, skipping stuff symlinks for {client_id}.")
        ### annotations
        client_annotations_dir = client_dir / "annotations"        
        client_annotations_dir.mkdir(parents=True, exist_ok=True)
        ### annotations, 不可用 symlink 指向 coco 原始資料夾 (caption 會出錯)
        # source_annotations_dir = project_root / "coco" / "annotations"
        # client_annotations_dir = client_dir / "annotations"
        # if source_annotations_dir.is_dir():
        #     if client_annotations_dir.exists() or client_annotations_dir.is_symlink():
        #         client_annotations_dir.unlink()
        #     os.symlink(source_annotations_dir.resolve(), client_annotations_dir)
        #     print(f"  - Symlinked annotations to {client_annotations_dir}")
        # else:
        #     print(f"  - No source annotations found, skipping annotations symlink for {client_id}.")        


        symlink_img, symlink_det, symlink_stuff = 0, 0, 0
        client_image_names = set()
        for image_name in file_chunks[i-1]:
            base_name = Path(image_name).stem
            label_name = f"{base_name}.txt"
            source_image_path = source_images_dir / image_name
            source_label_path = source_labels_dir / label_name
            source_stuff_label_path = source_stuff_labels_dir / label_name

            # 影像 symlink
            if source_image_path.exists():
                os.symlink(source_image_path.resolve(), client_images_dir / image_name)
                symlink_img += 1
                client_image_names.add(base_name)
            else:
                print(f"Warning: Image file not found for {image_name}, skipping.")

            # detection label symlink
            if source_label_path.exists():
                os.symlink(source_label_path.resolve(), client_labels_dir / label_name)
                symlink_det += 1
            else:
                print(f"Warning: Detection label not found for {image_name}, skipping.")

            # stuff label symlink
            if source_stuff_labels_dir.is_dir() and source_stuff_label_path.exists():
                os.symlink(source_stuff_label_path.resolve(), client_stuff_labels_dir / label_name)
                symlink_stuff += 1
            elif source_stuff_labels_dir.is_dir():
                print(f"Warning: Stuff label not found for {image_name}, skipping.")

        # 分割 captions_train2017.json，產生 client 專屬 captions_train2017.json

        coco_ann_path = project_root / "coco" / "annotations" / "captions_train2017.json"
        client_ann_path = client_annotations_dir / "captions_train2017.json"
        if coco_ann_path.is_file():
            with open(coco_ann_path, 'r') as f:
                coco_ann = json.load(f)
            # 建立 image_id 對應表
            image_id_map = {str(img['file_name']).split('.')[0]: img['id'] for img in coco_ann['images']}
            client_image_ids = set([image_id_map[name] for name in client_image_names if name in image_id_map])
            # 過濾 images/annotations
            client_images = [img for img in coco_ann['images'] if img['id'] in client_image_ids]
            client_annotations = [ann for ann in coco_ann['annotations'] if ann['image_id'] in client_image_ids]
            # 寫出 client captions_train.json
            client_coco_ann = coco_ann.copy()
            client_coco_ann['images'] = client_images
            client_coco_ann['annotations'] = client_annotations
            with open(client_ann_path, 'w') as f:
                json.dump(client_coco_ann, f)
            print(f"  - Generated client captions_train2017.json at {client_ann_path}")
        else:
            print(f"  - No source captions_train2017.json found, skipping client annotation json for {client_id}.")

        # --- 4. Generate Client YAML file ---
        client_yaml_data = original_yaml_data.copy()
        client_yaml_data['path'] = str(client_dir)
        client_yaml_data['train'] = 'images/train2017'
        client_yaml_data['val'] = str(project_root / "coco" / "images" / "val2017")
        client_yaml_data['stuff'] = str('stuff/train2017')
        client_yaml_data['test'] = str(project_root / "coco" / "test-dev2017.txt")
        client_yaml_data.pop('download', None)  # 移除 download 欄位

        client_yaml_path = output_dir / f"{client_id}.yaml"
        with open(client_yaml_path, 'w') as f:
            yaml.dump(client_yaml_data, f, sort_keys=False, default_flow_style=False)

        print(f"  - Generated YAML config at: {client_yaml_path}")

    print("--- Data preparation complete. ---")


if __name__ == "__main__":
    # This script is in 'src/', so the project root is one level up.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Partition a dataset for Federated Learning.")
    
    parser.add_argument('--dataset-name', type=str, required=True, help='Name of the dataset (e.g., kitti).')
    parser.add_argument('--num-clients', type=int, default=4, help='Number of clients.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    # Optional arguments for custom paths, useful for testing
    parser.add_argument('--source-dir', type=Path, default=None, help='(Optional) Path to the source dataset directory.')
    parser.add_argument('--output-dir', type=Path, default=None, help='(Optional) Path to the output directory.')
    
    args = parser.parse_args()

    prepare_data(
        dataset_name=args.dataset_name,
        num_clients=args.num_clients,
        seed=args.seed,
        project_root=PROJECT_ROOT,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )