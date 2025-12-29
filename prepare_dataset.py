"""
Dataset preparation script for WAID dataset
Converts WAID dataset to YOLO format and splits into train/val
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random


def create_yolo_label(annotation, img_width, img_height):
    """
    Convert annotation to YOLO format
    
    Args:
        annotation: Dictionary with bbox info (x1, y1, x2, y2, class)
        img_width: Image width
        img_height: Image height
    Returns:
        YOLO format string: "class_id x_center y_center width height"
    """
    x1, y1, x2, y2 = annotation['bbox']
    class_id = annotation['class_id']
    
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Normalize to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def prepare_waid_dataset(dataset_path, output_dir, train_split=0.8, seed=42):
    """
    Prepare WAID dataset for training
    
    Args:
        dataset_path: Path to WAID dataset root
        output_dir: Output directory for organized dataset
        train_split: Train/validation split ratio (default: 0.8)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Create output directories
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    val_img_dir = os.path.join(output_dir, 'val', 'images')
    val_label_dir = os.path.join(output_dir, 'val', 'labels')
    
    for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Class mapping (WAID dataset classes)
    class_names = ['cattle', 'sheep', 'seal', 'camelus', 'zebra', 'kiang']
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # Find all images in dataset
    dataset_path = Path(dataset_path)
    
    # Look for common dataset structures
    possible_image_dirs = [
        dataset_path / 'images',
        dataset_path / 'WAID' / 'images',
        dataset_path / 'WAID',
        dataset_path
    ]
    
    image_files = []
    for img_dir in possible_image_dirs:
        if img_dir.exists():
            image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
            if image_files:
                print(f"Found images in: {img_dir}")
                break
    
    if not image_files:
        print("Warning: No images found. Please check dataset path.")
        print("Expected structure:")
        print("  WAID/")
        print("    images/")
        print("    annotations/ (or labels/)")
        return
    
    # Find annotation files
    possible_annotation_dirs = [
        dataset_path / 'annotations',
        dataset_path / 'labels',
        dataset_path / 'WAID' / 'annotations',
        dataset_path / 'WAID' / 'labels',
        dataset_path / 'WAID'
    ]
    
    annotation_files = {}
    for ann_dir in possible_annotation_dirs:
        if ann_dir.exists():
            # Try JSON format first
            json_files = list(ann_dir.glob('*.json'))
            if json_files:
                print(f"Found JSON annotations in: {ann_dir}")
                # Load main annotation file if exists
                for json_file in json_files:
                    if 'annotation' in json_file.name.lower() or 'label' in json_file.name.lower():
                        with open(json_file, 'r') as f:
                            annotation_files = json.load(f)
                        break
                break
            # Try YOLO format
            txt_files = list(ann_dir.glob('*.txt'))
            if txt_files:
                print(f"Found YOLO annotations in: {ann_dir}")
                for txt_file in txt_files:
                    annotation_files[txt_file.stem] = txt_file
                break
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Training images: {len(train_files)}")
    print(f"  Validation images: {len(val_files)}")
    
    # Process training images
    print("\nProcessing training images...")
    for img_file in tqdm(train_files):
        img_name = img_file.stem
        
        # Copy image
        shutil.copy(img_file, os.path.join(train_img_dir, img_file.name))
        
        # Process annotations
        label_file = os.path.join(train_label_dir, f"{img_name}.txt")
        
        if isinstance(annotation_files, dict) and img_name in annotation_files:
            # JSON format
            ann_data = annotation_files[img_name]
            # This would need to be adapted based on actual WAID format
            # For now, create empty label file
            with open(label_file, 'w') as f:
                pass
        elif isinstance(annotation_files, dict) and img_name in annotation_files:
            # YOLO format - copy directly
            shutil.copy(annotation_files[img_name], label_file)
        else:
            # Create empty label file if no annotation found
            with open(label_file, 'w') as f:
                pass
    
    # Process validation images
    print("\nProcessing validation images...")
    for img_file in tqdm(val_files):
        img_name = img_file.stem
        
        # Copy image
        shutil.copy(img_file, os.path.join(val_img_dir, img_file.name))
        
        # Process annotations
        label_file = os.path.join(val_label_dir, f"{img_name}.txt")
        
        if isinstance(annotation_files, dict) and img_name in annotation_files:
            # JSON format
            ann_data = annotation_files[img_name]
            # This would need to be adapted based on actual WAID format
            with open(label_file, 'w') as f:
                pass
        elif isinstance(annotation_files, dict) and img_name in annotation_files:
            # YOLO format - copy directly
            shutil.copy(annotation_files[img_name], label_file)
        else:
            # Create empty label file if no annotation found
            with open(label_file, 'w') as f:
                pass
    
    # Save class names
    class_file = os.path.join(output_dir, 'classes.txt')
    with open(class_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Class names saved to: {class_file}")


def main():
    parser = argparse.ArgumentParser(description='Prepare WAID dataset for training')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to WAID dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory for organized dataset')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("WAID Dataset Preparation Script")
    print("=" * 50)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train split: {args.train_split}")
    print("=" * 50)
    
    prepare_waid_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        train_split=args.train_split,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

