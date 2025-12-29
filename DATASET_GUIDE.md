# WAID Dataset Preparation Guide

## Dataset Information

**WAID (Wildlife Aerial Images from Drone)** is a large-scale dataset for wildlife detection with drones, published in Applied Sciences.

- **Official Repository**: [https://github.com/xiaohuicui/WAID](https://github.com/xiaohuicui/WAID)
- **Total Images**: 14,375 annotated aerial images
- **Classes**: 6 animal classes (cattle, sheep, seal, camelus, zebra, kiang)
- **Format**: Aerial drone imagery with bounding box annotations

## Download Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/xiaohuicui/WAID.git
   cd WAID
   ```

2. **Check Dataset Structure**:
   The dataset may be organized in different formats. Common structures include:
   ```
   WAID/
   ├── images/              # All images
   ├── annotations/         # Annotation files (JSON or YOLO format)
   └── README.md            # Dataset documentation
   ```

## Dataset Preparation

### Option 1: Using the Preparation Script (Recommended)

The provided `prepare_dataset.py` script automates the dataset organization:

```bash
python prepare_dataset.py \
    --dataset_path /path/to/WAID \
    --output_dir ./dataset \
    --train_split 0.8
```

**Parameters:**
- `--dataset_path`: Path to the WAID dataset root directory
- `--output_dir`: Output directory for organized dataset (default: `./dataset`)
- `--train_split`: Train/validation split ratio (default: 0.8)
- `--seed`: Random seed for reproducibility (default: 42)

### Option 2: Manual Preparation

If the script doesn't work with your dataset format, manually organize as follows:

1. **Create Directory Structure**:
   ```bash
   mkdir -p dataset/train/images
   mkdir -p dataset/train/labels
   mkdir -p dataset/val/images
   mkdir -p dataset/val/labels
   ```

2. **Split Images**:
   - Use 80% for training, 20% for validation
   - Copy images to respective `images/` directories

3. **Prepare Labels**:
   - Convert annotations to YOLO format if needed
   - Each label file should match the image filename (e.g., `image001.jpg` → `image001.txt`)
   - Format: `class_id x_center y_center width height` (all normalized to [0, 1])

## YOLO Label Format

Each label file (`.txt`) should contain one line per object:

```
class_id x_center y_center width height
```

**Example:**
```
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25
```

Where:
- `class_id`: Class index (0-5)
- `x_center`: Normalized x-coordinate of box center (0-1)
- `y_center`: Normalized y-coordinate of box center (0-1)
- `width`: Normalized box width (0-1)
- `height`: Normalized box height (0-1)

## Class Mapping

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | cattle | Cattle/Bovine animals |
| 1 | sheep | Sheep |
| 2 | seal | Seals |
| 3 | camelus | Camels |
| 4 | zebra | Zebras |
| 5 | kiang | Kiangs (wild ass) |

## Verification

After preparation, verify your dataset structure:

```bash
# Check directory structure
tree dataset/ -L 3

# Count files
find dataset/train/images -type f | wc -l
find dataset/train/labels -type f | wc -l
find dataset/val/images -type f | wc -l
find dataset/val/labels -type f | wc -l
```

Expected structure:
```
dataset/
├── train/
│   ├── images/          # Training images (e.g., 11,500 images)
│   └── labels/          # Training labels (same number of .txt files)
├── val/
│   ├── images/          # Validation images (e.g., 2,875 images)
│   └── labels/          # Validation labels (same number of .txt files)
└── classes.txt          # Class names (created by script)
```

## Troubleshooting

### Issue: Script can't find images
- **Solution**: Check the dataset path and ensure images are in a subdirectory named `images/` or `WAID/images/`

### Issue: Annotations not found
- **Solution**: Check if annotations are in JSON or YOLO format. The script supports both, but you may need to adapt the conversion logic based on the actual WAID format.

### Issue: Label files don't match images
- **Solution**: Ensure label filenames match image filenames exactly (except extension: `.jpg` → `.txt`)

### Issue: Wrong label format
- **Solution**: Convert annotations to YOLO format manually. The format requires normalized coordinates (0-1 range).

## Citation

If you use the WAID dataset, please cite the original publication:

```
WAID: A Large-Scale Dataset for Wildlife Detection with Drones
Published in Applied Sciences
GitHub: https://github.com/xiaohuicui/WAID
```

## Additional Resources

- Check the WAID repository README for specific download instructions
- Contact the dataset authors if you encounter format issues
- Refer to the main README.md for training instructions after dataset preparation

