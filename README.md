# MobileYOLO-Lite: A Compact Detection Framework for Small Animals in Drone-Based Wildlife Monitoring

This repository contains the Python implementation of MobileYOLO-Lite, a lightweight object detection framework designed for detecting small animals in drone imagery.

## Overview

MobileYOLO-Lite is specifically designed to address the challenges of:
- **Small object detection**: Animals appearing at very low resolution (< 20 pixels)
- **Resource constraints**: Deployment on embedded systems with limited memory and compute
- **Complex backgrounds**: Dense vegetation, shadows, and natural terrain
- **Real-time inference**: Fast processing for continuous drone monitoring

## Architecture

The framework consists of four main modules:

1. **PatchPack**: Reshapes 608×608×3 images into 304×304×12 tensors without convolution, preserving local detail
2. **Ghost-Mobile Backbone**: Lightweight feature extraction using Ghost blocks and MobileNet-style bottlenecks
3. **Lite-BiDFPN**: Multi-scale feature fusion using weighted addition instead of concatenation
4. **Sparse Tiny-Head**: Detection head with attention gates that focus computation on animal-like regions

Additionally, the **SGDM-AQS** optimizer provides stable training under sparse gradients with adaptive quantization.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Implimentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the WAID dataset:
   - Visit the [WAID GitHub repository](https://github.com/xiaohuicui/WAID)
   - Download or clone the dataset
   - The dataset contains 14,375 annotated aerial images with 6 animal classes

## Dataset Preparation

### Download WAID Dataset

The code is designed to work with the **WAID (Wildlife Aerial Images from Drone)** dataset.

**Dataset Repository**: [https://github.com/xiaohuicui/WAID](https://github.com/xiaohuicui/WAID)

The WAID dataset is published in Applied Sciences and contains 14,375 annotated aerial images with 6 animal classes.

### Dataset Structure

After downloading and preparing the dataset, organize it in the following structure:

```
dataset/
├── train/
│   ├── images/          # Training images
│   └── labels/          # YOLO-format labels
└── val/
    ├── images/          # Validation images
    └── labels/          # YOLO-format labels
```

### Label Format

Each label file should be in YOLO format with normalized coordinates:
```
class_id x_center y_center width height
```

**Class IDs:**
- 0: cattle
- 1: sheep
- 2: seal
- 3: camelus
- 4: zebra
- 5: kiang

### Dataset Preparation Script

Use the provided `prepare_dataset.py` script to help organize the WAID dataset:

```bash
python prepare_dataset.py --dataset_path /path/to/WAID --output_dir ./dataset --train_split 0.8
```

For detailed dataset preparation instructions, see [DATASET_GUIDE.md](DATASET_GUIDE.md).

## Training

Train the model using:

```bash
python train.py \
    --data_dir /path/to/dataset \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001 \
    --num_classes 6 \
    --input_size 608 \
    --save_dir ./checkpoints
```

### Training Parameters

- `--data_dir`: Root directory containing train/val splits
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--num_classes`: Number of classes (default: 6)
- `--input_size`: Input image size (default: 608)
- `--device`: Device to use (cuda/cpu, default: cuda)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--resume`: Path to checkpoint to resume training from

## Inference

Run inference on a single image:

```bash
python inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --image /path/to/image.jpg \
    --output output.jpg \
    --conf_threshold 0.5 \
    --nms_threshold 0.5
```

### Inference Parameters

- `--checkpoint`: Path to trained model checkpoint
- `--image`: Path to input image
- `--output`: Path to save output image with detections
- `--conf_threshold`: Confidence threshold for filtering detections (default: 0.5)
- `--nms_threshold`: IoU threshold for Non-Maximum Suppression (default: 0.5)



## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── mobile_yolo_lite.py    # Main model
│   ├── patchpack.py            # Patch reshaping module
│   ├── ghost_mobile.py         # Ghost-Mobile backbone
│   ├── lite_bidfpn.py          # Lite-BiDFPN neck
│   ├── sparse_head.py          # Sparse detection head
│   └── sgdm_aqs.py             # SGDM-AQS optimizer
├── utils/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loader
│   ├── loss.py                 # Loss functions
│   └── postprocess.py          # Post-processing utilities
├── train.py                    # Training script
├── inference.py                # Inference script
├── prepare_dataset.py           # Dataset preparation script
├── test_model.py               # Model testing script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Key Features

1. **PatchPack Module**: Preserves spatial detail without convolution overhead
2. **Ghost Convolution**: Reduces FLOPs by generating feature maps from cheap operations
3. **Weighted Feature Fusion**: Lowers memory usage compared to concatenation
4. **Sparse Attention**: Focuses computation on relevant regions
5. **Dual-Scale Detection**: Handles both tiny (152×152) and medium-small (76×76) objects
6. **SGDM-AQS Optimizer**: Stable training with adaptive quantization

## Citation

If you use this code in your research, please cite:

```
MobileYOLO-Lite: A Compact and Detail-Preserving Detection Framework 
for Small Animals in Drone-Based Wildlife Monitoring
M. Mangaleswaran
SRM Institute of Science and Technology
```

## License

This project, **MobileYOLO-Lite: A Compact and Detail-Preserving Detection Framework for Small Animals in Drone-Based Wildlife Monitoring**, is intended **only for academic and research purposes**.

Commercial use, redistribution for profit, or deployment in commercial products is **not permitted** without prior written permission from the authors or the authorized institution.

Users may study, modify, and extend the code for non-commercial research, provided that proper citation of the original work is maintained.

For commercial licensing or permission requests, please contact the corresponding author.

## Contact

For questions or issues, please contact:
- Email: mmangaleswaran479@gmail.com
- Email: mm0103@srmist.edu.in

