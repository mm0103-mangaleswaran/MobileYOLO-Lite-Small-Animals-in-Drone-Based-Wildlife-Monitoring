"""
Dataset utilities for WAID dataset
Handles loading and preprocessing of drone images
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import json
import cv2


class WAIDDataset(Dataset):
    """
    WAID (Wildlife Aerial Image Dataset) loader
    Supports YOLO format annotations
    """
    
    def __init__(self, image_dir, label_dir, input_size=608, augment=True):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            input_size: Input image size (608x608)
            augment: Whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.augment = augment
        
        # Load image and label files
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        self.num_classes = 6  # cattle, sheep, seal, camelus, zebra, kiang
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load labels
        label_path = os.path.join(self.label_dir, 
                                 self.image_files[idx].replace('.jpg', '.txt')
                                                      .replace('.jpeg', '.txt')
                                                      .replace('.png', '.txt'))
        
        boxes = []
        classes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to absolute coordinates
                        img_h, img_w = image.shape[:2]
                        x1 = (x_center - width / 2) * img_w
                        y1 = (y_center - height / 2) * img_h
                        x2 = (x_center + width / 2) * img_w
                        y2 = (y_center + height / 2) * img_h
                        
                        boxes.append([x1, y1, x2, y2])
                        classes.append(cls)
        
        # Data augmentation
        if self.augment:
            image, boxes = self._augment(image, boxes)
        
        # Resize and normalize
        image, boxes = self._resize_and_normalize(image, boxes)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert boxes to tensor
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.long)
        
        return image, boxes, classes
    
    def _augment(self, image, boxes):
        """Apply data augmentation"""
        # Horizontal flip
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            if len(boxes) > 0:
                boxes = np.array(boxes)
                img_w = image.shape[1]
                boxes[:, [0, 2]] = img_w - boxes[:, [2, 0]]
                boxes = boxes.tolist()
        
        # Color jitter
        if np.random.random() < 0.5:
            image = self._color_jitter(image)
        
        return image, boxes
    
    def _color_jitter(self, image):
        """Apply color jitter"""
        # Brightness
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        # Contrast
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        return image
    
    def _resize_and_normalize(self, image, boxes):
        """Resize image to input_size and adjust boxes"""
        h, w = image.shape[:2]
        
        # Calculate scaling
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h))
        
        # Pad to input_size
        new_image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        new_image[:new_h, :new_w] = image
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes = np.array(boxes)
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            boxes = boxes.tolist()
        
        return new_image, boxes


def collate_fn(batch):
    """Custom collate function for variable-length boxes"""
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    classes = [item[2] for item in batch]
    return images, boxes, classes

