"""
Post-processing utilities for YOLO predictions
Includes decoding, NMS, and visualization
"""

import torch
import torch.nn.functional as F
import numpy as np


def decode_yolo_output(predictions, grid_size, num_anchors=3, num_classes=6, 
                       conf_threshold=0.5, input_size=608):
    """
    Decode YOLO output predictions to bounding boxes
    
    Args:
        predictions: Raw predictions (B, num_anchors*(5+num_classes), H, W)
        grid_size: Grid resolution (152 or 76)
        num_anchors: Number of anchor boxes
        num_classes: Number of classes
        conf_threshold: Confidence threshold
        input_size: Input image size
    Returns:
        boxes: (N, 4) bounding boxes in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        classes: (N,) class indices
    """
    B, C, H, W = predictions.shape
    
    # Reshape predictions
    predictions = predictions.view(B, num_anchors, 5 + num_classes, H, W)
    predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
    
    # Extract components
    xy = torch.sigmoid(predictions[..., 0:2])  # Center coordinates
    wh = torch.exp(predictions[..., 2:4])      # Width and height
    obj_conf = torch.sigmoid(predictions[..., 4:5])  # Objectness
    cls_conf = torch.sigmoid(predictions[..., 5:])   # Class probabilities
    
    # Generate grid coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=predictions.device),
        torch.arange(W, device=predictions.device),
        indexing='ij'
    )
    grid_x = grid_x.float().unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.float().unsqueeze(0).unsqueeze(0)
    
    # Calculate absolute coordinates
    stride = input_size / grid_size
    pred_x = (xy[..., 0] + grid_x) * stride
    pred_y = (xy[..., 1] + grid_y) * stride
    pred_w = wh[..., 0] * stride
    pred_h = wh[..., 1] * stride
    
    # Convert to [x1, y1, x2, y2] format
    boxes = torch.stack([
        pred_x - pred_w / 2,
        pred_y - pred_h / 2,
        pred_x + pred_w / 2,
        pred_y + pred_h / 2
    ], dim=-1)
    
    # Calculate final confidence
    cls_scores, cls_indices = torch.max(cls_conf, dim=-1)
    final_conf = obj_conf.squeeze(-1) * cls_scores
    
    # Filter by confidence threshold
    mask = final_conf > conf_threshold
    
    # Flatten and filter
    boxes = boxes[mask].view(-1, 4)
    scores = final_conf[mask].view(-1)
    classes = cls_indices[mask].view(-1)
    
    return boxes, scores, classes


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: (N, 4) boxes in [x1, y1, x2, y2] format
        box2: (M, 4) boxes in [x1, y1, x2, y2] format
    Returns:
        iou: (N, M) IoU matrix
    """
    # Calculate intersection
    x1 = torch.max(box1[:, 0:1], box2[:, 0].unsqueeze(0))
    y1 = torch.max(box1[:, 1:2], box2[:, 1].unsqueeze(0))
    x2 = torch.min(box1[:, 2:3], box2[:, 2].unsqueeze(0))
    y2 = torch.min(box1[:, 3:4], box2[:, 3].unsqueeze(0))
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Calculate union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)
    return iou


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    
    Args:
        boxes: (N, 4) bounding boxes
        scores: (N,) confidence scores
        iou_threshold: IoU threshold for suppression
    Returns:
        keep: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Sort by scores
    _, order = scores.sort(descending=True)
    
    keep = []
    while len(order) > 0:
        # Keep the box with highest score
        i = order[0]
        keep.append(i.item())
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        iou = calculate_iou(boxes[i:i+1], boxes[order[1:]])
        
        # Remove boxes with high IoU
        mask = iou.squeeze(0) < iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

