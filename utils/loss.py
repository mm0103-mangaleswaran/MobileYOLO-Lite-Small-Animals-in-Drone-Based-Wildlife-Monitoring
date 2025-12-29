"""
Loss functions for MobileYOLO-Lite training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    """
    YOLO loss function combining:
    - Bounding box regression (MSE)
    - Objectness loss (BCE)
    - Classification loss (BCE)
    """
    
    def __init__(self, num_classes=6, num_anchors=3, lambda_coord=5.0, 
                 lambda_noobj=0.5, lambda_obj=1.0, lambda_cls=1.0):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets, grid_size):
        """
        Args:
            predictions: Raw predictions (B, num_anchors*(5+num_classes), H, W)
            targets: List of target boxes and classes for each image
            grid_size: Grid resolution (152 or 76)
        Returns:
            Total loss
        """
        B, C, H, W = predictions.shape
        
        # Reshape predictions
        predictions = predictions.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract components
        pred_xy = torch.sigmoid(predictions[..., 0:2])
        pred_wh = predictions[..., 2:4]
        pred_obj = predictions[..., 4:5]
        pred_cls = predictions[..., 5:]
        
        # Initialize loss components
        loss_coord = 0.0
        loss_obj = 0.0
        loss_noobj = 0.0
        loss_cls = 0.0
        
        # Process each image in batch
        for b in range(B):
            target_boxes = targets['boxes'][b]  # (N, 4) in [x1, y1, x2, y2]
            target_classes = targets['classes'][b]  # (N,)
            
            if len(target_boxes) == 0:
                # No objects: only noobj loss
                loss_noobj += self.lambda_noobj * torch.sigmoid(pred_obj[b]).mean()
                continue
            
            # Convert target boxes to grid coordinates
            stride = 608.0 / grid_size
            target_xy = torch.zeros((len(target_boxes), 2), device=pred_xy.device)
            target_wh = torch.zeros((len(target_boxes), 2), device=pred_wh.device)
            
            for i, box in enumerate(target_boxes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                # Grid coordinates
                grid_x = int(center_x / stride)
                grid_y = int(center_y / stride)
                grid_x = max(0, min(W - 1, grid_x))
                grid_y = max(0, min(H - 1, grid_y))
                
                # Normalized coordinates
                target_xy[i, 0] = (center_x / stride) - grid_x
                target_xy[i, 1] = (center_y / stride) - grid_y
                target_wh[i, 0] = width / stride
                target_wh[i, 1] = height / stride
            
            # Calculate IoU with predictions to find best anchor
            # Simplified: use first anchor for each target
            for i in range(len(target_boxes)):
                grid_x = int((target_boxes[i, 0] + target_boxes[i, 2]) / 2.0 / stride)
                grid_y = int((target_boxes[i, 1] + target_boxes[i, 3]) / 2.0 / stride)
                grid_x = max(0, min(W - 1, grid_x))
                grid_y = max(0, min(H - 1, grid_y))
                
                anchor_idx = 0  # Use first anchor
                
                # Coordinate loss
                coord_loss = self.mse_loss(
                    pred_xy[b, anchor_idx, grid_y, grid_x],
                    target_xy[i]
                ).sum()
                loss_coord += self.lambda_coord * coord_loss
                
                # Width/height loss
                wh_loss = self.mse_loss(
                    pred_wh[b, anchor_idx, grid_y, grid_x],
                    torch.log(target_wh[i] + 1e-8)
                ).sum()
                loss_coord += self.lambda_coord * wh_loss
                
                # Objectness loss (positive)
                obj_target = torch.ones_like(pred_obj[b, anchor_idx, grid_y, grid_x, 0])
                obj_loss = F.binary_cross_entropy_with_logits(
                    pred_obj[b, anchor_idx, grid_y, grid_x, 0],
                    obj_target
                )
                loss_obj += self.lambda_obj * obj_loss
                
                # Classification loss
                cls_target = F.one_hot(target_classes[i], self.num_classes).float()
                cls_loss = F.binary_cross_entropy_with_logits(
                    pred_cls[b, anchor_idx, grid_y, grid_x],
                    cls_target
                ).sum()
                loss_cls += self.lambda_cls * cls_loss
        
        # No-object loss for grid cells without targets
        # Create mask excluding cells with objects
        noobj_mask = torch.ones(B, self.num_anchors, H, W, device=pred_obj.device, dtype=torch.bool)
        
        # Mark cells with objects as False
        for b in range(B):
            target_boxes = targets['boxes'][b]
            if len(target_boxes) > 0:
                stride = 608.0 / grid_size
                for box in target_boxes:
                    center_x = (box[0] + box[2]) / 2.0
                    center_y = (box[1] + box[3]) / 2.0
                    grid_x = int(center_x / stride)
                    grid_y = int(center_y / stride)
                    grid_x = max(0, min(W - 1, grid_x))
                    grid_y = max(0, min(H - 1, grid_y))
                    noobj_mask[b, :, grid_y, grid_x] = False
        
        # Calculate noobj loss only for cells without objects
        if noobj_mask.any():
            loss_noobj += self.lambda_noobj * F.binary_cross_entropy_with_logits(
                pred_obj[noobj_mask],
                torch.zeros_like(pred_obj[noobj_mask]),
                reduction='mean'
            )
        
        total_loss = loss_coord + loss_obj + loss_noobj + loss_cls
        return total_loss / B  # Average over batch

