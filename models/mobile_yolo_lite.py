"""
MobileYOLO-Lite: Complete detection framework
Combines PatchPack, Ghost-Mobile Backbone, Lite-BiDFPN, and Sparse Tiny-Head
"""

import torch
import torch.nn as nn
from .patchpack import PatchPack
from .ghost_mobile import GhostMobileBackbone
from .lite_bidfpn import LiteBiDFPN
from .sparse_head import SparseTinyHead


class MobileYOLOLite(nn.Module):
    """
    MobileYOLO-Lite: Compact detection framework for small animals in drone imagery
    
    Architecture:
    1. PatchPack: Reshapes 608×608×3 -> 304×304×12
    2. Ghost-Mobile Backbone: Extracts multi-scale features
    3. Lite-BiDFPN: Fuses features with weighted addition
    4. Sparse Tiny-Head: Detects objects with attention gates
    """
    
    def __init__(self, num_classes=6, input_size=608, num_anchors=3):
        super(MobileYOLOLite, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_anchors = num_anchors
        
        # Patch reshaping module
        self.patchpack = PatchPack(patch_size=2)
        
        # Ghost-Mobile backbone
        self.backbone = GhostMobileBackbone(in_channels=12)
        
        # Lite-BiDFPN neck
        self.neck = LiteBiDFPN(
            in_channels_list=[16, 24, 32],
            out_channels=64
        )
        
        # Sparse detection head
        self.head = SparseTinyHead(
            in_channels=64,
            num_classes=num_classes,
            num_anchors=num_anchors
        )
    
    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, 608, 608)
        Returns:
            Dictionary with predictions and intermediate features
        """
        # Patch reshaping
        x = self.patchpack(x)  # (B, 12, 304, 304)
        
        # Backbone feature extraction
        backbone_features = self.backbone(x)  # p3, p4, p5
        
        # Feature fusion
        fused_features = self.neck(backbone_features)  # p3, p4, p5
        
        # Detection
        predictions = self.head(fused_features)
        
        return predictions
    
    def decode_predictions(self, predictions, conf_threshold=0.5, nms_threshold=0.5):
        """
        Decode raw predictions to bounding boxes
        
        Args:
            predictions: Dictionary with 'pred_152' and 'pred_76'
            conf_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
        Returns:
            List of detections: [boxes, scores, classes]
        """
        from utils.postprocess import decode_yolo_output, nms
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Process 152×152 predictions
        pred_152 = predictions['pred_152']
        boxes_152, scores_152, classes_152 = decode_yolo_output(
            pred_152, grid_size=152, conf_threshold=conf_threshold
        )
        
        # Process 76×76 predictions
        pred_76 = predictions['pred_76']
        boxes_76, scores_76, classes_76 = decode_yolo_output(
            pred_76, grid_size=76, conf_threshold=conf_threshold
        )
        
        # Combine predictions
        if len(boxes_152) > 0 and len(boxes_76) > 0:
            all_boxes = torch.cat([boxes_152, boxes_76], dim=0)
            all_scores = torch.cat([scores_152, scores_76], dim=0)
            all_classes = torch.cat([classes_152, classes_76], dim=0)
        elif len(boxes_152) > 0:
            all_boxes = boxes_152
            all_scores = scores_152
            all_classes = classes_152
        elif len(boxes_76) > 0:
            all_boxes = boxes_76
            all_scores = scores_76
            all_classes = classes_76
        else:
            # No detections
            device = predictions['pred_152'].device
            all_boxes = torch.zeros((0, 4), device=device)
            all_scores = torch.zeros((0,), device=device)
            all_classes = torch.zeros((0,), dtype=torch.long, device=device)
            return all_boxes, all_scores, all_classes
        
        # Apply NMS
        if len(all_boxes) > 0:
            keep = nms(all_boxes, all_scores, nms_threshold)
            all_boxes = all_boxes[keep]
            all_scores = all_scores[keep]
            all_classes = all_classes[keep]
        
        return all_boxes, all_scores, all_classes

