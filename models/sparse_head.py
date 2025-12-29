"""
Sparse Tiny-Head: Detection head with sparse attention gates
Focuses computation on regions with animal-like patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttentionGate(nn.Module):
    """
    Sparse attention gate that filters background regions
    Returns binary mask to conditionally apply convolution
    """
    
    def __init__(self, in_channels, threshold=0.5):
        super(SparseAttentionGate, self).__init__()
        self.threshold = threshold
        
        # Lightweight attention network
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            Binary attention mask (B, 1, H, W)
        """
        attention = self.attention_net(x)
        # Binarize attention
        mask = (attention > self.threshold).float()
        return mask


class GatedConv2d(nn.Module):
    """
    Gated convolution that applies operation only where attention is active
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, mask):
        """
        Args:
            x: Input features (B, C, H, W)
            mask: Attention mask (B, 1, H, W)
        Returns:
            Gated convolution output
        """
        # Apply convolution only where mask is active
        out = self.conv(x * mask)
        out = self.bn(out)
        out = self.activation(out)
        return out


class TinyHead(nn.Module):
    """
    Sparse Tiny-Head for small animal detection
    Uses dual-scale output grids (152×152 and 76×76)
    """
    
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(TinyHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Sparse attention gate
        self.attention_gate = SparseAttentionGate(in_channels)
        
        # Gated convolution layers
        self.gated_conv = GatedConv2d(in_channels, in_channels, 3, 1, 1)
        
        # Output projection: (x, y, w, h, objectness, class_scores)
        # 5 = bbox (4) + objectness (1)
        self.output_conv = nn.Conv2d(
            in_channels, 
            num_anchors * (5 + num_classes), 
            1, 1, 0
        )
    
    def forward(self, x):
        """
        Args:
            x: Input feature map (B, C, H, W)
        Returns:
            Detection predictions (B, num_anchors*(5+num_classes), H, W)
        """
        # Generate attention mask
        mask = self.attention_gate(x)
        
        # Apply gated convolution
        x = self.gated_conv(x, mask)
        
        # Generate predictions
        predictions = self.output_conv(x)
        
        return predictions, mask


class SparseTinyHead(nn.Module):
    """
    Complete sparse detection head with dual-scale outputs
    Processes 152×152 and 76×76 feature maps
    """
    
    def __init__(self, in_channels=64, num_classes=6, num_anchors=3):
        super(SparseTinyHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Two detection heads for different scales
        self.head_152 = TinyHead(in_channels, num_classes, num_anchors)
        self.head_76 = TinyHead(in_channels, num_classes, num_anchors)
    
    def forward(self, features):
        """
        Args:
            features: Dictionary with 'p3' (152×152) and 'p4' (76×76)
        Returns:
            Dictionary with predictions and attention masks
        """
        # Process 152×152 features (tiny objects)
        pred_152, mask_152 = self.head_152(features['p3'])
        
        # Process 76×76 features (medium-small objects)
        pred_76, mask_76 = self.head_76(features['p4'])
        
        return {
            'pred_152': pred_152,  # (B, num_anchors*(5+num_classes), 152, 152)
            'pred_76': pred_76,    # (B, num_anchors*(5+num_classes), 76, 76)
            'mask_152': mask_152,
            'mask_76': mask_76
        }

