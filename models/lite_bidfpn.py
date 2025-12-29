"""
Lite-BiDFPN: Lightweight Bidirectional Feature Pyramid Network
Uses weighted fusion instead of concatenation to reduce memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFusion(nn.Module):
    """
    Weighted feature fusion using learnable scalar weights
    Avoids concatenation to reduce memory footprint
    """
    
    def __init__(self, num_inputs=2, epsilon=1e-4):
        super(WeightedFusion, self).__init__()
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs) / num_inputs)
    
    def forward(self, features):
        """
        Args:
            features: List of feature tensors to fuse
        Returns:
            Fused feature tensor
        """
        # Normalize weights
        weights = F.relu(self.weights)
        weights = weights / (weights.sum() + self.epsilon)
        
        # Weighted sum
        fused = sum(w * f for w, f in zip(weights, features))
        return fused


class LiteBiDFPN(nn.Module):
    """
    Lite-BiDFPN: Lightweight bidirectional feature pyramid network
    Performs top-down and bottom-up fusion with weighted addition
    """
    
    def __init__(self, in_channels_list=[16, 24, 32], out_channels=64):
        super(LiteBiDFPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # Input projection layers
        self.input_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        # Top-down fusion (high-res to low-res)
        self.top_down_fusion = nn.ModuleList([
            WeightedFusion(num_inputs=2) for _ in range(len(in_channels_list) - 1)
        ])
        
        # Bottom-up fusion (low-res to high-res)
        self.bottom_up_fusion = nn.ModuleList([
            WeightedFusion(num_inputs=2) for _ in range(len(in_channels_list) - 1)
        ])
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='nearest')
            for _ in range(len(in_channels_list) - 1)
        ])
        
        # Downsampling layers
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list) - 1)
        ])
    
    def forward(self, features):
        """
        Args:
            features: Dictionary with keys 'p3', 'p4', 'p5'
                    p3: (B, 16, 152, 152)
                    p4: (B, 24, 76, 76)
                    p5: (B, 32, 38, 38)
        Returns:
            Dictionary with refined features at same resolutions
        """
        # Project inputs to common channel dimension
        p3 = self.input_layers[0](features['p3'])  # (B, 64, 152, 152)
        p4 = self.input_layers[1](features['p4'])  # (B, 64, 76, 76)
        p5 = self.input_layers[2](features['p5'])  # (B, 64, 38, 38)
        
        # Top-down path: P5 -> P4 -> P3
        # P5 to P4
        p5_up = self.upsample_layers[1](p5)  # (B, 64, 76, 76)
        p4_td = self.top_down_fusion[1]([p4, p5_up])  # (B, 64, 76, 76)
        
        # P4 to P3
        p4_td_up = self.upsample_layers[0](p4_td)  # (B, 64, 152, 152)
        p3_td = self.top_down_fusion[0]([p3, p4_td_up])  # (B, 64, 152, 152)
        
        # Bottom-up path: P3 -> P4 -> P5
        # P3 to P4
        p3_bu_down = self.downsample_layers[0](p3_td)  # (B, 64, 76, 76)
        p4_bu = self.bottom_up_fusion[0]([p4_td, p3_bu_down])  # (B, 64, 76, 76)
        
        # P4 to P5
        p4_bu_down = self.downsample_layers[1](p4_bu)  # (B, 64, 38, 38)
        p5_bu = self.bottom_up_fusion[1]([p5, p4_bu_down])  # (B, 64, 38, 38)
        
        return {
            'p3': p3_td,  # (B, 64, 152, 152)
            'p4': p4_bu,  # (B, 64, 76, 76)
            'p5': p5_bu   # (B, 64, 38, 38)
        }

