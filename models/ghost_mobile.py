"""
Ghost-Mobile Backbone: Lightweight feature extraction using Ghost blocks
and MobileNet-style depthwise separable convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostModule(nn.Module):
    """
    Ghost Convolution Module
    Generates more feature maps from fewer primary convolutions
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 ratio=2, dw_size=3, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, 
                     kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, 
                     dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class LinearBottleneck(nn.Module):
    """
    MobileNet-style Linear Bottleneck with depthwise separable convolution
    """
    
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(LinearBottleneck, self).__init__()
        hidden_dim = in_channels * expansion
        
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Pointwise projection
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GhostMobileBackbone(nn.Module):
    """
    Ghost-Mobile Backbone combining Ghost blocks and Linear Bottlenecks
    Extracts multi-scale features at 152×152, 76×76, and 38×38 resolutions
    """
    
    def __init__(self, in_channels=12):
        super(GhostMobileBackbone, self).__init__()
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Bottleneck blocks with increasing channels
        # Block 1: 152×152×16
        self.block1 = LinearBottleneck(16, 16, stride=1, expansion=6)
        
        # Block 2: 76×76×24 (downsample)
        self.block2 = nn.Sequential(
            LinearBottleneck(16, 24, stride=2, expansion=6),
            LinearBottleneck(24, 24, stride=1, expansion=6)
        )
        
        # Block 3: 38×38×32 (downsample)
        self.block3 = nn.Sequential(
            LinearBottleneck(24, 32, stride=2, expansion=6),
            LinearBottleneck(32, 32, stride=1, expansion=6)
        )
        
        # Block 4: 19×19×64
        self.block4 = nn.Sequential(
            LinearBottleneck(32, 64, stride=2, expansion=6),
            LinearBottleneck(64, 64, stride=1, expansion=6),
            LinearBottleneck(64, 64, stride=1, expansion=6)
        )
        
        # Block 5: 10×10×96
        self.block5 = nn.Sequential(
            LinearBottleneck(64, 96, stride=2, expansion=6),
            LinearBottleneck(96, 96, stride=1, expansion=6),
            LinearBottleneck(96, 96, stride=1, expansion=6)
        )
        
        # Block 6: 5×5×160
        self.block6 = nn.Sequential(
            LinearBottleneck(96, 160, stride=2, expansion=6),
            LinearBottleneck(160, 160, stride=1, expansion=6),
            LinearBottleneck(160, 160, stride=1, expansion=6)
        )
        
        # Block 7: 3×3×320
        self.block7 = nn.Sequential(
            LinearBottleneck(160, 320, stride=2, expansion=6),
            LinearBottleneck(320, 320, stride=1, expansion=6)
        )
        
        # Final projection to 10×10×1280
        self.final_conv = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # Feature extraction layers for multi-scale outputs
        self.feature_layers = nn.ModuleDict({
            'p3': nn.Identity(),  # 152×152×16 (from block1)
            'p4': nn.Identity(),  # 76×76×24 (from block2)
            'p5': nn.Identity()   # 38×38×32 (from block3)
        })
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 12, 304, 304)
        Returns:
            Dictionary with multi-scale feature maps:
            - p3: (B, 16, 152, 152)
            - p4: (B, 24, 76, 76)
            - p5: (B, 32, 38, 38)
        """
        x = self.stem(x)  # (B, 16, 152, 152)
        
        # Extract P3 (152×152)
        p3 = self.block1(x)  # (B, 16, 152, 152)
        
        # Extract P4 (76×76)
        p4 = self.block2(p3)  # (B, 24, 76, 76)
        
        # Extract P5 (38×38)
        p5 = self.block3(p4)  # (B, 32, 38, 38)
        
        # Continue through remaining blocks
        x = self.block4(p5)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.final_conv(x)
        
        return {
            'p3': self.feature_layers['p3'](p3),
            'p4': self.feature_layers['p4'](p4),
            'p5': self.feature_layers['p5'](p5)
        }

