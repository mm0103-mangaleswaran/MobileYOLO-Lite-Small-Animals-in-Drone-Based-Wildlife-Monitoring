"""
PatchPack Module: Reshapes 2×2 patches into channel-rich tensors
Converts 608×608×3 input to 304×304×12 output without convolution
"""

import torch
import torch.nn as nn


class PatchPack(nn.Module):
    """
    Patch reshaping module that divides input into 2×2 patches
    and rearranges them along the channel dimension.
    
    Input: (B, 3, 608, 608)
    Output: (B, 12, 304, 304)
    """
    
    def __init__(self, patch_size=2):
        super(PatchPack, self).__init__()
        self.patch_size = patch_size
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Reshaped tensor of shape (B, C*patch_size^2, H//patch_size, W//patch_size)
        """
        B, C, H, W = x.shape
        
        # Ensure dimensions are divisible by patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input dimensions ({H}, {W}) must be divisible by patch_size ({self.patch_size})"
        
        # Reshape to extract patches
        # (B, C, H, W) -> (B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.view(B, C, H // self.patch_size, self.patch_size, 
                   W // self.patch_size, self.patch_size)
        
        # Rearrange: (B, C, H//patch_size, W//patch_size, patch_size, patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        
        # Flatten patch dimensions into channels
        # (B, C, H//patch_size, W//patch_size, patch_size^2)
        x = x.view(B, C, H // self.patch_size, W // self.patch_size, 
                   self.patch_size * self.patch_size)
        
        # Rearrange to (B, C*patch_size^2, H//patch_size, W//patch_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(B, C * self.patch_size * self.patch_size, 
                   H // self.patch_size, W // self.patch_size)
        
        return x

