"""
SGDM-AQS Optimizer: Sparse Gradient Descent with Momentum and Adaptive Quantization
Stabilizes training under sparse gradients and supports mixed-precision quantization
"""

import torch
import torch.optim as optim
from torch.optim import Optimizer
import numpy as np


class SGDM_AQS(Optimizer):
    """
    SGDM-AQS: Sparse Gradient Descent with Momentum and Adaptive Quantization Strategy
    Features:
    - Dual momentum buffers (short-term and long-term)
    - Binary gradient gates to filter sparse/noisy gradients
    - Adaptive bit-width quantization based on activation variance
    """
    
    def __init__(self, params, lr=0.001, momentum_short=0.9, momentum_long=0.99,
                 weight_decay=0.0001, epsilon=1e-8, gate_threshold=0.01,
                 quantize=True, bit_width_low=4, bit_width_high=8):
        defaults = dict(
            lr=lr,
            momentum_short=momentum_short,
            momentum_long=momentum_long,
            weight_decay=weight_decay,
            epsilon=epsilon,
            gate_threshold=gate_threshold,
            quantize=quantize,
            bit_width_low=bit_width_low,
            bit_width_high=bit_width_high
        )
        super(SGDM_AQS, self).__init__(params, defaults)
    
    def _get_activation_variance(self, param):
        """
        Estimate activation variance for adaptive quantization
        Uses parameter statistics as proxy for activation variance
        """
        if param.grad is None:
            return 0.0
        # Use gradient variance as proxy for activation variance
        grad_var = torch.var(param.grad).item()
        return grad_var
    
    def _quantize_weights(self, param, bit_width):
        """
        Quantize weights to specified bit width
        """
        if bit_width >= 8:
            return param  # No quantization for high bit-width
        
        # Quantization range
        qmin = -(2 ** (bit_width - 1))
        qmax = 2 ** (bit_width - 1) - 1
        
        # Scale factor
        scale = (param.max() - param.min()) / (qmax - qmin)
        scale = max(scale, 1e-8)
        
        # Quantize
        quantized = torch.clamp(
            torch.round(param / scale),
            qmin, qmax
        )
        
        # Dequantize
        dequantized = quantized * scale
        return dequantized
    
    def _apply_gradient_gate(self, grad, threshold):
        """
        Apply binary gate to filter sparse/noisy gradients
        """
        # Compute gradient magnitude
        grad_magnitude = torch.abs(grad)
        
        # Binary gate: keep gradients above threshold
        gate = (grad_magnitude > threshold).float()
        
        # Apply gate
        gated_grad = grad * gate
        
        return gated_grad
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply gradient gate
                grad = self._apply_gradient_gate(grad, group['gate_threshold'])
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Initialize momentum buffers if needed
                state = self.state[p]
                if 'momentum_short' not in state:
                    state['momentum_short'] = torch.zeros_like(p.data)
                if 'momentum_long' not in state:
                    state['momentum_long'] = torch.zeros_like(p.data)
                
                # Update short-term momentum
                state['momentum_short'].mul_(group['momentum_short']).add_(grad)
                
                # Update long-term momentum
                state['momentum_long'].mul_(group['momentum_long']).add_(grad)
                
                # Merge dual momentum
                # Weighted combination favoring short-term for responsiveness
                alpha = 0.7  # Weight for short-term momentum
                merged_momentum = (alpha * state['momentum_short'] + 
                                 (1 - alpha) * state['momentum_long'])
                
                # Adaptive quantization based on activation variance
                if group['quantize']:
                    activation_var = self._get_activation_variance(p)
                    # Lower variance -> lower bit width
                    if activation_var < 0.01:
                        bit_width = group['bit_width_low']
                    else:
                        bit_width = group['bit_width_high']
                    
                    # Quantize weights (for storage efficiency)
                    # Note: In practice, this might be applied periodically
                    # rather than every step to avoid overhead
                    if p.numel() > 1000:  # Only quantize larger tensors
                        p.data = self._quantize_weights(p.data, bit_width)
                
                # Update parameters
                p.data.add_(merged_momentum, alpha=-group['lr'])
        
        return loss

