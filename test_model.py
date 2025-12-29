"""
Test script to verify MobileYOLO-Lite model implementation
"""

import torch
from models import MobileYOLOLite

def test_model():
    """Test model instantiation and forward pass"""
    print("Testing MobileYOLO-Lite model...")
    
    # Create model
    model = MobileYOLOLite(num_classes=6, input_size=608, num_anchors=3)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 608, 608)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        predictions = model(dummy_input)
    
    print("\nOutput shapes:")
    print(f"  pred_152: {predictions['pred_152'].shape}")
    print(f"  pred_76: {predictions['pred_76'].shape}")
    print(f"  mask_152: {predictions['mask_152'].shape}")
    print(f"  mask_76: {predictions['mask_76'].shape}")
    
    # Test decode predictions
    print("\nTesting decode_predictions...")
    boxes, scores, classes = model.decode_predictions(
        predictions, 
        conf_threshold=0.5, 
        nms_threshold=0.5
    )
    print(f"  Detected {len(boxes)} boxes")
    
    print("\n✓ All tests passed!")

if __name__ == '__main__':
    test_model()

