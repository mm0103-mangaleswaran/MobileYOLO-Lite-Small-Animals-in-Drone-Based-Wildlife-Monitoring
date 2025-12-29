"""
Comprehensive test suite for MobileYOLO-Lite
Tests all modules and components
"""

import torch
import sys
import traceback

def test_imports():
    """Test all imports"""
    print("=" * 60)
    print("TEST 1: Testing Imports")
    print("=" * 60)
    try:
        from models import MobileYOLOLite, PatchPack, GhostMobileBackbone, LiteBiDFPN, SparseTinyHead, SGDM_AQS
        from utils import WAIDDataset, YOLOLoss, decode_yolo_output, nms
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_patchpack():
    """Test PatchPack module"""
    print("\n" + "=" * 60)
    print("TEST 2: Testing PatchPack Module")
    print("=" * 60)
    try:
        from models import PatchPack
        
        patchpack = PatchPack(patch_size=2)
        x = torch.randn(2, 3, 608, 608)
        out = patchpack(x)
        
        assert out.shape == (2, 12, 304, 304), f"Expected (2, 12, 304, 304), got {out.shape}"
        print(f"✓ PatchPack: Input {x.shape} -> Output {out.shape}")
        return True
    except Exception as e:
        print(f"✗ PatchPack test failed: {e}")
        traceback.print_exc()
        return False

def test_backbone():
    """Test Ghost-Mobile Backbone"""
    print("\n" + "=" * 60)
    print("TEST 3: Testing Ghost-Mobile Backbone")
    print("=" * 60)
    try:
        from models import GhostMobileBackbone
        
        backbone = GhostMobileBackbone(in_channels=12)
        x = torch.randn(2, 12, 304, 304)
        features = backbone(x)
        
        assert 'p3' in features, "Missing p3 feature"
        assert 'p4' in features, "Missing p4 feature"
        assert 'p5' in features, "Missing p5 feature"
        assert features['p3'].shape == (2, 16, 152, 152), f"p3 shape incorrect: {features['p3'].shape}"
        assert features['p4'].shape == (2, 24, 76, 76), f"p4 shape incorrect: {features['p4'].shape}"
        assert features['p5'].shape == (2, 32, 38, 38), f"p5 shape incorrect: {features['p5'].shape}"
        
        print(f"✓ Backbone outputs:")
        print(f"  p3: {features['p3'].shape}")
        print(f"  p4: {features['p4'].shape}")
        print(f"  p5: {features['p5'].shape}")
        return True
    except Exception as e:
        print(f"✗ Backbone test failed: {e}")
        traceback.print_exc()
        return False

def test_bidfpn():
    """Test Lite-BiDFPN"""
    print("\n" + "=" * 60)
    print("TEST 4: Testing Lite-BiDFPN")
    print("=" * 60)
    try:
        from models import LiteBiDFPN
        
        neck = LiteBiDFPN(in_channels_list=[16, 24, 32], out_channels=64)
        features = {
            'p3': torch.randn(2, 16, 152, 152),
            'p4': torch.randn(2, 24, 76, 76),
            'p5': torch.randn(2, 32, 38, 38)
        }
        fused = neck(features)
        
        assert fused['p3'].shape == (2, 64, 152, 152), f"p3 shape incorrect: {fused['p3'].shape}"
        assert fused['p4'].shape == (2, 64, 76, 76), f"p4 shape incorrect: {fused['p4'].shape}"
        assert fused['p5'].shape == (2, 64, 38, 38), f"p5 shape incorrect: {fused['p5'].shape}"
        
        print(f"✓ Lite-BiDFPN outputs:")
        print(f"  p3: {fused['p3'].shape}")
        print(f"  p4: {fused['p4'].shape}")
        print(f"  p5: {fused['p5'].shape}")
        return True
    except Exception as e:
        print(f"✗ Lite-BiDFPN test failed: {e}")
        traceback.print_exc()
        return False

def test_head():
    """Test Sparse Tiny-Head"""
    print("\n" + "=" * 60)
    print("TEST 5: Testing Sparse Tiny-Head")
    print("=" * 60)
    try:
        from models import SparseTinyHead
        
        head = SparseTinyHead(in_channels=64, num_classes=6, num_anchors=3)
        features = {
            'p3': torch.randn(2, 64, 152, 152),
            'p4': torch.randn(2, 64, 76, 76)
        }
        predictions = head(features)
        
        assert 'pred_152' in predictions, "Missing pred_152"
        assert 'pred_76' in predictions, "Missing pred_76"
        assert 'mask_152' in predictions, "Missing mask_152"
        assert 'mask_76' in predictions, "Missing mask_76"
        
        # Check output shapes: num_anchors * (5 + num_classes) = 3 * (5 + 6) = 33
        assert predictions['pred_152'].shape == (2, 33, 152, 152), f"pred_152 shape incorrect"
        assert predictions['pred_76'].shape == (2, 33, 76, 76), f"pred_76 shape incorrect"
        
        print(f"✓ Sparse Tiny-Head outputs:")
        print(f"  pred_152: {predictions['pred_152'].shape}")
        print(f"  pred_76: {predictions['pred_76'].shape}")
        print(f"  mask_152: {predictions['mask_152'].shape}")
        print(f"  mask_76: {predictions['mask_76'].shape}")
        return True
    except Exception as e:
        print(f"✗ Sparse Tiny-Head test failed: {e}")
        traceback.print_exc()
        return False

def test_full_model():
    """Test complete MobileYOLO-Lite model"""
    print("\n" + "=" * 60)
    print("TEST 6: Testing Complete MobileYOLO-Lite Model")
    print("=" * 60)
    try:
        from models import MobileYOLOLite
        
        model = MobileYOLOLite(num_classes=6, input_size=608, num_anchors=3)
        x = torch.randn(2, 3, 608, 608)
        
        with torch.no_grad():
            predictions = model(x)
        
        assert 'pred_152' in predictions, "Missing pred_152"
        assert 'pred_76' in predictions, "Missing pred_76"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Full model test passed")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Output shapes: pred_152={predictions['pred_152'].shape}, pred_76={predictions['pred_76'].shape}")
        return True
    except Exception as e:
        print(f"✗ Full model test failed: {e}")
        traceback.print_exc()
        return False

def test_optimizer():
    """Test SGDM-AQS optimizer"""
    print("\n" + "=" * 60)
    print("TEST 7: Testing SGDM-AQS Optimizer")
    print("=" * 60)
    try:
        from models import SGDM_AQS, MobileYOLOLite
        
        model = MobileYOLOLite(num_classes=6)
        optimizer = SGDM_AQS(model.parameters(), lr=0.001)
        
        # Dummy forward and backward
        x = torch.randn(1, 3, 608, 608)
        pred = model(x)
        loss = pred['pred_152'].mean()  # Dummy loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✓ SGDM-AQS optimizer test passed")
        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_loss():
    """Test YOLO loss function"""
    print("\n" + "=" * 60)
    print("TEST 8: Testing YOLO Loss Function")
    print("=" * 60)
    try:
        from utils import YOLOLoss
        
        criterion = YOLOLoss(num_classes=6, num_anchors=3)
        
        # Create dummy predictions and targets
        pred = torch.randn(2, 33, 152, 152)  # 3 anchors * (5 + 6 classes) = 33
        targets = {
            'boxes': [
                torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32),
                torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
            ],
            'classes': [
                torch.tensor([0, 1], dtype=torch.long),
                torch.tensor([2], dtype=torch.long)
            ]
        }
        
        loss = criterion(pred, targets, grid_size=152)
        
        assert loss.item() >= 0, "Loss should be non-negative"
        print(f"✓ YOLO Loss test passed (loss = {loss.item():.4f})")
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        traceback.print_exc()
        return False

def test_postprocess():
    """Test post-processing functions"""
    print("\n" + "=" * 60)
    print("TEST 9: Testing Post-Processing Functions")
    print("=" * 60)
    try:
        from utils.postprocess import decode_yolo_output, nms, calculate_iou
        
        # Create dummy predictions
        pred = torch.randn(1, 33, 152, 152)
        boxes, scores, classes = decode_yolo_output(pred, grid_size=152, conf_threshold=0.1)
        
        print(f"✓ Decode output: {len(boxes)} boxes detected")
        
        # Test IoU calculation
        if len(boxes) >= 2:
            iou = calculate_iou(boxes[:1], boxes[1:2])
            print(f"✓ IoU calculation: {iou.item():.4f}")
        
        # Test NMS
        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold=0.5)
            print(f"✓ NMS: {len(keep)} boxes kept from {len(boxes)}")
        
        print("✓ Post-processing test passed")
        return True
    except Exception as e:
        print(f"✗ Post-processing test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MOBILEYOLO-LITE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_patchpack,
        test_backbone,
        test_bidfpn,
        test_head,
        test_full_model,
        test_optimizer,
        test_loss,
        test_postprocess
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED! Project is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

