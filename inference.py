"""
Inference script for MobileYOLO-Lite
"""

import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from PIL import Image
import os

from models import MobileYOLOLite
from utils.postprocess import decode_yolo_output, nms


def load_model(checkpoint_path, device, num_classes=6):
    """Load trained model"""
    model = MobileYOLOLite(num_classes=num_classes, input_size=608)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size=608):
    """Preprocess image for inference"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    image = cv2.resize(image, (new_w, new_h))
    
    # Pad to input_size
    new_image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    new_image[:new_h, :new_w] = image
    
    # Normalize
    image_tensor = torch.from_numpy(new_image).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, scale, (new_w, new_h)


def draw_boxes(image, boxes, scores, classes, class_names, scale):
    """Draw bounding boxes on image"""
    h, w = image.shape[:2]
    
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.cpu().numpy()
        
        # Scale back to original image size
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # Clip to image bounds
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f'{class_names[cls]}: {score:.2f}'
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='MobileYOLO-Lite Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Path to save output image')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='NMS IoU threshold')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Class names (WAID dataset)
    class_names = ['cattle', 'sheep', 'seal', 'camelus', 'zebra', 'kiang']
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print('Loading model...')
    model = load_model(args.checkpoint, device, args.num_classes)
    
    # Preprocess image
    print('Preprocessing image...')
    image_tensor, scale, (new_w, new_h) = preprocess_image(args.image)
    image_tensor = image_tensor.to(device)
    
    # Inference
    print('Running inference...')
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Decode predictions
    print('Decoding predictions...')
    boxes_152, scores_152, classes_152 = decode_yolo_output(
        predictions['pred_152'], grid_size=152, conf_threshold=args.conf_threshold
    )
    boxes_76, scores_76, classes_76 = decode_yolo_output(
        predictions['pred_76'], grid_size=76, conf_threshold=args.conf_threshold
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
        print('No detections found!')
        return
    
    # Apply NMS
    keep = nms(all_boxes, all_scores, args.nms_threshold)
    all_boxes = all_boxes[keep]
    all_scores = all_scores[keep]
    all_classes = all_classes[keep]
    
    print(f'Found {len(all_boxes)} detections')
    
    # Load original image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw boxes
    image = draw_boxes(image, all_boxes, all_scores, all_classes, class_names, scale)
    
    # Save output
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, image_bgr)
    print(f'Output saved to {args.output}')


if __name__ == '__main__':
    main()

