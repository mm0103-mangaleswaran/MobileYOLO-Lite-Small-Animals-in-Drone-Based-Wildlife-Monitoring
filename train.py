"""
Training script for MobileYOLO-Lite
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import json

from models import MobileYOLOLite
from models.sgdm_aqs import SGDM_AQS
from utils.dataset import WAIDDataset, collate_fn
from utils.loss import YOLOLoss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, boxes_list, classes_list) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Prepare targets
        targets = {
            'boxes': boxes_list,
            'classes': classes_list
        }
        
        # Calculate loss for both scales
        loss_152 = criterion(predictions['pred_152'], targets, grid_size=152)
        loss_76 = criterion(predictions['pred_76'], targets, grid_size=76)
        loss = loss_152 + loss_76
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, boxes_list, classes_list in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            
            predictions = model(images)
            
            targets = {
                'boxes': boxes_list,
                'classes': classes_list
            }
            
            loss_152 = criterion(predictions['pred_152'], targets, grid_size=152)
            loss_76 = criterion(predictions['pred_76'], targets, grid_size=76)
            loss = loss_152 + loss_76
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Train MobileYOLO-Lite')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train/val splits')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=6,
                       help='Number of classes')
    parser.add_argument('--input_size', type=int, default=608,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Dataset
    train_dataset = WAIDDataset(
        image_dir=os.path.join(args.data_dir, 'train', 'images'),
        label_dir=os.path.join(args.data_dir, 'train', 'labels'),
        input_size=args.input_size,
        augment=True
    )
    
    val_dataset = WAIDDataset(
        image_dir=os.path.join(args.data_dir, 'val', 'images'),
        label_dir=os.path.join(args.data_dir, 'val', 'labels'),
        input_size=args.input_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    
    # Model
    model = MobileYOLOLite(
        num_classes=args.num_classes,
        input_size=args.input_size
    ).to(device)
    
    # Loss function
    criterion = YOLOLoss(num_classes=args.num_classes)
    
    # Optimizer (SGDM-AQS)
    optimizer = SGDM_AQS(
        model.parameters(),
        lr=args.lr,
        momentum_short=0.9,
        momentum_long=0.99,
        weight_decay=0.0001
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save training history
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print('Training completed!')


if __name__ == '__main__':
    main()

