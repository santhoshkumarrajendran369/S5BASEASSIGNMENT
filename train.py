import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced transforms for better training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(8),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # Split ratio adjusted for more training data
    train_size = int(0.95 * len(train_dataset))
    finetune_size = len(train_dataset) - train_size
    train_dataset, finetune_dataset = torch.utils.data.random_split(train_dataset, [train_size, finetune_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=128,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    finetune_loader = torch.utils.data.DataLoader(
        finetune_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # First phase with more aggressive learning rate
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.05,
                         momentum=0.9,
                         nesterov=True,
                         weight_decay=2e-4)
    
    # More aggressive learning rate scheduling
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.15,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=20.0,
        final_div_factor=200.0,
        anneal_strategy='cos'
    )
    
    def train_phase(loader, phase_name, use_scheduler=True):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print statistics
            running_loss += loss.item()
            if batch_idx % 20 == 19:
                print(f'{phase_name} Phase - Batch {batch_idx+1}/{len(loader)}, '
                      f'Loss: {running_loss/20:.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                running_loss = 0.0
        
        return 100. * correct / total
    
    # Initial training phase
    print("Starting initial training phase...")
    initial_accuracy = train_phase(train_loader, "Initial", use_scheduler=True)
    
    # Fine-tuning phase with different optimizer
    print("\nStarting fine-tuning phase...")
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    final_accuracy = train_phase(finetune_loader, "Fine-tuning", use_scheduler=False)
    
    print(f'Final Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 
              f'models/model_{timestamp}_acc{final_accuracy:.1f}.pth')
    
if __name__ == "__main__":
    train() 