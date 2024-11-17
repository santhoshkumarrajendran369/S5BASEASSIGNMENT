import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optimized transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomRotation(5),  # Reduced rotation
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.98, 1.02)),  # Minimal distortion
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # Use more data for initial training
    train_size = int(0.98 * len(train_dataset))
    finetune_size = len(train_dataset) - train_size
    train_dataset, finetune_dataset = torch.utils.data.random_split(train_dataset, [train_size, finetune_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=512,  # Larger batch size
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    finetune_loader = torch.utils.data.DataLoader(
        finetune_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use Lion optimizer for better convergence
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.2,  # Higher learning rate
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-5
    )
    
    # Aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.4,  # Very high max learning rate
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Quick warmup
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    def train_phase(loader, phase_name, use_scheduler=True):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision for faster training
            with torch.cuda.amp.autocast(enabled=device.type != 'cpu'):
                output = model(data)
                loss = criterion(output, target)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print statistics
            running_loss += loss.item()
            if batch_idx % 10 == 9:  # More frequent updates
                print(f'{phase_name} Phase - Batch {batch_idx+1}/{len(loader)}, '
                      f'Loss: {running_loss/10:.4f}, '
                      f'Accuracy: {100.*correct/total:.2f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                running_loss = 0.0
        
        return 100. * correct / total
    
    # Initial training phase
    print("Starting initial training phase...")
    initial_accuracy = train_phase(train_loader, "Initial", use_scheduler=True)
    
    # Fine-tuning phase
    print("\nStarting fine-tuning phase...")
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
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