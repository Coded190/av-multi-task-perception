import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

# Modern PyTorch AMP (Automatic Mixed Precision)
from torch.amp import autocast, GradScaler
from typing import Any

class Trainer:
    """
    Command Execution Engine.
    Orchestrates forward pass, multi-task loss derivation, backpropagation, and state updates.
    Utilizes AMP for optimal 16 GB VRAM utilization.
    """
    def __init__(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        optimizer: Optimizer, 
        criterion: nn.Module, 
        device: str
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # Initialize the gradient scaler for 16-bit precision training
        self.scaler = GradScaler('cuda' if 'cuda' in device else 'cpu')

    def train_epoch(self, epoch: int) -> float:
        self.model.train() # Unlock model for updates
        total_loss = 0.0
        
        print(f"\n--- Epoch {epoch} Training Started ---")
        
        for batch_idx, (images, targets) in enumerate(self.dataloader):
            # Move data to target device (GPU)
            images = images.to(self.device)
            targets['masks'] = targets['masks'].to(self.device)
            # Boxes and labels remain on CPU/lists in this proxy setup until assigned to anchors
            
            # 1. Zero the gradients from the previous step
            self.optimizer.zero_grad(set_to_none=True)
            
            # 2. Forward Pass with Mixed Precision context
            with autocast(device_type=self.device):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['loss_total']
            
            # 3. Scaled Backpropagation
            self.scaler.scale(loss).backward()
            
            # 4. Scaled Optimizer Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(self.dataloader)}] -> Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f}")
        
        return avg_loss

# --- Verification Pipeline (KISS) ---
if __name__ == "__main__":
    print("Initializing Core Engine Verification Run...\n")
    
    import yaml
    from src.models.multi_task_net import MultiTaskNet
    from src.losses.total_loss import MultiTaskLoss
    from src.datasets.av_dataset import AVMultiTaskDataset, collate_fn
    from src.datasets.transforms import get_train_transforms

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Engine target device: {device}")
    
    with open("config/base_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    model = MultiTaskNet().to(device)
    criterion = MultiTaskLoss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Dataset
    transforms = get_train_transforms()
    dataset = AVMultiTaskDataset(config_path="config/base_config.yaml", transform=transforms)
    
    # We use a tiny batch size and limit the dataloader to verify the engine without a full training run
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Initialize Engine
    trainer = Trainer(model, dataloader, optimizer, criterion, device)
    
    # Run a mock single epoch (breaking early just to test the loop mechanics)
    print("\nExecuting dry-run of Trainer Engine...")
    trainer.model.train()
    
    # Manually step exactly 1 batch for verification
    images, targets = next(iter(dataloader))
    images = images.to(device)
    targets['masks'] = targets['masks'].to(device)
    
    optimizer.zero_grad()
    with autocast(device_type=device):
        outputs = trainer.model(images)
        loss_dict = trainer.criterion(outputs, targets)
        loss = loss_dict['loss_total']
        
    trainer.scaler.scale(loss).backward()
    trainer.scaler.step(optimizer)
    trainer.scaler.update()
    
    print("\nTrainer Engine step successful! Scaler updated, weights modified.")