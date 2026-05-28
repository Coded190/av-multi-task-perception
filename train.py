import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split

# Absolute imports from your core source package
from src.models.multi_task_net import MultiTaskNet
from src.datasets.av_dataset import AVMultiTaskDataset, collate_fn
from src.datasets.transforms import get_train_transforms
from src.losses.total_loss import MultiTaskLoss
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

def main():
    print("========================================================")
    print("  Initializing AV Multi-Task Perception Training Loop   ")
    print("========================================================\n")

    # 1. Parse Configuration (Single Source of Truth)
    config_path = "config/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Smart Device Management (Ensuring cross-platform robustness)
    if torch.cuda.is_available():
        device = "cuda"
        # Enable TensorFloat32 (TF32) on Ampere/Ada Lovelace architectures for a massive speedup
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = "mps"  # Leveraging Apple Silicon GPU acceleration on your Mac
    else:
        device = "cpu"
    print(f"[*] Targeting execution engine environment device: {device.upper()}")

    # 3. Instantiate Dataset & Apply Strategic Partitions (80/20 Split)
    transforms = get_train_transforms(imgsz=config['data']['imgsz'])
    full_dataset = AVMultiTaskDataset(config_path=config_path, transform=transforms)
    
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    # Deterministic split via seed generator
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"[*] Dataset Partitions Created | Total: {total_samples} | Train: {train_size} | Val: {val_size}")

    # 4. Construct Data Loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )

    # 5. Initialize Network Components
    print("[*] Instantiating Network Model Architectures...")
    model = MultiTaskNet(
        in_channels=3, 
        num_det_classes=3, 
        num_seg_classes=2
    ).to(device)

    # 6. Apply Torch Inductor Compiler Optimization (PyTorch 2.0+ Feature)
    # Highly performant on Linux/Windows CUDA systems; safely bypassed on incompatible systems.
    if device == "cuda":
        try:
            print("[*] Launching torch.compile() Kernel Optimization...")
            model = torch.compile(model) # type: ignore
            print("    -> Compilation complete! Performance graph optimization active.")
        except Exception as e:
            print(f"    -> Compilation skipped or unsupported: {e}. Running in standard mode.")
    else:
        print("[*] Skipping torch.compile() (Optimizations best suited for native CUDA devices).")

    # 7. Setup Mathematical Loss and Optimizer
    criterion = MultiTaskLoss(config=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 8. Mount Execution Engines
    trainer = Trainer(model, train_loader, optimizer, criterion, device)
    evaluator = Evaluator(model, val_loader, device)

    # 9. Core Training / Evaluation Orchestration Loop
    num_epochs = 5  # Scalable target epoch count
    best_iou = -1.0
    
    os.makedirs("weights", exist_ok=True)

    print(f"\n[*] Commencing Training Run over {num_epochs} Epochs...")
    for epoch in range(1, num_epochs + 1):
        # Trigger training operations
        avg_train_loss = trainer.train_epoch(epoch)
        
        # Trigger validation operations
        val_metrics = evaluator.evaluate()
        current_iou = val_metrics['mask_iou']

        # 10. Checkpoint Serialization (Save the best weights)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'mask_iou': current_iou
        }
        
        # Save latest tracking state
        torch.save(checkpoint, "weights/latest_model.pth")
        
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(checkpoint, "weights/best_model.pth")
            print(f"[Saved New Best Model] -> weights/best_model.pth at Mask IoU: {best_iou:.4f}")

    print("\n========================================================")
    print("Training Routine Execution Successfully Finalized! ")
    print("========================================================")

if __name__ == "__main__":
    main()