import os
import yaml
import torch
from torch.utils.data import DataLoader

# Absolute imports from your core source package
from src.models.multi_task_net import MultiTaskNet
from src.datasets.av_dataset import AVMultiTaskDataset, collate_fn
from src.datasets.transforms import get_train_transforms
from src.engine.evaluator import Evaluator

def main():
    print("========================================================")
    print("      AV Multi-Task Perception: Evaluation Engine       ")
    print("========================================================\n")

    # 1. Parse Configuration
    config_path = "config/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Smart Device Management
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[*] Evaluation environment hardware targeting: {device.upper()}")

    # 3. Load Target Weights
    weight_path = "weights/best_model.pth"
    if not os.path.exists(weight_path):
        print(f"[!] Error: Weights not found at '{weight_path}'. Did you run train.py first?")
        return

    # 4. Instantiate Network Architecture
    # We initialize the bare structure first before injecting the learned parameters
    model = MultiTaskNet(
        in_channels=3, 
        num_det_classes=3, 
        num_seg_classes=2
    ).to(device)

    print(f"[*] Loading serialized weights from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    
    # Handle potential prefix mismatches if the model was saved while inside torch.compile()
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict, strict=True)
    print("    -> Weights successfully loaded into network architecture.")

    # 5. Prepare Test Dataset & Dataloader
    # In a production environment, you would point this to a dedicated "data/test" folder.
    # For this verification run, we load the standard configuration data directory.
    transforms = get_train_transforms(imgsz=config['data']['imgsz'])
    test_dataset = AVMultiTaskDataset(config_path=config_path, transform=transforms)
    
    batch_size = config['training']['batch_size']
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    print(f"[*] Test Dataloader initialized with {len(test_dataset)} samples.")

    # 6. Mount Evaluation Engine
    evaluator = Evaluator(model, test_loader, device)

    # 7. Execute Validation Run
    print("\n[*] Commencing Global Inference Run...")
    metrics = evaluator.evaluate()

    # 8. Generate Final Performance Report
    print("\n========================================================")
    print("                 FINAL INFERENCE REPORT                 ")
    print("========================================================")
    print(f"Checkpoint Epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"Segmentation IoU : {metrics['mask_iou'] * 100:.2f}%")
    print(f"Detection mAP@50 : {metrics['map_50'] * 100:.2f}%")
    print("========================================================")
    
    if metrics['mask_iou'] < 0.10:
        print("\n[!] Diagnostic Warning: Mask IoU is extremely low.")
        print("    This is expected if training with mock annotations or only a few epochs.")
        print("    Once you connect real annotations, this will rise significantly.")

if __name__ == "__main__":
    main()