import torch
from typing import Dict, Any
from torch.utils.data import DataLoader

from src.utils.metrics import compute_mask_iou, compute_proxy_map

class Evaluator:
    """
    Read-only validation harness for the Multi-Task Network.
    """
    def __init__(self, model: torch.nn.Module, dataloader: DataLoader, device: str):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self) -> Dict[str, float]:
        print("\n--- Starting Validation Phase ---")
        self.model.eval() # Lock model (disables dropout, locks batchnorm)
        
        total_iou = 0.0
        total_map = 0.0
        batches = 0

        # Disable gradient calculations to save VRAM and speed up inference
        with torch.no_grad():
            for images, targets in self.dataloader:
                images = images.to(self.device)
                targets['masks'] = targets['masks'].to(self.device)
                
                # Forward Pass
                outputs = self.model(images)
                
                # Compute Metrics
                # Segmentation metric uses the first class channel (e.g., drivable area)
                iou = compute_mask_iou(outputs['segmentation'][:, 0:1], targets['masks'])
                total_iou += iou
                
                # Proxy Detection Metric
                map_score = compute_proxy_map(outputs['detection']['bbox_preds'], targets['boxes'])
                total_map += map_score
                
                batches += 1

        avg_iou = total_iou / batches if batches > 0 else 0
        avg_map = total_map / batches if batches > 0 else 0

        print(f"Validation Complete | Mask IoU: {avg_iou:.4f} | Proxy mAP: {avg_map:.4f}")
        return {"mask_iou": avg_iou, "map_50": avg_map}