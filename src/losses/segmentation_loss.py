import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    """
    Computes a hybrid BCE/Cross-Entropy and Dice Loss for semantic segmentation.
    Ideal for handling imbalanced classes like narrow lane markers.
    """
    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
        """
        pred_masks: Logits from the network [Batch, Classes, H, W]
        target_masks: Ground truth class indices [Batch, H, W]
        """
        # 1. Standard Cross Entropy Loss (Pixel-wise classification)
        ce_loss = F.cross_entropy(pred_masks, target_masks)
        
        # 2. Dice Loss (Intersection over Union for Masks)
        # Apply softmax to get probabilities
        probs = F.softmax(pred_masks, dim=1)
        
        # Convert target indices to one-hot encoded tensors to match probs shape
        num_classes = pred_masks.shape[1]
        targets_one_hot = F.one_hot(target_masks, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Compute intersection and cardinality, ignoring the batch and class dimensions initially
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(2, 3))
        
        # Add epsilon (1e-6) to prevent division by zero
        dice_score = (2.0 * intersection) / (cardinality + 1e-6)
        
        # Dice Loss is 1 - Dice Score (averaged across batch and classes)
        dice_loss = 1.0 - dice_score.mean()
        
        # Combine losses
        return (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)