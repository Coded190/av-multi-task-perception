import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    """
    Computes Classification and Localization (Box) losses for matched predictions.
    Note: In a full pipeline, an assignment algorithm (like Hungarian Matching) 
    must route the specific targets to the predictions before these functions are called.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_classes: torch.Tensor, pred_boxes: torch.Tensor, target_classes: torch.Tensor, target_boxes: torch.Tensor) -> dict:
        """
        Assumes predictions and targets have already been structurally matched.
        """
        # 1. Classification Loss (Cross Entropy)
        # Penalizes the network for guessing the wrong object category (e.g. Car vs Pedestrian)
        cls_loss = F.cross_entropy(pred_classes, target_classes)
        
        # 2. Box Regression Loss (Smooth L1 / Huber Loss)
        # We use Smooth L1 because it is less sensitive to wild outliers than standard MSE.
        # For advanced production, you would swap this for Complete-IoU (CIoU) loss.
        box_loss = F.smooth_l1_loss(pred_boxes, target_boxes, beta=1.0)
        
        return {
            "loss_cls": cls_loss,
            "loss_box": box_loss
        }