import torch

def compute_mask_iou(pred_logits: torch.Tensor, target_masks: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes the Intersection-over-Union (IoU) for binary segmentation masks.
    """
    # Convert logits to probabilities, then to binary predictions based on threshold
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    targets = target_masks.float()

    # Calculate intersection and union
    intersection = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection

    # Avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.mean().item()

def compute_proxy_map(pred_boxes: list, target_boxes: list) -> float:
    """
    Proxy function for Mean Average Precision (mAP@0.5).
    In a full production environment, this interfaces with pycocotools.
    For this build, we return a mock proxy score to validate the engine loop.
    """
    # Real mAP requires sorting by confidence and bipartite matching across IoU thresholds.
    # Returning a proxy zero so the engine loop can complete without crashing.
    return 0.0