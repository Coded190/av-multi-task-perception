import torch
import torch.nn as nn
from typing import Dict, Any

# Absolute imports based on our project structure
from src.models.backbone import ResNet18Backbone
from src.models.neck import FPNNeck
from src.models.heads import DetectionHead, SegmentationHead

class MultiTaskNet(nn.Module):
    """
    Master Orchestrator Model for AV Perception.
    Combines the Backbone, FPN Neck, and specialized Task Heads.
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        num_det_classes: int = 3, 
        num_seg_classes: int = 2
    ):
        super().__init__()
        
        # 1. Backbone: Extracts raw multi-scale visual features
        self.backbone = ResNet18Backbone(in_channels=in_channels)
        
        # We read the output channels dynamically to maintain the Open-Closed Principle
        backbone_out_channels = self.backbone.out_channels
        
        # 2. Neck: Fuses multi-scale features into a rich Feature Pyramid
        fpn_out_channels = 256
        self.neck = FPNNeck(
            in_channels_list=backbone_out_channels, 
            out_channels=fpn_out_channels
        )
        
        # 3. Heads: Processes the pyramid into distinct AV tasks
        self.detection_head = DetectionHead(
            in_channels=fpn_out_channels, 
            num_classes=num_det_classes
        )
        self.segmentation_head = SegmentationHead(
            in_channels=fpn_out_channels, 
            num_classes=num_seg_classes
        )

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        End-to-end forward pass.
        Input: Raw image tensor (B, 3, H, W)
        Output: Dictionary containing detection lists and the segmentation mask.
        """
        # Step 1: Extract features (c1, c2, c3, c4)
        backbone_features = self.backbone(x)
        
        # Step 2: Refine into a Feature Pyramid (p1, p2, p3, p4)
        fpn_features = self.neck(backbone_features)
        
        # Step 3: Branch into parallel tasks
        det_outputs = self.detection_head(fpn_features)
        seg_output = self.segmentation_head(fpn_features)
        
        # Return a unified dictionary with all predictions
        return {
            "detection": det_outputs,    # Contains 'cls_scores' and 'bbox_preds'
            "segmentation": seg_output   # Contains the (B, Classes, H, W) mask
        }


# --- Verification Pipeline (KISS) ---
if __name__ == "__main__":
    print("Initializing Master Multi-Task Network Verification Run...\n")
    
    # Instantiate the full network
    model = MultiTaskNet(in_channels=3, num_det_classes=3, num_seg_classes=2)
    
    # Simulate a single batch of 2 images from the AV camera (640x640)
    # Moving the model and tensor to CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(2, 3, 640, 640).to(device)
    
    print(f"Executing Forward Pass on Device: {device}...")
    
    # Run the full assembly line
    with torch.no_grad():
        outputs = model(dummy_input)
        
    print("\n--- End-to-End Extraction Success! ---")
    
    # Verify Segmentation outputs
    seg_out = outputs["segmentation"]
    print(f"\nSegmentation Output Mask Shape: {list(seg_out.shape)} -> Expected: [2, 2, 640, 640]")
    
    # Verify Detection outputs
    det_out = outputs["detection"]
    print("\nDetection Output Shapes:")
    for i, scale in enumerate(['Scale p2', 'Scale p3', 'Scale p4']):
        cls_shape = list(det_out['cls_scores'][i].shape)
        box_shape = list(det_out['bbox_preds'][i].shape)
        print(f"  {scale}:")
        print(f"    Class Scores: {cls_shape}")
        print(f"    Box Preds:    {box_shape}")