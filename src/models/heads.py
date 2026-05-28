import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple 

class DetectionHead(nn.Module):
    """ 
    Decoupled Object Detection Head (Anchor-Free Style).
    Takes FPN pyramid features and processes classification and box regression
    in separate parallel convolutional branches.
    """
    def __init__(self, in_channels: int = 256, num_classes: int=3):
        # num_classes = 3 (e.g., Vehicles, Pedestrians, Traffic Signs)
        super().__init__()
        self.num_classes = num_classes
        
        # We decouple the tasks. Classification needs to look at visual textures,
        # while Box regression needs to look at geometry and edges.
        
        # Classification Branch
        self.cls_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Final 1x1 Conv mapping to class scores
        self.cls_out = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Bounding Box Branch
        self.reg_stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Final 1x1 Conv mapping to (x, y, w, h) coordinates
        self.reg_out = nn.Conv2d(in_channels, 4, kernel_size=1)
    
    def forward(self, fpn_features: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """ 
        Processes standard detection scales: p2, p3, and p4.
        Returns a dictionary containing lists of tensors for each scale.
        """
        # Typically, detection ignores the highest resolution p1 map to save compute,
        # relying on p2, p3, and p4 for small, medium, and large objects.
        target_scales = ['p2', 'p3', 'p4']
        
        cls_outputs = []
        reg_outputs = []
        
        for scale in target_scales:
            feature = fpn_features[scale]
            
            # Forward pass through classification branch
            cls_feat = self.cls_stem(feature)
            cls_outputs.append(self.cls_out(cls_feat))
            
            # Forward pass through regression branch
            reg_feat = self.reg_stem(feature)
            reg_outputs.append(self.reg_out(reg_feat))
            
        return {
            "cls_scores": cls_outputs, # List of (B, num_classes, H, W) for each scale
            "bbox_preds": reg_outputs # List of (B, 4, H, W) for each scale
        }
        

class SegmentationHead(nn.Module):
    """ 
    Semantic Segmentation Head.
    Takes the highest resolution feature map from the FPN (p1) and progressively
    upsamples it to match the original image resolution (e.g., 640x640).
    """
    def __init__(self, in_channels: int = 256, num_classes: int = 2):
        # num_classes = 2 (e.g., Background, Drivable Area)
        super().__init__()
        
        # We use a progressive upsampling decoder to go from 160x160 -> 640x640
        # Interpolation combined with 3x3 Convs prevents "checkerboard" artifact
        # common with raw Transposed Convolutions, saving VRAM.
        
        self.decoder = nn.Sequential(
            # First Upsample: 160x160 -> 320x320
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            
            # Second Upsample: 320x320 -> 640x640
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            
            # Final 1x1 Conv projection to target classes
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, fpn_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ 
        Extracts p1 (highest resolution) and decodes it.
        Input: Dict containing p1 of shape (B, 256, 160, 160)
        Output: Segmentation mask of shape (B, Classes, 640, 640)
        """
        p1 = fpn_features['p1']
        seg_mask = self.decoder(p1)
        return seg_mask
        

# --- Verification Pipeline (KISS) ---
if __name__ == "__main__":
    print("Initializing Multi-Task Heads Verification Run...\n")
    
    # Simulate FPN Outputs (Batch=2, Channels=256)
    dummy_fpn = {
        "p1": torch.randn(2, 256, 160, 160),
        "p2": torch.randn(2, 256, 80, 80),
        "p3": torch.randn(2, 256, 40, 40),
        "p4": torch.randn(2, 256, 20, 20)
    }
    
    # Verify Detection Head
    det_head = DetectionHead(in_channels=256, num_classes=3)
    det_outputs = det_head(dummy_fpn)
    
    print("--- Detection Head Outputs ---")
    for i, scale in enumerate(['p2 (Small)', 'p3 (Medium)', 'p4 (Large)']):
        cls_shape = list(det_outputs['cls_scores'][i].shape)
        box_shape = list(det_outputs['bbox_preds'][i].shape)
        print(f"Scale {scale}:")
        print(f"  Class Scores: {cls_shape} | Box Preds: {box_shape}")
        
    print("\n--- Segmentation Head Outputs ---")
    # 2. Verify Segmentation Head
    seg_head = SegmentationHead(in_channels=256, num_classes=2)
    seg_output = seg_head(dummy_fpn)
    print(f"Final Semantic Mask: {list(seg_output.shape)} -> Expected: [2, 2, 640, 640]")