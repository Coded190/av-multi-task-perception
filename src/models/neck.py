import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) module.
    Takes multi-scale features from the backbone and fuses them top-down to create semantically rich, high-resolution feature maps.
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        # SOLID Principle: The neck adapts dynamically to whatever channel list
        # the backbone provides, maintaining separation of concerns.
        
        # Step 1: 1x1 Convolutions
        # Standardizes the channel depth of all incoming maps to a unified 'out_channels' (e.g., 256)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Step 2: 3x3 Convolutions
        # Applied after merging features to smooth out the tensor and reduce aliasing artifacts
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Fuses features in a top-down manner.
        Input: Dictionary of backbone tensors (c1, c2, c3, c4)
        Output: Dictionary of refined pyramid tensors (p1, p2, p3, p4)
        """
        # Extract features from the backbone dictionary
        c1 = features['c1']
        c2 = features['c2']
        c3 = features['c3']
        c4 = features['c4']
        
        # Apply 1x1 convolutions to standardize channels
        l1 = self.lateral_convs[0](c1)
        l2 = self.lateral_convs[1](c2)
        l3 = self.lateral_convs[2](c3)
        l4 = self.lateral_convs[3](c4)
        
        # --- Top-Down FPN Pathway ---
        
        # Deepest layer acts as the foundation
        p4_basis = l4
        
        # Upsample the deeper layer and add it to the current layer
        # F.interpolate scales spatial dimentions (H,W) without affecting channels (C)
        p3_basis = l3 + F.interpolate(p4_basis, size=l3.shape[-2:], mode="nearest")
        p2_basis = l2 + F.interpolate(p3_basis, size=l2.shape[-2:], mode="nearest")
        p1_basis = l1 + F.interpolate(p2_basis, size=l1.shape[-2:], mode="nearest")
        
        # Apply 3x3 smoothing convolutions to finalize the feature pyramid
        p4 = self.fpn_convs[3](p4_basis)
        p3 = self.fpn_convs[2](p3_basis)
        p2 = self.fpn_convs[1](p2_basis)
        p1 = self.fpn_convs[0](p1_basis)
        
        # Return the refined maps to be passed to the Detection/Segmentation heads
        return {
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
        }
            
# --- Verification Pipeline (KISS) ---
if __name__ == "__main__":
    print("Initializing FPN Neck Verification Run...")
    
    # Simulate the exact dictionary of tensors output by our ResNet-18 Backbone
    # (Batch=2, Channels, Height, Width)
    dummy_features = {
        "c1": torch.randn(2, 64, 160, 160),
        "c2": torch.randn(2, 128, 80, 80),
        "c3": torch.randn(2, 256, 40, 40),
        "c4": torch.randn(2, 512, 20, 20),
    }
    
    # The backbone's out_channels list
    in_channels_list = [64, 128, 256, 512]
    
    # Instantiate the Neck, unifying all layers to 256 channels
    neck = FPNNeck(in_channels_list=in_channels_list, out_channels=256)
    
    # Run the forward pass
    with torch.no_grad():
        fpn_outputs = neck(dummy_features)
        
    print("\nFPN Extraction Success! Refined pyramid shapes verified:")
    for stage_name, feature_tensor in fpn_outputs.items():
        print(f"Pyramid Layer [{stage_name}] -> Tensor Shape: {list(feature_tensor.shape)}")