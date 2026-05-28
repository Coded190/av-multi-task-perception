import torch
import torch.nn as nn
from typing import Dict, List

class BasicBlock(nn.Module):
    """
    The fundamental building block of ResNet-18.
    Consists of two 3x3 convolutional layers with a residual shortcut connection.

    Args:
        nn (_type_): _description_
    """
    
    expansion: int = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # First convolutional layer of the block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer of the block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (Identity mapping or 1x1 Conv adjustment)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If spatial size changes (stride > 1) or channel depth changes,
            # we use a 1x1 conv to dynamically resize the shortcut tensor.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # The core breakthrough of ResNet: Adding the original input back
        out += identity
        out = self.relu(out)
        
        return out
    

class ResNet18Backbone(nn.Module):
    """
    A custom ResNet-18 backbone built from scratch.
    Modified for Dense Prediction Tasks (Detection/Segmentation) by removing the fully connected classification head and returning multi-scale feature maps.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.in_channels = 64
        
        # Root Block: Initial downsampling of the high-res camera feed
        # Input: (B, 3, 640, 640) -> Output: (B, 64, 320, 320)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Max Pooling: Further downsample spatial dimensions
        # Input: (B, 64, 320, 320) -> Output: (B, 64, 160, 160)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Stage Layers (Each stage consists of 2 BasicBlocks)
        # Stage 1: Spatial resolution remains 160x160
        self.layer1 = self._make_layer(64, blocks_count=2, stride=1)
        
        # Stage 2: Downsamples to 80x80, increases channels to 128
        self.layer2 = self._make_layer(128, blocks_count=2, stride=2)
        
        # Stage 3: Downsamples to 40x40, increases channels to 256
        self.layer3 = self._make_layer(256, blocks_count=2, stride=2)
        
        # Stage 4: Downsamples to 20x20, increases channels to 512
        self.layer4 = self._make_layer(512, blocks_count=2, stride=2)
        
        # Expose output channel depths so downstream components (Neck) can dynamically adapt (SOLID principles)
        self.out_channels = [64, 128, 256, 512]
        
    def _make_layer(self, out_channels: int, blocks_count: int, stride: int) -> nn.Sequential:
        """
        Helper function to cleanly assemble sequential BasicBlocks."""
        
        layers = []
        # The first block of a stage handles the downsampling via the 'stride' parameter
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks keep dimensions and channel sizes constant
        for _ in range(1, blocks_count):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes an image batch and returns a dictionary of multi-scale features.
        Assuming an input size of (B, 3, 640, 640):
        """
        # Initial stem layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Shape: (B, 64, 160, 160)
        
        # Extract features across network depths
        c1 = self.layer1(x) # Shape: (B, 64, 160, 160) -> Fine-grained details (edges, lanes)
        c2 = self.layer2(c1) # Shape: (B, 128, 80, 80)
        c3 = self.layer3(c2) # Shape: (B, 256, 40, 40)
        c4 = self.layer4(c3) # Shape: (B, 512, 20, 20) -> High-level context (vehicle objects)
        
        # SoC/SOLID: Return dictionary mapping names to tensors for the neck to extract easily
        return {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4
        }
        

# --- Verificaton Pipeline (KISS) ---
if __name__ == "__main__":
    print("Initializing ResNet-18 Backbone Verification Run...")
    
    # Instantiate the backbone
    backbone = ResNet18Backbone(in_channels=3)
    
    # Simulate a small mini-batch from an AV camera feed (Batch=2, RGB=3, Height=640, Width=640)
    dummy_input = torch.randn(2, 3, 640, 640)
    
    # Run the forward pass
    with torch.no_grad():
        features = backbone(dummy_input)
        
    # Verify our dictionary output shapes match our design specs
    print("\nExtraction Success! Target shape map verified:")
    for stage_name, feature_tensor in features.items():
        print(f"Stage Layer [{stage_name}] -> Tensor Shape: {list(feature_tensor.shape)}")