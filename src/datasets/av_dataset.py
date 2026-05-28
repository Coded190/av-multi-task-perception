import os
import cv2
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict

# Absolute imports
from src.datasets.transforms import get_train_transforms

class AVMultiTaskDataset(Dataset):
    """
    Custom PyTorch Dataset for Multi-Task AV Perception.
    Loads images and pairs them with Bounding Boxes (Detection) and Masks (Segmentation).
    """
    def __init__(self, config_path: str = "config/base_config.yaml", transform=None):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.images_dir = self.config['data']['processed_dir']
        self.transform = transform
        
        # Grab all valid JPEG files
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Load Image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # OpenCV loads in BGR, convert to RGB for standard neural network input
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Mock Annotations (Placeholder since Label Studio step was skipped)
        # Bounding box in COCO format: [x_min, y_min, width, height]
        mock_boxes = [[100.0, 150.0, 200.0, 100.0]]
        mock_labels = [1]  # 1 = Vehicle
        
        # Mask: Bottom half of the image is marked as drivable area (1), top is background (0)
        h, w, _ = image.shape
        mock_mask = np.zeros((h, w), dtype=np.uint8)
        mock_mask[h // 2 :, :] = 1 

        # 3. Apply Transforms
        if self.transform:
            augmented = self.transform(
                image=image,
                bboxes=mock_boxes,
                class_labels=mock_labels,
                mask=mock_mask
            )
            image_tensor = augmented['image']
            
            # Convert targets to PyTorch tensors
            # Handle edge case where albumentations drops boxes that fall completely outside the image
            boxes = torch.tensor(augmented['bboxes'], dtype=torch.float32) if len(augmented['bboxes']) > 0 else torch.empty((0, 4))
            labels = torch.tensor(augmented['class_labels'], dtype=torch.int64)
            mask_tensor = augmented['mask'].to(torch.int64)
        else:
            # Fallback if no transform is provided
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            boxes = torch.tensor(mock_boxes, dtype=torch.float32)
            labels = torch.tensor(mock_labels, dtype=torch.int64)
            mask_tensor = torch.from_numpy(mock_mask).to(torch.int64)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "mask": mask_tensor
        }
        
        return image_tensor, targets

# --- Custom Collate Function ---
def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Images and Masks stack perfectly, but Bounding Boxes vary in count per image.
    We must package boxes as a list of tensors rather than a single stacked tensor.
    """
    images = torch.stack([item[0] for item in batch])
    
    # Masks can be stacked because they all share the exact same spatial dimensions
    masks = torch.stack([item[1]['mask'] for item in batch])
    
    # Boxes and labels must remain lists because Image A might have 2 cars, and Image B might have 5
    targets = {
        "boxes": [item[1]['boxes'] for item in batch],
        "labels": [item[1]['labels'] for item in batch],
        "masks": masks
    }
    return images, targets


# --- Verification Pipeline (KISS) ---
if __name__ == "__main__":
    print("Initializing Data Ingestion Verification Run...\n")
    
    # Create a dummy image just in case data/processed is empty
    os.makedirs("data/processed", exist_ok=True)
    if not any(f.endswith('.jpg') for f in os.listdir("data/processed")):
        print("No images found. Creating a dummy image for testing...")
        cv2.imwrite("data/processed/dummy_test_frame.jpg", np.zeros((1080, 1920, 3), dtype=np.uint8))
    
    # Initialize Transforms and Dataset
    transforms = get_train_transforms(imgsz=640)
    dataset = AVMultiTaskDataset(config_path="config/base_config.yaml", transform=transforms)
    
    # Read batch_size from the config via the dataset
    bs = dataset.config['training']['batch_size']
    
    dataloader = DataLoader(
        dataset, 
        batch_size=bs, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    print(f"Dataset initialized with {len(dataset)} items.")
    print(f"Loading a single batch (Batch Size = {bs})...\n")
    
    # Extract one batch
    images, targets = next(iter(dataloader))
    
    print("--- Batch Tensor Verification ---")
    print(f"Image Batch Shape: {list(images.shape)} -> Expected: [{images.shape[0]}, 3, 640, 640]")
    print(f"Mask Batch Shape:  {list(targets['masks'].shape)} -> Expected: [{images.shape[0]}, 640, 640]")
    
    print("\nBounding Boxes (per image in batch):")
    for i in range(len(targets['boxes'])):
        print(f"  Image {i} -> Boxes: {list(targets['boxes'][i].shape)}, Labels: {list(targets['labels'][i].shape)}")