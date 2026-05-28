import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(imgsz: int = 640):
    """
    Standard transforms for the multi-task training pipeline.
    Handles image resizing, normalization, and tensor conversion.
    Albumentations automatically syncs these geometric changes to the boxes and masks.
    """
    return A.Compose(
        [
            A.Resize(height=imgsz, width=imgsz),
            # Standard ImageNet normalization values
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        # Specify COCO format [x_min, y_min, width, height] for the incoming bounding boxes
        bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
    )