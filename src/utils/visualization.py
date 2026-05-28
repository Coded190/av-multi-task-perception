import cv2
import numpy as np

def draw_predictions(
    frame: np.ndarray, 
    boxes: list, 
    labels: list, 
    scores: list, 
    mask: np.ndarray = None, 
    class_names: dict = None
) -> np.ndarray:
    """
    Overlays predicted bounding boxes and segmentation masks onto an image frame.
    
    Args:
        frame: Original BGR image frame from OpenCV.
        boxes: List of bounding boxes [x_min, y_min, x_max, y_max].
        labels: List of class integer labels.
        scores: List of confidence scores.
        mask: 2D numpy array containing binary mask predictions (1 for class, 0 for bg).
        class_names: Dictionary mapping class integers to string names.
    """
    if class_names is None:
        class_names = {0: 'Background', 1: 'Vehicle', 2: 'Pedestrian'}

    output_frame = frame.copy()

    # 1. Blend Segmentation Mask (e.g., Green for Drivable Area)
    if mask is not None:
        # Create a blank colored canvas
        color_mask = np.zeros_like(output_frame)
        # Apply neon green to regions where the mask is active
        color_mask[mask > 0] = [0, 255, 0] 
        
        # Alpha blend the mask onto the image (50% opacity)
        output_frame = cv2.addWeighted(output_frame, 1.0, color_mask, 0.4, 0)

    # 2. Draw Bounding Boxes and Labels
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw red rectangle
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Create label text and background for visibility
        label_text = f"{class_names.get(int(label), 'Obj')}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(output_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 255), -1)
        cv2.putText(output_frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output_frame