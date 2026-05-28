import cv2
import yaml
import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.multi_task_net import MultiTaskNet
from src.utils.visualization import draw_predictions

def get_inference_transforms(imgsz: int = 640):
    """Transforms strictly for inference (no augmentations, just resize and format)."""
    return A.Compose([
        A.Resize(height=imgsz, width=imgsz),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def main():
    print("========================================================")
    print("          AV Multi-Task Perception: Live Stream         ")
    print("========================================================\n")

    # 1. Setup & Configuration
    config_path = "config/base_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    imgsz = config['data']['imgsz']
    transforms = get_inference_transforms(imgsz=imgsz)

    # 2. Load Model & Weights
    model = MultiTaskNet(in_channels=3, num_det_classes=3, num_seg_classes=2).to(device)
    weight_path = "weights/best_model.pth"
    
    try:
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=True)
        print(f"[*] Weights loaded successfully from {weight_path}")
    except FileNotFoundError:
        print(f"[!] Warning: {weight_path} not found. Running with untrained random weights for testing.")

    model.eval() # Lock for inference

    # 3. Initialize Video Stream (Change '0' to your raw video path if you want to stream a file)
    video_source = "data/raw/drive_clip_01.mp4" # Alternatively use 0 for webcam
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"[!] Error: Cannot open video source {video_source}")
        return

    print("[*] Starting Video Stream. Press 'q' to exit...")

    # 4. Inference Loop
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[*] End of video stream.")
                break

            # Preprocess: Resize frame for the network while keeping original for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            augmented = transforms(image=rgb_frame)
            input_tensor = augmented['image'].unsqueeze(0).to(device) # Add batch dimension

            # Forward Pass
            outputs = model(input_tensor)

            # Process Segmentation Mask (Extract class 1: Drivable Area)
            # Apply sigmoid and threshold to convert logits to a binary mask
            seg_logits = outputs['segmentation'][0, 0] 
            seg_probs = torch.sigmoid(seg_logits)
            binary_mask = (seg_probs > 0.5).cpu().numpy().astype(np.uint8)

            # Process Bounding Boxes
            # Note: Because we bypassed complex detection decoding earlier, we will simulate 
            # the output layout that a typical decoder (YOLO/Retina) provides to the NMS algorithm.
            # Format: [x1, y1, x2, y2, confidence, class_id]
            
            # --- MOCK DECODER LOGIC FOR DEMO ---
            # In a full production script, you replace this block with your actual anchor decoding logic.
            mock_boxes = torch.tensor([[100.0, 100.0, 250.0, 200.0], [120.0, 110.0, 260.0, 210.0]]).to(device)
            mock_scores = torch.tensor([0.95, 0.85]).to(device)
            mock_labels = torch.tensor([1, 1]).to(device)
            # -----------------------------------

            # Apply Non-Maximum Suppression (NMS)
            nms_threshold = 0.45
            keep_indices = torchvision.ops.nms(mock_boxes, mock_scores, nms_threshold)
            
            final_boxes = mock_boxes[keep_indices].cpu().numpy()
            final_scores = mock_scores[keep_indices].cpu().numpy()
            final_labels = mock_labels[keep_indices].cpu().numpy()

            # Resize the mask back up to the original frame dimensions for smooth display
            h, w = frame.shape[:2]
            display_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Scale bounding boxes back up to the original frame size
            scale_x = w / imgsz
            scale_y = h / imgsz
            scaled_boxes = []
            for box in final_boxes:
                scaled_boxes.append([box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y])

            # Draw Outputs
            output_frame = draw_predictions(
                frame=frame, 
                boxes=scaled_boxes, 
                labels=final_labels, 
                scores=final_scores, 
                mask=display_mask
            )

            # Render
            cv2.imshow("AV Multi-Task Perception Dashboard", output_frame)

            # Check for quit key ('q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()