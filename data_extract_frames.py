import os
import glob
import cv2

def extract_and_resize_frames(
    raw_dir: str, 
    processed_dir: str, 
    sample_interval: int = 50, 
    target_size: tuple = (640, 640)
):
    """
    Scans the raw data folder for clips, samples frames periodically, 
    resizes them to target dimensions, and saves them as JPEGs.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Find all generated mp4 clips in the raw directory
    video_paths = sorted(glob.glob(os.path.join(raw_dir, "drive_clip_*.mp4")))
    
    if not video_paths:
        print(f"No clips found matching 'drive_clip_*.mp4' in {raw_dir}")
        return

    print(f"Found {len(video_paths)} video clips to process.")
    print(f"Sampling every {sample_interval} frames. Target resolution: {target_size}")

    total_images_saved = 0

    for video_path in video_paths:
        clip_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing {clip_name}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            continue

        frame_idx = 0
        saved_from_clip = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break  # End of video clip reached

            # Sample periodically to ensure visual diversity and reduce redundancy
            if frame_idx % sample_interval == 0:
                # Resize from 1920x1080 down to the network's working 640x640 size
                # Using INTER_LINEAR as it balances processing speed and image quality
                resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                
                # Format a clean, zero-padded filename for correct sorting order
                out_filename = f"{clip_name}_frame_{frame_idx:06d}.jpg"
                out_path = os.path.join(processed_dir, out_filename)
                
                # Save as high-quality JPEG
                cv2.imwrite(out_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                saved_from_clip += 1
                total_images_saved += 1

            frame_idx += 1

        cap.release()
        print(f"Extracted {saved_from_clip} frames from {clip_name}.")

    print(f"\nExtraction complete! Total images stored in '{processed_dir}': {total_images_saved}")

if __name__ == "__main__":
    RAW_FOLDER = "data/raw"
    PROCESSED_FOLDER = "data/processed"
    
    # sample_interval=50 extracts ~1 frame per second for a 49 FPS video.
    # Change to 25 if you want more data (~2 frames per second).
    extract_and_resize_frames(
        raw_dir=RAW_FOLDER, 
        processed_dir=PROCESSED_FOLDER, 
        sample_interval=50, 
        target_size=(640, 640)
    )