import os
import cv2

def split_and_chunk_stereo_video(video_path: str, output_dir: str, segment_minutes: int = 5, keep_perspective: str = "left"):
    """
    Splits a stereo side-by-side video in half and chunks it into smaller time segments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Gather video metadata
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    full_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate crop dimensions (Vertical cut down the center for Side-by-Side stereo)
    cropped_width = full_width // 2
    
    # Calculate how many frames belong in a 5-minute chunk
    frames_per_segment = fps * 60 * segment_minutes
    
    print(f"Original Video Specs: {full_width}x{height} @ {fps} FPS. Total Frames: {total_frames}")
    print(f"Target Cropped Specs: {cropped_width}x{height} (Keeping {keep_perspective} view)")
    print(f"Chunking into {segment_minutes}-minute segments (~{frames_per_segment} frames each)...")

    segment_index = 1
    frame_count = 0
    out = None

    # Setup standard MP4 codec compatible with most annotation software
    fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break  # End of video reached

        # 1. Slice out the desired camera perspective
        if keep_perspective == "left":
            # Crop from x=0 to x=middle
            cropped_frame = frame[:, 0:cropped_width]
        else:
            # Crop from x=middle to x=end
            cropped_frame = frame[:, cropped_width:full_width]

        # 2. Manage the 5-minute video chunks
        if frame_count % frames_per_segment == 0:
            # If an older segment is open, close it cleanly
            if out is not None:
                out.release()
                print(f"Finished writing Clip {segment_index - 1}")

            # Initialize a new 5-minute video writer file
            output_filename = os.path.join(output_dir, f"drive_clip_{segment_index:02d}.mp4")
            out = cv2.VideoWriter(output_filename, fourcc, fps, (cropped_width, height))
            print(f"Starting Clip {segment_index}: {output_filename}")
            segment_index += 1

        # Write the cropped frame to the active segment file
        if out is not None:
            out.write(cropped_frame)
        
        frame_count += 1

    # Final cleanup execution
    if out is not None:
        out.release()
    cap.release()
    print(f"\nProcessing complete! Generated {segment_index - 1} clips in '{output_dir}'.")

if __name__ == "__main__":
    # Update these paths to match your local filename
    RAW_VIDEO = "data/raw/WIN_20251012_11_05_50_Pro.mp4" 
    OUTPUT_FOLDER = "data/raw/"  # Keeps the resulting 5-min clips in raw data folder
    
    split_and_chunk_stereo_video(
        video_path=RAW_VIDEO, 
        output_dir=OUTPUT_FOLDER, 
        segment_minutes=5, 
        keep_perspective="right" # Switch to "right" if you prefer the other camera
    )