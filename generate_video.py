import sys
import yaml
import torch
import numpy as np
from skimage import img_as_ubyte, transform
from imageio import mimwrite, get_reader
from skimage import io
from demo import load_checkpoints, make_animation

def resize_frame_with_aspect_ratio(frame, target_height, target_width):
    """Resize a frame while maintaining aspect ratio, with padding or cropping."""
    original_height, original_width = frame.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    resized_frame = transform.resize(frame, (new_height, new_width), anti_aliasing=True, preserve_range=True).astype(np.float32)

    # Add padding or crop to fit the exact target size
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.float32)
    start_y = (target_height - new_height) // 2
    start_x = (target_width - new_width) // 2
    padded_frame[start_y:start_y + new_height, start_x:start_x + new_width] = resized_frame
    return padded_frame

def preprocess_frame(frame):
    """Ensure pixel values are scaled to [0, 1] for model input."""
    if frame.max() > 1.0:
        frame = frame / 255.0  # Scale pixel values to [0, 1]
    return frame

def postprocess_frame(frame):
    """Convert frame back to uint8 for video output."""
    frame = np.clip(frame, 0, 1)  # Ensure pixel values are within [0, 1]
    return img_as_ubyte(frame)

def synchronize_frames(source_frames, target_frame_count):
    """Synchronize frames to match the target frame count using interpolation."""
    frame_indices = np.linspace(0, len(source_frames) - 1, target_frame_count).astype(int)
    synchronized_frames = [source_frames[i] for i in frame_indices]
    return synchronized_frames

def generate_video(source_image_path, driving_video_path, output_path):
    # Load configuration and checkpoints
    config_path = 'model/config/vox-adv-256.yaml'
    checkpoint_path = 'model/checkpoints/vox-adv-cpk.pth.tar'
    
    # Load the pre-trained model
    generator, kp_detector = load_checkpoints(config_path=config_path, 
                                              checkpoint_path=checkpoint_path)
    
    # Read and preprocess source image
    source_image = io.imread(source_image_path)
    source_image = resize_frame_with_aspect_ratio(source_image, 256, 256)
    source_image = preprocess_frame(source_image)
    
    # Read driving video
    try:
        driving_video = [frame for frame in get_reader(driving_video_path, format="FFMPEG")]
    except Exception as e:
        print(f"Error reading driving video: {e}")
        return

    # Check if the driving video is empty
    if len(driving_video) == 0:
        print("Error: Driving video contains no frames.")
        return
    
    # Resize and preprocess all driving video frames
    resized_driving_video = [preprocess_frame(resize_frame_with_aspect_ratio(frame, 256, 256)) for frame in driving_video]
    
    # Synchronize the frame count
    resized_driving_video = synchronize_frames(resized_driving_video, len(resized_driving_video))
    
    # Generate animation
    try:
        predictions = make_animation(source_image, resized_driving_video, generator, kp_detector, relative=True)
    except RuntimeError as e:
        print(f"Runtime error during animation generation: {e}")
        return
    
    # Postprocess frames and save output video
    output_frames = [postprocess_frame(frame) for frame in predictions]
    mimwrite(output_path, output_frames, fps=30)
    print(f"Video saved at {output_path}")

if __name__ == "__main__":
    # Handle command-line arguments or set default values
    if len(sys.argv) > 2:
        source_image_path = sys.argv[1]  # First argument: Source image path
        driving_video_path = sys.argv[2]  # Second argument: Driving video path
    else:
        # Default paths for testing
        print("No arguments provided. Using default paths for testing.")
        source_image_path = 'static/source.jpg'
        driving_video_path = 'static/driving.mp4'

    output_path = 'static/generated_video.mp4'  # Output video path
    
    try:
        generate_video(source_image_path, driving_video_path, output_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
