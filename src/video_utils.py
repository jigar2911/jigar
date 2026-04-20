import cv2
import os
import subprocess
from tqdm import tqdm
import numpy as np

def process_video(video_path, mask_generator, inpainter, output_path):
    """
    video_path: Path to input video
    mask_generator: A function that takes a frame and returns a mask
    inpainter: Inpainter instance
    output_path: Path to save processed video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use a unique temporary file for the video without audio to avoid race conditions
    import uuid
    temp_output = f"temp_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames, desc="Processing video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get mask for this frame
        mask = mask_generator(frame_rgb)

        # Inpaint
        inpainted_frame_rgb = inpainter.inpaint(frame_rgb, mask)

        # Convert back to BGR
        inpainted_frame_bgr = cv2.cvtColor(inpainted_frame_rgb, cv2.COLOR_RGB2BGR)

        out.write(inpainted_frame_bgr)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

    # Try to copy audio from original video using ffmpeg
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_output,
            '-i', video_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            output_path
        ]
        subprocess.run(cmd, check=True)
        os.remove(temp_output)
    except Exception as e:
        print(f"Error adding audio: {e}. Saving without audio.")
        os.rename(temp_output, output_path)

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None
