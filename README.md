# AI Watermark Remover (Pro)

A professional-grade tool to remove watermarks from both photos and videos using AI-powered inpainting.

## Features
- **Photo Removal:** Remove logos, text, and objects from static images.
- **Video Removal:** Frame-by-frame processing with audio preservation.
- **Smart Inpainting:** Uses OpenCV's Telea algorithm for seamless background filling.
- **Auto-Detection Hooks:** Structure included for automatic caption and repeating logo detection.
- **Gradio UI:** Easy-to-use web interface with manual masking tools.
- **Dual Mode:** Supports both CPU and NVIDIA GPU acceleration.

## Installation

### Windows (One-Click)
1. Double-click `setup.bat`. This will:
   - Create a Python virtual environment.
   - Install all dependencies.
   - Detect if you have an NVIDIA GPU and install the correct PyTorch version.
   - Create a `start_app.bat` file.
2. Run `start_app.bat` to launch the program.

### Linux / Docker
1. Build the image:
   ```bash
   docker build -t watermark-remover -f docker/Dockerfile .
   ```
2. Run the container:
   ```bash
   docker run -it --gpus all -p 7860:7860 watermark-remover
   ```

## How to Use
1. **Photos:**
   - Upload an image in the "Photo" tab.
   - Use the brush tool to draw over the watermark.
   - Click "Remove Watermark".
2. **Videos:**
   - Upload a video in the "Video" tab.
   - Click "Step 1: Extract First Frame".
   - Mask the watermark on the extracted frame.
   - Click "Step 3: Process Entire Video".

## Requirements
- Python 3.10+
- FFmpeg (required for video processing)
- NVIDIA GPU (optional, but recommended for video)
