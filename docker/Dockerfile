# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python and FFmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Expose Gradio port
EXPOSE 7860

# Run the app
CMD ["python3", "src/app.py"]
