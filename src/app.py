import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os
import uuid
import webbrowser
from inpainting import get_inpainter
from video_utils import process_video, extract_first_frame
from detection import find_similar_logos, detect_text_masks

inpainter = get_inpainter()

def combine_layers(input_image_data):
    image = input_image_data['background']
    layers = input_image_data['layers']
    mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
    for layer in layers:
        layer_np = np.array(layer.convert("L"))
        mask_np = cv2.bitwise_or(mask_np, layer_np)
    return image, mask_np

def handle_image(input_image_data):
    if input_image_data is None: return None
    image, mask_np = combine_layers(input_image_data)
    image_np = np.array(image.convert("RGB"))

    # Inpaint
    result = inpainter.inpaint(image_np, mask_np)
    return Image.fromarray(result)

def handle_video(video_path, input_image_data):
    if not video_path or input_image_data is None:
        return None

    image, mask_np = combine_layers(input_image_data)
    unique_id = uuid.uuid4().hex
    output_path = f"output_{unique_id}.mp4"

    def static_mask_gen(frame):
        if (mask_np.shape[0] != frame.shape[0]) or (mask_np.shape[1] != frame.shape[1]):
            return cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_np

    process_video(video_path, static_mask_gen, inpainter, output_path)
    return output_path

def load_video_frame(video_path):
    if not video_path:
        return None
    frame = extract_first_frame(video_path)
    return Image.fromarray(frame)

def auto_detect_watermarks(input_image_data):
    if input_image_data is None: return None
    image = input_image_data['background']
    image_np = np.array(image.convert("RGB"))

    mask_np = detect_text_masks(image_np)
    return {"background": image, "layers": [Image.fromarray(mask_np)], "composite": None}

def find_similar_btn_handler(input_image_data):
    if input_image_data is None: return None
    image, mask_np = combine_layers(input_image_data)
    image_np = np.array(image.convert("RGB"))

    new_mask_np = find_similar_logos(image_np, mask_np)
    return {"background": image, "layers": [Image.fromarray(new_mask_np)], "composite": None}

with gr.Blocks() as demo:
    gr.Markdown("# 🎥 AI Watermark Remover (Pro)")
    gr.Markdown("Remove watermarks from photos and videos using AI-powered inpainting.")

    with gr.Tab("Photo"):
        img_input = gr.ImageEditor(label="Upload Photo & Mask Watermark", type="pil", layers=True)
        with gr.Row():
            img_auto_btn = gr.Button("Auto-Detect (Captions)")
            img_similar_btn = gr.Button("Find Similar (Logo)")
            img_process_btn = gr.Button("Remove Watermark", variant="primary")
        img_output = gr.Image(label="Result")

        img_auto_btn.click(auto_detect_watermarks, inputs=[img_input], outputs=img_input)
        img_similar_btn.click(find_similar_btn_handler, inputs=[img_input], outputs=img_input)
        img_process_btn.click(handle_image, inputs=[img_input], outputs=img_output)

    with gr.Tab("Video"):
        vid_input = gr.Video(label="Upload Video")
        vid_frame_btn = gr.Button("Step 1: Extract First Frame for Masking")
        vid_mask_input = gr.ImageEditor(label="Step 2: Mask Watermark on Frame", type="pil", layers=True)

        vid_frame_btn.click(load_video_frame, inputs=[vid_input], outputs=vid_mask_input)

        with gr.Row():
            vid_auto_btn = gr.Button("Auto-Detect (Captions)")
            vid_similar_btn = gr.Button("Find Similar (Logo)")
            vid_process_btn = gr.Button("Step 3: Process Entire Video", variant="primary")

        vid_output = gr.Video(label="Result Video")

        vid_auto_btn.click(auto_detect_watermarks, inputs=[vid_mask_input], outputs=vid_mask_input)
        vid_similar_btn.click(find_similar_btn_handler, inputs=[vid_mask_input], outputs=vid_mask_input)
        vid_process_btn.click(handle_video, inputs=[vid_input, vid_mask_input], outputs=vid_output)

if __name__ == "__main__":
    # share=True creates a public link (gradio.live)
    # which allows "all users" to access it as requested.
    print("Launching AI Watermark Remover...")
    # Attempt to open browser automatically
    try:
        webbrowser.open("http://127.0.0.1:7860")
    except:
        pass
    demo.launch(server_name="0.0.0.0", share=True)
