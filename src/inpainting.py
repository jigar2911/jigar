import cv2
import numpy as np
import torch
from PIL import Image
import os

class Inpainter:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = self._load_model()

    def _load_model(self):
        # In a full implementation, we would load LaMa or similar here.
        # For this prototype, we'll use OpenCV's inpainting as a baseline
        # but structured so an AI model can be swapped in.
        return None

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        image: RGB image as numpy array (H, W, 3)
        mask: Grayscale mask (H, W), where 255 is the area to inpaint
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Dilate mask slightly to ensure edges are covered
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Telea's algorithm is often better than Navier-Stokes for smaller watermarks
        res = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return res

def get_inpainter():
    return Inpainter()
