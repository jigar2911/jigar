import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader
# It will download models on first use
reader = None

def get_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'])
    return reader

def detect_text_masks(image):
    """
    Detect text in the image and return a binary mask.
    """
    ocr_reader = get_reader()
    results = ocr_reader.readtext(image)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (bbox, text, prob) in results:
        # bbox is a list of [x, y] coordinates
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask

def find_similar_logos(image, full_mask):
    """
    Finds regions in 'image' that look like the one currently masked in 'full_mask'.
    Returns an updated mask.
    """
    # Find bounding box of the mask
    coords = cv2.findNonZero(full_mask)
    if coords is None:
        return full_mask

    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return full_mask

    template_roi = image[y:y+h, x:x+w]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_template = cv2.cvtColor(template_roi, cv2.COLOR_RGB2GRAY)

    # Template matching
    res = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # Strict threshold for logos
    loc = np.where(res >= threshold)

    new_mask = full_mask.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(new_mask, pt, (pt[0] + w, pt[1] + h), 255, -1)

    return new_mask
