import numpy as np
from PIL import Image
import cv2


def create_CLAHE(img: Image):
    np_img = np.array(img)
    lab_image = cv2.cvtColor(np_img, cv2.COLOR_BGR2LAB)

    # Apply CLAHE separately to each channel
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])

    # Convert back to BGR color space
    result_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    result_image = Image.fromarray(result_image)

    return result_image