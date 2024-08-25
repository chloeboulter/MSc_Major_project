import cv2
import numpy as np

def highlight_altered_pixels(original_image_path, modified_image_path, output_image_path):
    # Load the original and modified images
    original_img = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    modified_img = cv2.imread(modified_image_path, cv2.IMREAD_COLOR)

    # Ensure that the images have the same shape
    if original_img.shape != modified_img.shape:
        raise ValueError("The original and modified images must have the same dimensions.")

    # Create a mask where altered pixels are marked
    altered_pixels_mask = (original_img != modified_img).any(axis=2)

    # Create a copy of the original image to highlight altered pixels
    highlighted_img = original_img.copy()

    # Highlight altered pixels in red
    highlighted_img[altered_pixels_mask] = [0, 0, 255]  # Red color in BGR format

    # Save the resulting image
    cv2.imwrite(output_image_path, highlighted_img)

    return highlighted_img

# Example usage
highlighted_img = highlight_altered_pixels("dalle_image.png", "output_image_lsb.png", "highlighted_altered_pixels_lsb.png")
