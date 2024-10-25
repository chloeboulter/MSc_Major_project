import cv2
import numpy as np


def calculate_psnr_24bit(cover_image, stego_image):
    # Convert images to floating point for precise difference calculation
    cover_image = cover_image.astype(np.float64)
    stego_image = stego_image.astype(np.float64)

    # Compute Mean Squared Error (MSE) across all channels
    mse = np.mean((cover_image - stego_image) ** 2)
    if mse == 0:  # No difference between images
        return float('inf')  # PSNR is infinity

    # Maximum pixel value for 8-bit per channel images is 255
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

# Load cover image and stego-image in color
cover_image = cv2.imread('house.png', cv2.IMREAD_COLOR)
stego_image = cv2.imread('house_output_8kb.png', cv2.IMREAD_COLOR)

# Calculate PSNR for 24-bit image
psnr_value = calculate_psnr_24bit(cover_image, stego_image)
print(f"PSNR for 24-bit color image: {psnr_value} dB")