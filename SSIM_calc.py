import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_ssim_24bit(cover_image, stego_image):
    # Ensure images are in the correct floating-point format for SSIM calculation
    cover_image = cover_image.astype(np.float64)
    stego_image = stego_image.astype(np.float64)

    # Calculate SSIM for each color channel (R, G, B) separately
    ssim_channels = []
    for channel in range(3):  # Loop through R, G, B channels
        ssim_value = ssim(cover_image[:, :, channel], stego_image[:, :, channel],
                          data_range=cover_image[:, :, channel].max() - cover_image[:, :, channel].min())
        ssim_channels.append(ssim_value)

    # Average SSIM across channels
    ssim_avg = np.mean(ssim_channels)
    return ssim_avg

# Load cover image and stego-image in color
cover_image = cv2.imread('house.png', cv2.IMREAD_COLOR)
stego_image = cv2.imread('house_output_8kb.png', cv2.IMREAD_COLOR)

# Calculate SSIM for 24-bit image
ssim_value = calculate_ssim_24bit(cover_image, stego_image)
print(f"SSIM for 24-bit color image: {ssim_value}")