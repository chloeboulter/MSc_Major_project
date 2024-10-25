import cv2
import numpy as np


def calculate_rmse(cover_image, stego_image):
    # Convert images to floating-point to ensure precise calculations
    cover_image = cover_image.astype(np.float64)
    stego_image = stego_image.astype(np.float64)

    # Compute MSE
    mse = np.mean((cover_image - stego_image) ** 2)

    # Calculate RMSE
    rmse = np.sqrt(mse)
    return rmse

# Load cover image and stego-image in color
cover_image = cv2.imread('house.png', cv2.IMREAD_COLOR)
stego_image = cv2.imread('house_output_8kb.png', cv2.IMREAD_COLOR)

# Calculate RMSE
rmse_value = calculate_rmse(cover_image, stego_image)
print(f"RMSE: {rmse_value}")