import cv2
import numpy as np


def calculate_capacity(image):
    # Determine the number of pixels or blocks available for embedding
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate maximum number of bits that can be embedded
    max_bits = total_pixels * 3  # Assuming 3 LSBs per pixel (for RGB images)

    return max_bits


def evaluate_security(image, stego_image):
    # Example: Compare histograms of original and stego images
    hist_original, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    hist_stego, _ = np.histogram(stego_image.flatten(), bins=256, range=(0, 256))

    # Calculate correlation or distance metrics between histograms
    correlation = np.corrcoef(hist_original, hist_stego)[0, 1]

    return correlation


def add_noise(image, noise_level=0.1):
    """
    Add Gaussian noise to the image.

    Parameters:
    - image: Input image (numpy array)
    - noise_level: Standard deviation of Gaussian noise (default: 0.1)

    Returns:
    - Noisy image (numpy array)
    """
    noise = np.random.normal(scale=noise_level, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


# Function to extract data from a noisy image (example implementation)
def extract_data(image):
    """
    Example function to extract data from an image.

    Parameters:
    - image: Noisy image (numpy array)

    Returns:
    - Extracted data (example)
    """
    # Example: Extract data using image processing techniques
    extracted_data = image.mean()  # Example: Mean intensity as extracted data
    return extracted_data


def evaluate_robustness(original_image, processed_image):
    """
    Evaluate robustness by adding noise to the processed image
    and measuring the impact on extracted data accuracy.

    Parameters:
    - original_image: Original image (numpy array)
    - processed_image: Processed image (stego image) (numpy array)

    Returns:
    - Error rate or accuracy of extracted data
    """
    noisy_image = add_noise(processed_image)
    extracted_data = extract_data(noisy_image)

    # Example: Calculate error rate or accuracy compared to original data
    original_data = extract_data(original_image)
    error_rate = np.abs(original_data - extracted_data)

    return error_rate


def calculate_psnr(original_image, stego_image):
    mse = np.mean((original_image - stego_image) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr



original_image = cv2.imread('/home/bloors/MSc-Major-Project/images/4.1.01.tiff', cv2.IMREAD_GRAYSCALE)
processed_image = cv2.imread('/dalle_image.png', cv2.IMREAD_GRAYSCALE)

stego_image = cv2.imread('/dalle_image.png')
cover_image = cv2.imread('/home/bloors/MSc-Major-Project/images/4.1.01.tiff')

error_rate = evaluate_robustness(original_image, processed_image)
print(f"Error Rate: {error_rate}")

psnr_value = calculate_psnr(cover_image, stego_image)
print(f"PSNR: {psnr_value}")


capacity = calculate_capacity(stego_image)
print(f"Capacity: {capacity} bits")

security_score = evaluate_security(original_image, stego_image)
print(f"Security Score: {security_score}")
