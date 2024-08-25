import cv2

# Read the PNG image
image = cv2.imread('image_with_hidden_data_efficiency_test.png')

# Set JPEG quality (0-100, higher means better quality and larger file size)
jpeg_quality = 85

# Convert and save as JPEG
cv2.imwrite('converted_image.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
