import cv2

# Step 1: Load the TIFF image
input_path = 'house.tiff'  # Path to the .tiff file
image = cv2.imread(input_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
else:
    # Step 2: Resize the image
    new_width = 256  # Set desired width
    new_height = 256  # Set desired height
    resized_image = cv2.resize(image, (new_width, new_height))

    # Step 3: Save the resized image as PNG
    output_path = 'house.png'  # Path to save the .png file
    cv2.imwrite(output_path, resized_image)
    print("Image successfully converted and resized.")