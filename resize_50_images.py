import cv2
import os

def resize_images_in_place(folder, new_size=(256, 256)):
    # Loop through all files in the specified folder
    for filename in os.listdir(folder):
        # Construct the full file path
        file_path = os.path.join(folder, filename)

        # Check if the file is an image (you can add more formats if needed)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"Skipping non-image file: {file_path}")
            continue

        # Read the image
        img = cv2.imread(file_path)

        # Check if the image was loaded successfully
        if img is None:
            print(f"Warning: {file_path} could not be loaded.")
            continue

        # Resize the image
        resized_img = cv2.resize(img, new_size)

        # Save the resized image back to the original path
        cv2.imwrite(file_path, resized_img)
        print(f"Resized and overwritten: {file_path}")

# Example usage
image_directory = 'images'  # Replace with your image directory
resize_images_in_place(image_directory)
