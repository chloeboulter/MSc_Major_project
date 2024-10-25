import os


def rename_images_in_directory(directory_path):
    # Get a list of all files in the specified directory
    files = os.listdir(directory_path)

    # Filter out only image files based on extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')  # Add more formats as needed
    image_files = [f for f in files if f.endswith(image_extensions)]

    # Rename each image file
    for index, filename in enumerate(image_files, start=1):
        # Create new file name
        new_name = f"image_{index}.png"  # Change the extension if needed
        old_file_path = os.path.join(directory_path, filename)
        new_file_path = os.path.join(directory_path, new_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_name}'")


# Example usage
directory_path = "/home/ntu-user/PycharmProjects/MSc_Major_project/images"  # Change this to the path where your images are located
rename_images_in_directory(directory_path)
