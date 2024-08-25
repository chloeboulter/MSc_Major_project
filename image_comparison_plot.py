import matplotlib.pyplot as plt
from PIL import Image

# Load images
image1 = Image.open('dalle_image.png')
image2 = Image.open('output_image_lsb.png')
image3 = Image.open('highlighted_altered_pixels_lsb.png')  # Load the third image

# Create a figure with three subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Display the first image in the first subplot with a label
axes[0].imshow(image1)
axes[0].set_title('Original Image')  # Set label for the first image
axes[0].axis('off')  # Hide the axis

# Display the second image in the second subplot with a label
axes[1].imshow(image2)
axes[1].set_title('Stego-image with hidden data')  # Set label for the second image
axes[1].axis('off')  # Hide the axis

# Display the third image in the third subplot with a label
axes[2].imshow(image3)
axes[2].set_title('Stego image showing which pixels\n have been altered to hide the text')  # Set label for the third image
axes[2].axis('off')  # Hide the axis

# Show the plot
plt.show()
