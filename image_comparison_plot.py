import matplotlib.pyplot as plt
from PIL import Image

# Load images
image1 = Image.open('dalle_image.png')
image2 = Image.open('image_with_hidden_data_efficiency_test.png')

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image in the first subplot with a label
axes[0].imshow(image1)
axes[0].set_title('Original Image')  # Set label for the first image
axes[0].axis('off')  # Hide the axis

# Display the second image in the second subplot with a label
axes[1].imshow(image2)
axes[1].set_title('Original Image with Hidden Data')  # Set label for the second image
axes[1].axis('off')  # Hide the axis

# Show the plot
plt.show()