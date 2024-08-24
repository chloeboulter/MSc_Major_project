from PIL import Image

def extract_lsb(image_path):
    """Extracts the LSBs from a color image and decodes the message.

    Args:
        image_path: The path to the image file.

    Returns:
        The decoded message.
    """

    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = img.load()

    message = ""
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            message += str(r & 1)
            message += str(g & 1)
            message += str(b & 1)

    # Assuming the message is terminated by a specific sequence (e.g., 'END')
    end_marker = "11111111"
    message_index = message.find(end_marker)
    if message_index != -1:
        message = message[:message_index]

    # Convert binary message to ASCII characters
    decoded_message = ""
    for i in range(0, len(message), 8):
        byte = message[i:i+8]
        decoded_message += chr(int(byte, 2))

    return decoded_message

# Example usage
image_path = "output_image.png"
decoded_message = extract_lsb(image_path)
print("Decoded Message: " + decoded_message)

