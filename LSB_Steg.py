from PIL import Image


def convert_webp_to_png(input_path, output_path):
    try:
        # Open the webp image
        with Image.open(input_path) as webp_image:
            # Save it as a png image
            webp_image.save(output_path, 'PNG')
        print(f"Image successfully converted to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
convert_webp_to_png('dalle_image.webp', 'dalle_image.png')


def embed_text_in_image(image_path, text, output_image_path):
    image = Image.open(image_path)
    encoded = image.copy()
    width, height = image.size
    index = 0

    # Convert text to binary
    binary_text = ''.join([format(ord(i), "08b") for i in text])

    # Add delimiter to the binary text
    delimiter = '00000000'  # Null character as a delimiter
    binary_text += delimiter

    for row in range(height):
        for col in range(width):
            if index < len(binary_text):
                pixel = list(image.getpixel((col, row)))
                for n in range(3):  # Iterate over RGB channels
                    if index < len(binary_text):
                        pixel[n] = pixel[n] & ~1 | int(binary_text[index])
                        index += 1
                encoded.putpixel((col, row), tuple(pixel))

    encoded.save(output_image_path)
    print("Text has been successfully embedded into the image.")


# Example usage
embed_text_in_image('dalle_image.png', 'Your secret message', 'output_image.png')

from PIL import Image


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    binary_text = ""

    for row in range(height):
        for col in range(width):
            pixel = image.getpixel((col, row))
            for n in range(3):  # Iterate over RGB channels
                binary_text += str(pixel[n] & 1)

    # Split by 8-bits
    all_bytes = [binary_text[i: i + 8] for i in range(0, len(binary_text), 8)]

    # Convert binary to characters
    decoded_text = ""
    for byte in all_bytes:
        decoded_text += chr(int(byte, 2))
        if decoded_text[-1] == '\x00':  # Null character as delimiter
            break

    return decoded_text[:-1]  # Remove the delimiter


# Example usage
hidden_text = extract_text_from_image('output_image.png')
print("Hidden text:", hidden_text)
