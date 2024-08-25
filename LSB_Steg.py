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


def calculate_text_size_in_kb(text):
    # Calculate the size of the text in bytes
    size_in_bytes = len(text.encode('utf-8'))
    # Convert bytes to kilobytes
    size_in_kb = size_in_bytes / 1024
    return size_in_kb

def repeat_text_until_size(text, target_size_kb):
    current_size_kb = calculate_text_size_in_kb(text)
    while current_size_kb < target_size_kb:
        text += text  # Double the text each time
        current_size_kb = calculate_text_size_in_kb(text)
    return text

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

initial_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin aliquet, urna nec laoreet dapibus, 
mi sapien cursus libero, non pulvinar nulla felis a purus. Cras convallis ex sed cursus iaculis. 
Sed vestibulum turpis ut risus condimentum, in pretium quam scelerisque. Suspendisse potenti. 
Maecenas ultricies velit quis ligula tempus, non fermentum sapien faucibus. Nulla facilisi. 
Praesent nec ultricies magna, sit amet lacinia purus. Aenean aliquam, nisi ut feugiat vulputate, 
turpis orci fermentum nulla, in laoreet sem turpis sit amet neque. Integer porttitor sapien nec 
quam aliquet, id volutpat risus venenatis. Donec euismod fringilla orci non commodo. In hac 
habitasse platea dictumst. Sed pharetra purus at magna malesuada, non suscipit ante eleifend. 
Etiam non tempor sapien. Proin sit amet auctor odio, a varius erat. Suspendisse pharetra 
feugiat magna non malesuada. Aliquam erat volutpat.
"""

# Example usage to reach a specific size in KB
target_size_kb = 100  # Specify the desired size in KB
text = repeat_text_until_size(initial_text, target_size_kb)

# Example usage
embed_text_in_image('dalle_image.png', text, 'output_image_lsb.png')

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
hidden_text = extract_text_from_image('output_image_lsb.png')
print("Hidden text:", hidden_text)
