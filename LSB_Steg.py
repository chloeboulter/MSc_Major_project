from PIL import Image


def convert_webp_to_png(input_path, output_path):
    try:
        with Image.open(input_path) as webp_image:
            webp_image.save(output_path, 'PNG')
        print(f"Image successfully converted to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def calculate_text_size_in_kb(text):
    size_in_bytes = len(text.encode('utf-8'))
    size_in_kb = size_in_bytes / 1024
    return size_in_kb


def repeat_text_until_size(text, target_size_kb):
    current_size_kb = calculate_text_size_in_kb(text)
    while current_size_kb < target_size_kb:
        text += text
        current_size_kb = calculate_text_size_in_kb(text)
    return text


def calculate_capacity(image_path):
    image = Image.open(image_path)
    width, height = image.size
    total_pixels = width * height

    # Total bits available for encoding (3 bits per pixel)
    total_bits = total_pixels * 3

    # Convert bits to maximum number of characters that can be encoded
    max_chars = total_bits // 8

    return max_chars


def embed_text_in_image(image_path, text, output_image_path):
    image = Image.open(image_path)
    encoded = image.copy()
    width, height = image.size
    index = 0

    binary_text = ''.join([format(ord(i), "08b") for i in text])
    delimiter = '00000000'  # Null character as a delimiter
    binary_text += delimiter

    for row in range(height):
        for col in range(width):
            if index < len(binary_text):
                pixel = list(image.getpixel((col, row)))
                for n in range(3):
                    if index < len(binary_text):
                        pixel[n] = pixel[n] & ~1 | int(binary_text[index])
                        index += 1
                encoded.putpixel((col, row), tuple(pixel))

    encoded.save(output_image_path)
    print("Text has been successfully embedded into the image.")


def extract_text_from_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    binary_text = ""

    for row in range(height):
        for col in range(width):
            pixel = image.getpixel((col, row))
            for n in range(3):
                binary_text += str(pixel[n] & 1)

    all_bytes = [binary_text[i: i + 8] for i in range(0, len(binary_text), 8)]

    decoded_text = ""
    for byte in all_bytes:
        decoded_text += chr(int(byte, 2))
        if decoded_text[-1] == '\x00':
            break

    return decoded_text[:-1]


# Example usage for converting image formats
convert_webp_to_png('dalle_image.webp', 'dalle_image.png')

# Calculate the maximum capacity for embedding text in the image
image_path = 'dalle_image.png'
max_chars = calculate_capacity(image_path)
print(f"The maximum number of characters that can be embedded in the image is: {max_chars} characters")

# Example usage to reach a specific size in KB
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

# Ensure the text does not exceed the capacity
target_size_kb = 100
# text = repeat_text_until_size(initial_text, target_size_kb)
text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"

if len(text) <= max_chars:
    embed_text_in_image(image_path, text, 'output_image_lsb.png')
else:
    print(f"Text is too long to embed in the image. Max capacity is {max_chars} characters.")

# Extract the hidden text from the image
hidden_text = extract_text_from_image('output_image_lsb.png')
print("Hidden text:", hidden_text)
