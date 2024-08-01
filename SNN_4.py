def string_to_ascii(s):
    ascii_values = [ord(char) for char in s]
    return ascii_values

def ascii_to_string(ascii_values):
    original_string = ''.join(chr(value) for value in ascii_values)
    return original_string

# Example usage
input_string = "Hello, World!"
print("Original String:", input_string)

# Convert string to ASCII values
ascii_values = string_to_ascii(input_string)
print("ASCII Values:", ascii_values)

# Convert ASCII values back to string
converted_string = ascii_to_string(ascii_values)
print("Converted String:", converted_string)