import nest
import numpy as np
import cv2

def create_snn_layer(num_neurons, neuron_model="iaf_psc_alpha", neuron_params=None):
    if neuron_params is None:
        neuron_params = {"tau_m": 20.0}
    neurons = nest.Create(neuron_model, num_neurons, params=neuron_params)
    return neurons


# Create multiple layers
layer1 = create_snn_layer(100)
layer2 = create_snn_layer(100)
layer3 = create_snn_layer(100)


def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)
# Loops through characters in string and converts to ordinal, then to binary and joins the binary for each char together


def binary_to_text(binary):
    text = ''.join(chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8))
    return text
# Loops through binary string in chunks of 8 bits, extracts 8-bit chunk and convert binary string to int, converts int
# to character and joins chars together


def binary_to_spikes(binary_data, neurons, duration=100.0):
    spike_times = []  # Initialise spike times list
    for i, bit in enumerate(binary_data):  # Loop through binary data
        if bit == '1':
            spike_times.append(i * duration)
    spike_generators = nest.Create("spike_generator", neurons, params={"spike_times": spike_times})
    # Create spike generators, spikes occur based on spike_times
    return spike_generators


def spikes_to_binary(spike_generators, binary_length):
    spike_times = nest.GetStatus(spike_generators, "spike_times")[0]  # Retrieve spike times
    binary_data = ['0'] * binary_length  # Creates a list for binary data with all 0s of binary_length
    for t in spike_times:  # Calculates index
        index = int(t // 100.0)  # Assumes spikes encoded with duration of 100ms
        if index < binary_length:
            binary_data[index] = '1'
            # if index is within range binary_length, sets binary_data[index] to 1, indicating spike
    return ''.join(binary_data)


# Function to connect and simulate layers
def simulate_snn_layer(input_spike_generators, target_layer, simulation_time=100.0):
    syn_dict = {"weight": 1.0, "delay": 1.0}
    nest.Connect(input_spike_generators, target_layer, syn_spec=syn_dict)
    nest.Simulate(simulation_time)

simulate_snn_layer(spike_generators_layer1, layer1)
simulate_snn_layer(layer1, layer2)
simulate_snn_layer(layer2, layer3)



def embed_text_in_image(image_path, binary_text, output_image_path):
    image = cv2.imread(image_path)  # Reads image
    data_index = 0  # Initialise index
    text_length = len(binary_text)  # Length of the binary text data
    height, width, channels = image.shape  # Dimensions of the image

    for row in range(height):
        for col in range(width):
            for channel in range(channels):  # Iterate over the R, G, B channels
                if data_index < text_length:  # Check if there is still data to embed
                    pixel_value = image[row, col, channel]  # Get current pixel value
                    lsb = int(binary_text[data_index])  # Get the LSB of binary_text at data_index

                    # Clear the LSB of pixel_value and set it to lsb
                    modified_pixel_value = (pixel_value & 254) | lsb  # Ensure LSB is 0 in pixel_value
                    image[row, col, channel] = modified_pixel_value
                    data_index += 1  # Move to next bit of binary_text

                if data_index >= text_length:  # Stop embedding if all data embedded
                    break
            if data_index >= text_length:
                break
        if data_index >= text_length:
            break

    cv2.imwrite(output_image_path, image)  # Save the modified image to output_image_path



def extract_text_from_image(image_path, text_length):
    image = cv2.imread(image_path)  # Read the input image
    binary_text = ''  # Initialize empty string to hold binary text
    data_index = 0  # Initialize index
    height, width, channels = image.shape  # Get dimensions
    total_bits = text_length * 8  # Calculate total number of bits to extract

    # Traverse each pixel of image
    for row in range(height):
        for col in range(width):
            for channel in range(channels):  # Read the R, G, B channels
                if data_index < total_bits:
                    pixel_value = image[row, col, channel]  # Get current pixel value
                    # Extract the LSB (bitwise AND with 1) and append to binary_text
                    binary_text += str(pixel_value & 1)
                    data_index += 1  # Move to next bit

                if data_index >= total_bits:  # Stop extraction if all data has been retrieved
                    break
            if data_index >= total_bits:
                break
        if data_index >= total_bits:
            break

    return binary_text


def encode_with_snn(image_path, text, output_image_path):
    binary_text = text_to_binary(text)  # convert text to binary
    binary_length = len(binary_text)  # calculate length of binary text

    neurons = create_snn()  # Create Spiking Neural Network and get list of neurons

    # Convert binary text to spike trains using created SNN neurons
    spike_generators = binary_to_spikes(binary_text, len(neurons))

    # Simulate the SNN for 1000.0 milliseconds
    nest.Simulate(1000.0)

    # Convert spike trains back to binary data
    embedded_spikes = spikes_to_binary(spike_generators, binary_length)

    # Embed the binary data into the image
    embed_text_in_image(image_path, embedded_spikes, output_image_path)


def decode_with_snn(image_path, text_length):
    binary_length = text_length * 8  # calculate number of bits to extract

    neurons = create_snn()  # Create Spiking Neural Network and get list of neurons

    # Extract binary data from the image
    extracted_binary = extract_text_from_image(image_path, text_length)

    # Convert extracted binary data to spike trains using created SNN neurons
    spike_generators = binary_to_spikes(extracted_binary, len(neurons))

    # Simulate the SNN for 1000.0 milliseconds
    nest.Simulate(1000.0)

    # Convert spike trains back to binary data
    decoded_spikes = spikes_to_binary(spike_generators, binary_length)

    # Convert decoded binary data back to text
    return binary_to_text(decoded_spikes)


input_image_path = '/home/bloors/MSc-Major-Project/images/4.1.01.tiff'
output_image_path = 'dalle_image.png'
hidden_text = 'Hidden Text'

encode_with_snn(input_image_path, hidden_text, output_image_path)
extracted_text = decode_with_snn(output_image_path, len(hidden_text))
print(f"Extracted Text: {extracted_text}")
