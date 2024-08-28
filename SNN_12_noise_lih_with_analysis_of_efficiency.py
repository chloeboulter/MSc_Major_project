import nest
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import psutil
from memory_profiler import memory_usage

# Start the total timer
total_start_time = time.time()

def text_to_ascii(text):
    ascii_values = [ord(c) for c in text]
    # print(f"Text to ASCII: {ascii_values}")
    return ascii_values

def ascii_to_current(ascii_val, offset=380):
    return ascii_val + offset

def calculate_text_size_in_kb(text):
    size_in_bytes = len(text.encode('utf-8'))
    size_in_kb = size_in_bytes / 1024
    return size_in_kb

def repeat_text_until_size(text, target_size_kb):
    current_size_kb = calculate_text_size_in_kb(text)
    while current_size_kb < target_size_kb:
        text += text  # Double the text each time
        current_size_kb = calculate_text_size_in_kb(text)
    return text

def simulate_text_encoding(text, sim_time=50.0):
    ascii_values = text_to_ascii(text)
    num_neurons = len(ascii_values)

    # Initialize NEST kernel
    nest.ResetKernel()

    # Create three layers with iaf_psc_alpha neurons
    layer1 = nest.Create('iaf_psc_alpha', num_neurons)
    layer2 = nest.Create('iaf_psc_alpha', num_neurons)
    layer3 = nest.Create('iaf_psc_alpha', num_neurons)
    noise_layer = nest.Create("poisson_generator", num_neurons)
    lateral_ih_layer = nest.Create("iaf_psc_alpha", num_neurons)

    nest.SetStatus(noise_layer, {"rate": 100.0})

    # Create spike recorders for each layer
    spikerecorder1 = nest.Create("spike_recorder")
    spikerecorder2 = nest.Create("spike_recorder")
    spikerecorder3 = nest.Create("spike_recorder")
    noise_spikerecorder = nest.Create("spike_recorder")
    spike_recorder_lateral_ih_layer = nest.Create("spike_recorder")

    # Set currents based on ASCII values for the first layer
    currents = [ascii_to_current(ascii_val) for ascii_val in ascii_values]
    # print(f"Currents: {currents}")
    nest.SetStatus(layer1, [{"I_e": current} for current in currents])

    # Connect layers with custom weights in a one-to-one fashion
    nest.Connect(layer1, layer2, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})
    nest.Connect(layer2, layer3, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})

    nest.Connect(noise_layer, layer2, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})

    nest.Connect(noise_layer, lateral_ih_layer, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})
    nest.Connect(lateral_ih_layer, layer2, syn_spec={"weight": -1500.0}, conn_spec={"rule": "one_to_one"})

    nest.Connect(layer1, spikerecorder1)
    nest.Connect(layer2, spikerecorder2)
    nest.Connect(layer3, spikerecorder3)
    nest.Connect(noise_layer, noise_spikerecorder)
    nest.Connect(lateral_ih_layer, spike_recorder_lateral_ih_layer)

    # Measure CPU usage before simulation
    cpu_usage_before = psutil.cpu_percent(interval=None)

    # Measure memory usage before simulation
    mem_usage_before = memory_usage()[0]

    # Start timing the simulation
    start_time = time.time()

    # Simulate the network
    nest.Simulate(sim_time)

    # End timing the simulation
    end_time = time.time()

    # Measure CPU usage after simulation
    cpu_usage_after = psutil.cpu_percent(interval=None)

    # Measure memory usage after simulation
    mem_usage_after = memory_usage()[0]

    # Calculate and print profiling data
    execution_time = end_time - start_time
    cpu_usage = cpu_usage_after - cpu_usage_before
    mem_usage = mem_usage_after - mem_usage_before

    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f} %")
    print(f"Memory Usage: {mem_usage:.2f} MiB")

    # Retrieve events
    events1 = spikerecorder1.get("events")
    events2 = spikerecorder2.get("events")
    events3 = spikerecorder3.get("events")
    noise_events = noise_spikerecorder.get("events")
    lateral_ih_events = spike_recorder_lateral_ih_layer.get("events")

    senders1, ts1 = events1["senders"], events1["times"]
    senders2, ts2 = events2["senders"], events2["times"]
    senders3, ts3 = events3["senders"], events3["times"]
    noise_senders, noise_ts = noise_events["senders"], noise_events["times"]
    lateral_ih_senders, lateral_ih_ts = lateral_ih_events["senders"], lateral_ih_events["times"]

    # print(f"Spike Senders Layer 1: {senders1}")
    # print(f"Spike Times Layer 1: {ts1}")
    # print(f"Spike Senders Layer 2: {senders2}")
    # print(f"Spike Times Layer 2: {ts2}")
    # print(f"Spike Senders Layer 3: {senders3}")
    # print(f"Spike Times Layer 3: {ts3}")
    # print(f"Noise Spike Senders: {noise_senders}")
    # print(f"Noise Spike Times: {noise_ts}")
    # print(f"Lateral Inhibition Spike Senders: {lateral_ih_senders}")
    # print(f"Lateral Inhibition Spike Times: {lateral_ih_ts}")

    # Calculate propagation delays
    delay_L1_L2 = ts2[0] - ts1[0]
    delay_L2_L3 = ts3[0] - ts2[0]
    delay_L1_L3 = ts3[0] - ts1[0]

    # Save original Layer 1 times for later comparison
    np.save("results/text_events_times_noise_layer1.npy", ts1)
    np.save("results/text_events_times_noise_layer2.npy", ts2)
    np.save("results/text_events_times_noise_layer3.npy", ts3)

    # Adjust spike times for decoding
    ts3_adjusted = ts3 - delay_L1_L3
    ts2_adjusted = ts2 - delay_L1_L2

    return senders3, ts3_adjusted

def embed_spike_data_in_image(image_path, senders, spike_times, output_image_path):
    # Start timing the encoding process
    start_time = time.time()

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Flatten the image data to simplify LSB embedding
    flat_data = img.flatten()

    # Combine senders and spike times into a single array of integers
    spike_data = np.array(list(zip(senders, spike_times.flatten()))).flatten().astype(np.uint8)

    # Ensure there's enough space in the image to store the spike data
    if len(spike_data) > len(flat_data):
        raise ValueError("The image is too small to hold the spike data.")

    # Embed spike data into the least significant bits of the image
    for i in range(len(spike_data)):
        flat_data[i] = (flat_data[i] & ~1) | (spike_data[i] & 1)  # Replace LSB with spike data

    # Reshape the data back into the original image shape
    embedded_img_data = flat_data.reshape(img.shape)

    # Save the modified image
    cv2.imwrite(output_image_path, embedded_img_data)

    # End timing the encoding process
    end_time = time.time()

    # Calculate the encoding time and data rate
    encoding_time = end_time - start_time
    total_bits_embedded = len(spike_data) * 8  # Each byte has 8 bits

    data_rate = total_bits_embedded / encoding_time
    print(f"Encoding Data Rate: {data_rate:.2f} bps")

def extract_spike_data_from_image(image_path, num_spike_pairs):
    # Start timing the decoding process
    start_time = time.time()

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Flatten the image data to simplify LSB extraction
    flat_data = img.flatten()

    # Extract the spike data from the LSB of the image data
    extracted_bits = [flat_data[i] & 1 for i in range(num_spike_pairs * 2)]

    # Convert the bits back into integers
    extracted_data = np.array(extracted_bits).reshape((num_spike_pairs, 2))
    senders = extracted_data[:, 0]
    spike_times = extracted_data[:, 1]

    # End timing the decoding process
    end_time = time.time()

    # Calculate the decoding time and data rate
    decoding_time = end_time - start_time
    total_bits_extracted = num_spike_pairs * 8 * 2  # Each pair of sender and spike time has 16 bits

    data_rate = total_bits_extracted / decoding_time
    print(f"Decoding Data Rate: {data_rate:.2f} bps")

    return senders, spike_times

def spikes_to_ascii(decoded_senders, decoded_times, original_senders, original_times, time_threshold=25.0):
    decoded_ascii_values = []

    for orig_sender, orig_time in zip(original_senders, original_times):
        for dec_sender, dec_time in zip(decoded_senders, decoded_times):
            if abs(dec_time - orig_time) <= time_threshold:
                decoded_ascii_values.append(orig_sender)
                break  # Move to the next original sender once a match is found

    return decoded_ascii_values

def ascii_to_text(ascii_values):
    text = ''.join([chr(val) for val in ascii_values])
    # print(f"ASCII to Text: {text}")
    return text

def decode_text(ascii_values, extracted_senders, extracted_times):
    original_senders = np.arange(1, len(ascii_values) + 1)  # Assuming senders are 1-indexed
    original_times = np.load("results/text_events_times_layer1.npy")  # Save these in simulate_text_encoding

    # Decode by matching patterns
    decoded_senders_mapped = spikes_to_ascii(extracted_senders, extracted_times, original_senders, original_times)
    decoded_ascii_values = [ascii_values[sender - 1] for sender in decoded_senders_mapped]  # Convert to 0-indexed

    decoded_text = ascii_to_text(decoded_ascii_values)

    return decoded_text

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
target_size_kb = 55  # Specify the desired size in KB
text = repeat_text_until_size(initial_text, target_size_kb)
ascii_values = text_to_ascii(text)

# Simulate the encoding to get the spike data
senders3, ts3_adjusted = simulate_text_encoding(text)

# Embed the spike data into an image
embed_spike_data_in_image("dalle_image.png", senders3, ts3_adjusted, "converted_image.jpg")

# Extract the spike data back from the image
extracted_senders, extracted_times = extract_spike_data_from_image("converted_image.jpg", len(senders3))

# Decode the text from the extracted spike data
decoded_text = decode_text(ascii_values, extracted_senders, extracted_times)
# (f"Decoded Text: {decoded_text}, Length of string: {len(decoded_text)}, Length of Original String: {len(text)}")

# Stop the total timer
total_end_time = time.time()
total_runtime = total_end_time - total_start_time
print(f"Total Runtime: {total_runtime:.2f} seconds")