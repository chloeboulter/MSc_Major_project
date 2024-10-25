import nest
import numpy as np
import os
import cv2
import time
import psutil
from memory_profiler import memory_usage

# Start the total timer
total_start_time = time.time()

def text_to_ascii(text):
    ascii_values = [ord(c) for c in text]
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

    # Create spike recorders for each layer
    spikerecorder1 = nest.Create("spike_recorder")
    spikerecorder2 = nest.Create("spike_recorder")
    spikerecorder3 = nest.Create("spike_recorder")

    # Set currents based on ASCII values for the first layer
    currents = [ascii_to_current(ascii_val) for ascii_val in ascii_values]
    nest.SetStatus(layer1, [{"I_e": current} for current in currents])

    # Connect layers with custom weights in a one-to-one fashion
    nest.Connect(layer1, layer2, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})
    nest.Connect(layer2, layer3, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})

    nest.Connect(layer1, spikerecorder1)
    nest.Connect(layer2, spikerecorder2)
    nest.Connect(layer3, spikerecorder3)

    # Measure CPU and memory usage before simulation
    cpu_usage_before = psutil.cpu_percent(interval=None)
    mem_usage_before = memory_usage()[0]

    # Start timing the simulation
    start_time = time.time()

    # Simulate the network
    nest.Simulate(sim_time)

    # End timing the simulation
    end_time = time.time()

    # Measure CPU and memory usage after simulation
    cpu_usage_after = psutil.cpu_percent(interval=None)
    mem_usage_after = memory_usage()[0]

    # Calculate profiling data
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

    senders3, ts3_adjusted = events3["senders"], events3["times"]

    # Save original Layer 1 times for later comparison
    np.save("results/text_events_times_layer1.npy", events1["times"])

    return senders3, ts3_adjusted

def embed_spike_data_in_multiple_images(base_image_path, senders, spike_times, num_images=50, output_folder="output_images"):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i in range(1, num_images + 1):
        # Load the image
        image_path = os.path.join(base_image_path, f"image_{i}.png")
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Image {image_path} not found. Skipping...")
            continue

        # Flatten the image data to simplify LSB embedding
        flat_data = img.flatten()

        # Combine senders and spike times into a single array of integers
        spike_data = np.array(list(zip(senders, spike_times.flatten()))).flatten().astype(np.uint8)

        # Ensure there's enough space in the image to store the spike data
        if len(spike_data) > len(flat_data):
            print(f"The image {image_path} is too small to hold the spike data.")
            continue

        # Embed spike data into the least significant bits of the image
        for j in range(len(spike_data)):
            flat_data[j] = (flat_data[j] & ~1) | (spike_data[j] & 1)  # Replace LSB with spike data

        # Reshape the data back into the original image shape
        embedded_img_data = flat_data.reshape(img.shape)

        # Save the modified image with a unique file name
        output_image_path = os.path.join(output_folder, f"embedded_image_{i}.png")
        cv2.imwrite(output_image_path, embedded_img_data)
        print(f"Embedded spike data into {output_image_path}")

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
target_size_kb = 8  # Specify the desired size in KB
text = repeat_text_until_size(initial_text, target_size_kb)
ascii_values = text_to_ascii(text)

# Simulate the encoding to get the spike data
senders3, ts3_adjusted = simulate_text_encoding(text)

# Embed the spike data into images located in the 'images' directory
embed_spike_data_in_multiple_images("images", senders3, ts3_adjusted, num_images=50)

# Stop the total timer
total_end_time = time.time()
total_runtime = total_end_time - total_start_time
print(f"Total Runtime: {total_runtime:.2f} seconds")
