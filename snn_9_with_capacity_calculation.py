import nest
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
import psutil
from memory_profiler import memory_usage


def text_to_ascii(text):
    ascii_values = [ord(c) for c in text]
    print(f"Text to ASCII: {ascii_values}")
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
        text += text
        current_size_kb = calculate_text_size_in_kb(text)
    return text


def simulate_text_encoding(text, sim_time=50.0):
    ascii_values = text_to_ascii(text)
    num_neurons = len(ascii_values)

    nest.ResetKernel()

    layer1 = nest.Create('iaf_psc_alpha', num_neurons)
    layer2 = nest.Create('iaf_psc_alpha', num_neurons)
    layer3 = nest.Create('iaf_psc_alpha', num_neurons)

    spikerecorder1 = nest.Create("spike_recorder")
    spikerecorder2 = nest.Create("spike_recorder")
    spikerecorder3 = nest.Create("spike_recorder")

    currents = [ascii_to_current(ascii_val) for ascii_val in ascii_values]
    print(f"Currents: {currents}")
    nest.SetStatus(layer1, [{"I_e": current} for current in currents])

    nest.Connect(layer1, layer2, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})
    nest.Connect(layer2, layer3, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})

    nest.Connect(layer1, spikerecorder1)
    nest.Connect(layer2, spikerecorder2)
    nest.Connect(layer3, spikerecorder3)

    cpu_usage_before = psutil.cpu_percent(interval=None)
    mem_usage_before = memory_usage()[0]

    start_time = time.time()
    nest.Simulate(sim_time)
    end_time = time.time()

    cpu_usage_after = psutil.cpu_percent(interval=None)
    mem_usage_after = memory_usage()[0]

    execution_time = end_time - start_time
    cpu_usage = cpu_usage_after - cpu_usage_before
    mem_usage = mem_usage_after - mem_usage_before

    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f} %")
    print(f"Memory Usage: {mem_usage:.2f} MiB")

    events1 = spikerecorder1.get("events")
    events2 = spikerecorder2.get("events")
    events3 = spikerecorder3.get("events")

    senders1, ts1 = events1["senders"], events1["times"]
    senders2, ts2 = events2["senders"], events2["times"]
    senders3, ts3 = events3["senders"], events3["times"]

    print(f"Spike Senders Layer 1: {senders1}")
    print(f"Spike Times Layer 1: {ts1}")
    print(f"Spike Senders Layer 2: {senders2}")
    print(f"Spike Times Layer 2: {ts2}")
    print(f"Spike Senders Layer 3: {senders3}")
    print(f"Spike Times Layer 3: {ts3}")

    delay_L1_L2 = ts2[0] - ts1[0]
    delay_L2_L3 = ts3[0] - ts2[0]
    delay_L1_L3 = ts3[0] - ts1[0]

    np.save("results/text_events_times_layer1.npy", ts1)

    ts3_adjusted = ts3 - delay_L1_L3
    ts2_adjusted = ts2 - delay_L1_L2

    return senders3, ts3_adjusted


def calculate_capacity(image_path, num_spike_pairs, bits_per_channel=1):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    height, width, channels = img.shape

    # Calculate maximum payload in bits
    max_payload_bits = width * height * channels * bits_per_channel

    # Calculate spike data size
    spike_data_size_bits = num_spike_pairs * 2 * 8  # Assuming 2 integers (sender and time) per spike pair

    # Effective capacity
    effective_capacity_bits = max_payload_bits - spike_data_size_bits
    effective_capacity_bytes = effective_capacity_bits // 8

    return effective_capacity_bytes


def embed_spike_data_in_image(image_path, senders, spike_times, output_image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    flat_data = img.flatten()

    spike_data = np.array(list(zip(senders, spike_times.flatten()))).flatten().astype(np.uint8)

    # Calculate and display the capacity
    capacity = calculate_capacity(image_path, len(senders))
    print(f"Calculated capacity: {capacity} bytes")

    if len(spike_data) > len(flat_data):
        raise ValueError("The image is too small to hold the spike data.")

    for i in range(len(spike_data)):
        flat_data[i] = (flat_data[i] & ~1) | (spike_data[i] & 1)

    embedded_img_data = flat_data.reshape(img.shape)
    cv2.imwrite(output_image_path, embedded_img_data)


def extract_spike_data_from_image(image_path, num_spike_pairs):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    flat_data = img.flatten()

    extracted_bits = [flat_data[i] & 1 for i in range(num_spike_pairs * 2)]
    extracted_data = np.array(extracted_bits).reshape((num_spike_pairs, 2))
    senders = extracted_data[:, 0]
    spike_times = extracted_data[:, 1]

    return senders, spike_times


def spikes_to_ascii(decoded_senders, decoded_times, original_senders, original_times, time_threshold=25.0):
    decoded_ascii_values = []

    for orig_sender, orig_time in zip(original_senders, original_times):
        for dec_sender, dec_time in zip(decoded_senders, decoded_times):
            if abs(dec_time - orig_time) <= time_threshold:
                decoded_ascii_values.append(orig_sender)
                break

    return decoded_ascii_values


def ascii_to_text(ascii_values):
    text = ''.join([chr(val) for val in ascii_values])
    print(f"ASCII to Text: {text}")
    return text


def decode_text(ascii_values, extracted_senders, extracted_times):
    original_senders = np.arange(1, len(ascii_values) + 1)
    original_times = np.load("results/text_events_times_layer1.npy")

    decoded_senders_mapped = spikes_to_ascii(extracted_senders, extracted_times, original_senders, original_times)
    decoded_ascii_values = [ascii_values[sender - 1] for sender in decoded_senders_mapped]

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

target_size_kb = 100
text = repeat_text_until_size(initial_text, target_size_kb)
ascii_values = text_to_ascii(text)

senders3, ts3_adjusted = simulate_text_encoding(text)

embed_spike_data_in_image("dalle_image.png", senders3, ts3_adjusted, "image_with_hidden_data_efficiency_test.png")

extracted_senders, extracted_times = extract_spike_data_from_image("image_with_hidden_data_efficiency_test.png",
                                                                   len(senders3))

decoded_text = decode_text(ascii_values, extracted_senders, extracted_times)
print(f"Decoded Text: {decoded_text}, Length of string: {len(decoded_text)}, Length of Original String: {len(text)}")
