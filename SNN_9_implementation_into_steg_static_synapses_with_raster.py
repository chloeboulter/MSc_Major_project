import nest
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def text_to_ascii(text):
    ascii_values = [ord(c) for c in text]
    print(f"Text to ASCII: {ascii_values}")
    return ascii_values


def ascii_to_current(ascii_val, offset=380):
    return ascii_val + offset


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
    print(f"Currents: {currents}")
    nest.SetStatus(layer1, [{"I_e": current} for current in currents])

    # Connect layers with custom weights in a one-to-one fashion
    nest.Connect(layer1, layer2, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})
    nest.Connect(layer2, layer3, syn_spec={"weight": 1500.0}, conn_spec={"rule": "one_to_one"})

    nest.Connect(layer1, spikerecorder1)
    nest.Connect(layer2, spikerecorder2)
    nest.Connect(layer3, spikerecorder3)

    # Simulate the network
    nest.Simulate(sim_time)

    # Retrieve events
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

    # Calculate propagation delays
    delay_L1_L2 = ts2[0] - ts1[0]
    delay_L2_L3 = ts3[0] - ts2[0]
    delay_L1_L3 = ts3[0] - ts1[0]

    # Save original Layer 1 times for later comparison
    np.save("results/text_events_times_layer1.npy", ts1)

    # Adjust spike times for decoding
    ts3_adjusted = ts3 - delay_L1_L3
    ts2_adjusted = ts2 - delay_L1_L2

    plot_raster([senders1, senders2, senders3],
                [ts1, ts2_adjusted, ts3_adjusted],
                sim_time)
    return senders3, ts3_adjusted



def plot_raster(senders, times, layer_name):
    """
    Generates a raster plot for a given layer's spike times.

    Parameters:
    - senders: The neuron IDs that emitted the spikes.
    - times: The times at which the spikes occurred.
    - layer_name: The name of the layer (used for the plot title).
    """
    plt.figure(figsize=(10, 6))
    plt.eventplot(times, lineoffsets=senders, colors='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.title(f'Raster Plot for {layer_name}')
    plt.show()


def embed_spike_data_in_image(image_path, senders, spike_times, output_image_path):
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


def extract_spike_data_from_image(image_path, num_spike_pairs):
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

    return senders, spike_times


def plot_raster(senders_list, times_list, sim_time):
    fig, axs = plt.subplots(len(senders_list), 1, figsize=(10, 10), sharex=True)

    for i, (senders, times) in enumerate(zip(senders_list, times_list)):
        if len(senders) == 0:
            # Handle the case where there are no spikes recorded for a layer
            axs[i].text(0.5, 0.5, 'No spikes recorded', transform=axs[i].transAxes,
                        ha='center', va='center', fontsize=12, color='red')
            axs[i].set_ylabel(f'Layer {i+1}')
            axs[i].set_xlim([0, sim_time])
            axs[i].set_ylim([0, 1])
            axs[i].grid(True)
        else:
            # Extract only the relevant senders and adjust y-axis limits accordingly
            unique_senders = np.unique(senders)
            axs[i].scatter(times, senders, s=2)
            axs[i].set_ylabel(f'Layer {i+1}')
            axs[i].set_xlim([0, sim_time])
            axs[i].set_ylim([min(unique_senders) - 0.5, max(unique_senders) + 0.5])
            axs[i].grid(True)

    axs[-1].set_xlabel('Time (ms)')
    plt.suptitle('Raster Plot of All Layers, encoding of the text string:\n "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def spikes_to_ascii(decoded_senders, decoded_times, original_senders, original_times, time_threshold=25.0):
    """
    Decodes spike data by matching the spike patterns in the decoded layer
    with the original spike patterns in the encoding layer using time-based matching.
    """
    decoded_ascii_values = []

    for orig_sender, orig_time in zip(original_senders, original_times):
        for dec_sender, dec_time in zip(decoded_senders, decoded_times):
            # If the decoded spike time is within a threshold of the original spike time
            if abs(dec_time - orig_time) <= time_threshold:
                decoded_ascii_values.append(orig_sender)
                break  # Move to the next original sender once a match is found

    return decoded_ascii_values


def ascii_to_text(ascii_values):
    text = ''.join([chr(val) for val in ascii_values])
    print(f"ASCII to Text: {text}")
    return text


def decode_text(ascii_values, extracted_senders, extracted_times):
    # Original senders and times from the first layer
    original_senders = np.arange(1, len(ascii_values) + 1)  # Assuming senders are 1-indexed
    original_times = np.load("results/text_events_times_layer1.npy")  # Save these in simulate_text_encoding

    # Decode by matching patterns
    decoded_senders_mapped = spikes_to_ascii(extracted_senders, extracted_times, original_senders, original_times)
    decoded_ascii_values = [ascii_values[sender - 1] for sender in decoded_senders_mapped]  # Convert to 0-indexed

    decoded_text = ascii_to_text(decoded_ascii_values)

    return decoded_text


# Example text to hide in an image
text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
ascii_values = text_to_ascii(text)

# Simulate the encoding to get the spike data
senders3, ts3_adjusted = simulate_text_encoding(text)

# Embed the spike data into an image
embed_spike_data_in_image("dalle_image.png", senders3, ts3_adjusted, "image_with_hidden_data.png")

# Extract the spike data back from the image
extracted_senders, extracted_times = extract_spike_data_from_image("image_with_hidden_data.png", len(senders3))

# Decode the text from the extracted spike data
decoded_text = decode_text(ascii_values, extracted_senders, extracted_times)
print(f"Decoded Text: {decoded_text}, Length of string: {len(decoded_text)}, Length of Original String: {len(text)}")
