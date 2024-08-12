import nest
import numpy as np
import matplotlib.pyplot as plt
import os

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

    # Save adjusted spike data for decoding
    os.makedirs("results", exist_ok=True)
    np.save("results/text_events_senders_layer3.npy", senders3)
    np.save("results/text_events_times_layer3.npy", ts3_adjusted)  # Use adjusted times

    # Plot raster plots for all three layers
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.vlines(ts1, senders1, senders1 + 1, color='black')
    plt.title('Layer 1')
    plt.ylabel('Neuron Index')

    plt.subplot(3, 1, 2)
    plt.vlines(ts2_adjusted, senders2, senders2 + 1, color='blue')
    plt.title('Layer 2 (Adjusted)')
    plt.ylabel('Neuron Index')

    plt.subplot(3, 1, 3)
    plt.vlines(ts3_adjusted, senders3, senders3 + 1, color='red')
    plt.title('Layer 3 (Adjusted)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')

    plt.tight_layout()
    plt.show()

    return events3

def spikes_to_ascii(decoded_senders, decoded_times, original_senders, original_times, time_threshold=2.0):
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

def decode_text(ascii_values):
    # Load the spike data from Layer 3 (decoded layer)
    decoded_senders = np.load("results/text_events_senders_layer3.npy")
    decoded_times = np.load("results/text_events_times_layer3.npy")

    # Original senders and times from the first layer
    original_senders = np.arange(1, len(ascii_values) + 1)  # Assuming senders are 1-indexed
    original_times = np.load("results/text_events_times_layer1.npy")  # Save these in simulate_text_encoding

    # Decode by matching patterns
    decoded_senders_mapped = spikes_to_ascii(decoded_senders, decoded_times, original_senders, original_times)
    decoded_ascii_values = [ascii_values[sender - 1] for sender in decoded_senders_mapped]  # Convert to 0-indexed

    decoded_text = ascii_to_text(decoded_ascii_values)

    return decoded_text

# Example text
text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
ascii_values = text_to_ascii(text)
events = simulate_text_encoding(text)
decoded_text = decode_text(ascii_values)
print(f"Decoded Text: {decoded_text}, Length of string: {len(decoded_text)}, Length of Original String: {len(text)}")
