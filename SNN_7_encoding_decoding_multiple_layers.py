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
    sr1 = nest.Create("spike_recorder")
    sr2 = nest.Create("spike_recorder")
    sr3 = nest.Create("spike_recorder")

    # Set currents based on ASCII values for layer1
    currents = [ascii_to_current(ascii_val) for ascii_val in ascii_values]
    print(f"Currents: {currents}")
    nest.SetStatus(layer1, [{"I_e": current} for current in currents])

    # Connect layers
    conn_dict = {'rule': 'one_to_one'}
    syn_dict = {'weight': 1500.0, 'delay': 1.0}
    nest.Connect(layer1, layer2, conn_dict, syn_dict)
    nest.Connect(layer2, layer3, conn_dict, syn_dict)

    # Connect layers to their respective spike recorders
    nest.Connect(layer1, sr1)
    nest.Connect(layer2, sr2)
    nest.Connect(layer3, sr3)

    # Simulate
    nest.Simulate(sim_time)

    # Collect events from all spike recorders
    events1 = sr1.get("events")
    events2 = sr2.get("events")
    events3 = sr3.get("events")

    # Save events from layer3 for decoding
    os.makedirs("results", exist_ok=True)
    np.save("results/text_events_senders.npy", events3["senders"])
    np.save("results/text_events_times.npy", events3["times"])

    # Create raster plot for all three layers
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    ax1.vlines(events1["times"], events1["senders"], events1["senders"] + 1, color='blue')
    ax1.set_ylabel('Layer 1 Neuron Index')
    ax1.set_title('Raster Plot for Layer 1')

    ax2.vlines(events2["times"], events2["senders"], events2["senders"] + 1, color='green')
    ax2.set_ylabel('Layer 2 Neuron Index')
    ax2.set_title('Raster Plot for Layer 2')

    ax3.vlines(events3["times"], events3["senders"], events3["senders"] + 1, color='red')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Layer 3 Neuron Index')
    ax3.set_title('Raster Plot for Layer 3')

    plt.tight_layout()
    plt.show()

    return events3

# Rest of the code remains the same

def decode_text(ascii_values):
    senders = np.load("results/text_events_senders.npy")
    times = np.load("results/text_events_times.npy")
    decoded_ascii_values = spikes_to_ascii(senders, times, ascii_values)
    decoded_text = ascii_to_text(decoded_ascii_values)

    return decoded_text

def spikes_to_ascii(senders, times, ascii_values):
    spike_dict = {idx + 1: ascii_val for idx, ascii_val in enumerate(ascii_values)}

    print(f"Spike Dictionary: {spike_dict}")

    # Create a list to store the decoded ASCII values
    decoded_ascii_values = []

    # Iterate through the range of sender IDs (1 to len(ascii_values))
    for sender in range(1, len(ascii_values) + 1):
        if sender in senders:
            decoded_ascii_values.append(spike_dict[sender])

    print(f"Decoded ASCII Values: {decoded_ascii_values}")

    return decoded_ascii_values

def ascii_to_text(ascii_values):
    text = ''.join([chr(val) for val in ascii_values])
    print(f"ASCII to Text: {text}")
    return text

text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
text_length = len(text)
events = simulate_text_encoding(text)

# Decoding functions remain the same

ascii_values = text_to_ascii(text)
decoded_text = decode_text(ascii_values)
ld = len(decoded_text)
print(f"Decoded Text: {decoded_text}, Length of string: {ld}, Length of Original String: {len(text)}")