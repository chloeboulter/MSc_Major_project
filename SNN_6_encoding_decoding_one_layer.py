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

    # Create layer with iaf_psc_alpha neurons
    layer = nest.Create('iaf_psc_alpha', num_neurons)
    spikerecorder = nest.Create("spike_recorder")

    # Set currents based on ASCII values
    currents = [ascii_to_current(ascii_val) for ascii_val in ascii_values]
    print(f"Currents: {currents}")
    nest.SetStatus(layer, [{"I_e": current} for current in currents])

    nest.Connect(layer, spikerecorder)
    nest.Simulate(sim_time)

    events = spikerecorder.get("events")
    senders = events["senders"]
    ts = events["times"]

    print(f"Spike Senders: {senders}")
    print(f"Spike Times: {ts}")

    os.makedirs("results", exist_ok=True)
    np.save("results/text_events_senders.npy", senders)
    np.save("results/text_events_times.npy", ts)

    plt.figure(figsize=(10, 6))
    plt.vlines(ts, senders, senders + 1, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot for Encoded Text')
    plt.grid()
    plt.show()

    return events

text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
text_length = len(text)
events = simulate_text_encoding(text)


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

def decode_text(ascii_values):
    senders = np.load("results/text_events_senders.npy")
    times = np.load("results/text_events_times.npy")
    decoded_ascii_values = spikes_to_ascii(senders, times, ascii_values)
    decoded_text = ascii_to_text(decoded_ascii_values)

    return decoded_text

ascii_values = text_to_ascii(text)
decoded_text = decode_text(ascii_values)
ld = len(decoded_text)
print(f"Decoded Text: {decoded_text}, Length of string: {ld}, Length of Original String: {len(text)}")
