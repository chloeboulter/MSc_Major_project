import nest
import numpy as np
import matplotlib.pyplot as plt
import os

def text_to_ascii(text):
    return [ord(c) for c in text]

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
    nest.SetStatus(layer, [{"I_e": current} for current in currents])

    nest.Connect(layer, spikerecorder)
    nest.Simulate(sim_time)

    events = spikerecorder.get("events")
    senders = events["senders"]
    ts = events["times"]

    os.makedirs("results", exist_ok=True)
    np.save("results/text_events.npy", events)
    np.save("results/text_senders.npy", senders)

    plt.figure(figsize=(10, 6))
    plt.vlines(ts, senders, senders + 1, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot for Encoded Text')
    plt.grid()
    plt.show()

    return events

text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
text_length = len(text)
events = simulate_text_encoding(text)

def spikes_to_ascii(events, ascii_values):
    senders = events["senders"]
    spike_dict = {idx + 1: ascii_val for idx, ascii_val in enumerate(ascii_values)}

    decoded_ascii_values = [spike_dict[sender] for sender in senders if sender in spike_dict]
    return decoded_ascii_values

def ascii_to_text(ascii_values):
    return ''.join([chr(val) for val in ascii_values])

def decode_text(ascii_values):
    events = np.load("results/text_events.npy", allow_pickle=True).item()
    decoded_ascii_values = spikes_to_ascii(events, ascii_values)
    decoded_text = ascii_to_text(decoded_ascii_values)

    return decoded_text

ascii_values = text_to_ascii(text)
decoded_text = decode_text(ascii_values)
ld = len(decoded_text)
print(f"Decoded Text: {decoded_text}, Length of string: {ld}, Length of Original String: {len(text)}")



