# import nest
# import matplotlib.pyplot as plt
# import numpy as np
#
# def string_to_ascii(s):
#     ascii_values = [ord(char) for char in s]
#     return ascii_values
#
# def ascii_to_string(ascii_values):
#     original_string = ''.join(chr(value) for value in ascii_values)
#     return original_string
#
# # Example usage
# input_string = "Hello, World!"
# print("Original String:", input_string)
#
# # Convert string to ASCII values
# ascii_values = string_to_ascii(input_string)
# print("ASCII Values:", ascii_values)
#
# # Convert ASCII values back to string
# converted_string = ascii_to_string(ascii_values)
# print("Converted String:", converted_string)

import numpy as np
import matplotlib.pyplot as plt
import os
import nest
import os

os.environ['TERM'] = 'xterm-256color'

# Parameters
results_dir = "results"
offset = 380
sim_time = 50.0

# Function to convert ASCII value to current
def ascii_to_current(ascii_value, offset=380):
    return ascii_value + offset

# Function to convert string to ASCII values
def string_to_ascii(string):
    return [ord(char) for char in string]

# Function to create and simulate the SNN for the given ASCII values
def simulate_raster_plot_from_string(string, current_func, sim_time=50.0):
    os.system('clear')
    print(f"Processing string: {string}")

    # Convert string to ASCII values
    ascii_values = string_to_ascii(string)
    num_neurons = len(ascii_values)

    # Initialize NEST kernel
    nest.ResetKernel()

    # Create layer with iaf_psc_alpha neurons
    layer = nest.Create('iaf_psc_alpha', num_neurons)
    # Create spike recorder
    spikerecorder = nest.Create("spike_recorder")

    # Set currents for each neuron based on ASCII values
    for i, ascii_value in enumerate(ascii_values):
        current = current_func(ascii_value)
        print(f"Setting current for neuron {i}: {current}")  # Debugging line
        print(f"Layer element type: {type(layer[i])}")  # Debugging line
        neuron_id = layer[i].global_id
        nest.SetStatus(nest.NodeCollection([neuron_id]), {"I_e": float(current)})

    nest.Connect(layer, spikerecorder)

    # Simulate
    print(f"Simulating for string: {string}")
    nest.Simulate(sim_time)
    print(f"Simulation completed for string: {string}")

    # Get spike times
    events = spikerecorder.get("events")
    senders = events["senders"]
    ts = events["times"]
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, string + "_events.npy"), events)
    np.save(os.path.join(results_dir, string + "_senders.npy"), senders)

    # Plot raster plot
    plt.figure(figsize=(10, 6))
    plt.vlines(ts, senders, senders + 1, color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title(f'Raster Plot for string: {string}')
    plt.grid()
    plt.show()

# Function to decode from spike trains to string
def decode_from_spike_trains(events, offset=380):
    senders = events['senders']
    unique_neurons = np.unique(senders)
    print("Unique neurons that spiked:", unique_neurons)  # Debugging line
    # Map back from neuron indices to ASCII values
    ascii_values = [(neuron_index - 1) - offset for neuron_index in unique_neurons]
    print("ASCII values:", ascii_values)  # Debugging line
    # Ensure the ascii values are in the valid range for chr()
    valid_ascii_values = [value for value in ascii_values if 0 <= value < 256]
    print("Valid ASCII values:", valid_ascii_values)  # Debugging line
    decoded_string = ''.join([chr(int(ascii_value)) for ascii_value in valid_ascii_values])
    return decoded_string

# Example strings to encode and simulate
strings = ["Hello", "World", "SNN"]

# Simulate raster plot for each string
for string in strings:
    simulate_raster_plot_from_string(string, ascii_to_current)

# Decode the spike trains for each string
for string in strings:
    events = np.load(os.path.join(results_dir, string + "_events.npy"), allow_pickle=True).item()
    decoded_string = decode_from_spike_trains(events, offset)
    print(f"Original string: {string}, Decoded string: {decoded_string}")
