import nest
import numpy as np
import matplotlib.pyplot as plt

def generate_text_block(size_kb):
    size_bytes = size_kb * 1024  # Convert KB to Bytes
    block = ""
    while len(block.encode('utf-8')) < size_bytes:
        block += "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " \
                 "Phasellus imperdiet, nulla et dictum interdum, nisi lorem egestas odio, " \
                 "vitae scelerisque enim ligula venenatis dolor. Maecenas nisl est, ultrices nec congue eget, " \
                 "aliquet ac urna. Suspendisse non nisl sit amet velit hendrerit rutrum. Ut leo. " \
                 "Ut a mi at velit hendrerit rutrum. Suspendisse non nisl sit amet velit hendrerit rutrum. " \
                 "Ut leo. Ut a mi at velit hendrerit rutrum.\n"
    # Trim to exactly 25KB
    block = block.encode('utf-8')[:size_bytes].decode('utf-8', 'ignore')
    return block

# Example text block
text_block = "Hello World!"

def text_to_binary(text):
    # Convert each character in the text to its binary representation
    binary_text = ''.join(format(ord(char), '08b') for char in text)
    return binary_text

def binary_to_spike_train(binary_sequence, dt, start_time=1.0):
    """
    Convert a binary sequence to spike train times.

    Parameters:
    binary_sequence (str): A string of binary digits (e.g., "1100101").
    dt (float): The time interval between potential spikes.
    start_time (float): The start time for the first spike to avoid zero time.

    Returns:
    list: A list of spike times.
    """
    spike_times = [i * dt + start_time for i, bit in enumerate(binary_sequence) if bit == '1']
    return spike_times

def spike_train_to_binary(spike_times, dt, start_time):
    binary_sequence = ""
    for t in np.arange(start_time, max(spike_times) + dt, dt):
        if t in spike_times:
            binary_sequence += '1'
        else:
            binary_sequence += '0'
    return binary_sequence

def binary_to_text(binary_sequence):
    text = ""
    for i in range(0, len(binary_sequence), 8):
        byte = binary_sequence[i:i+8]
        text += chr(int(byte, 2))
    return text

# Parameters
binary_sequence = text_to_binary(text_block)  # Example binary sequence from text block
dt = 1.0  # Time interval between potential spikes in ms
start_time = 1.0  # Start time for the first spike to avoid zero time

# Create NEST kernel
nest.ResetKernel()

# Create a spike generator
spike_generator = nest.Create('spike_generator')

# Convert binary sequence to spike times
spike_times = binary_to_spike_train(binary_sequence, dt, start_time)
print("Spike times generated:", spike_times)

# Set spike times to the spike generator
nest.SetStatus(spike_generator, {'spike_times': spike_times})

# Create neurons for the first, second, and third layers
neuron_layer1 = nest.Create('iaf_psc_alpha', 10)
neuron_layer2 = nest.Create('iaf_psc_alpha', 50)
neuron_layer3 = nest.Create('iaf_psc_alpha', 1)

# Set an external current to the neurons to help them reach the threshold
nest.SetStatus(neuron_layer1, {'I_e': 200.0})  # Adjust this value as needed
nest.SetStatus(neuron_layer2, {'I_e': 200.0})  # Adjust this value as needed
nest.SetStatus(neuron_layer3, {'I_e': 200.0})  # Adjust this value as needed

# Create spike recorders for the spike generator and neuron layers
spike_recorder_generator = nest.Create('spike_recorder')
spike_recorder_layer1 = nest.Create('spike_recorder')
spike_recorder_layer2 = nest.Create('spike_recorder')
spike_recorder_layer3 = nest.Create('spike_recorder')

# Connect spike generator to the first layer of neurons with increased synaptic weight
nest.Connect(spike_generator, neuron_layer1, syn_spec={'weight': 1500.0, 'delay': 1.0})

# Connect the first layer of neurons to the second layer
nest.Connect(neuron_layer1, neuron_layer2, syn_spec={'weight': 1000.0, 'delay': 1.0})

# Connect the second layer of neurons to the third layer
nest.Connect(neuron_layer2, neuron_layer3, syn_spec={'weight': 1000.0, 'delay': 1.0})

# Connect spike recorders to the spike generator and neuron layers
nest.Connect(spike_generator, spike_recorder_generator)
nest.Connect(neuron_layer1, spike_recorder_layer1)
nest.Connect(neuron_layer2, spike_recorder_layer2)
nest.Connect(neuron_layer3, spike_recorder_layer3)

# Simulate for a duration just beyond the last spike time
simulation_time = (len(binary_sequence) + 1) * dt + start_time
nest.Simulate(simulation_time)

# Get the recorded spike events from the spike generator and neuron layers
spike_events_generator = nest.GetStatus(spike_recorder_generator, 'events')[0]
spike_times_generator = spike_events_generator['times']
neuron_ids_generator = spike_events_generator['senders']

spike_events_layer1 = nest.GetStatus(spike_recorder_layer1, 'events')[0]
spike_times_layer1 = spike_events_layer1['times']
neuron_ids_layer1 = spike_events_layer1['senders']

spike_events_layer2 = nest.GetStatus(spike_recorder_layer2, 'events')[0]
spike_times_layer2 = spike_events_layer2['times']
neuron_ids_layer2 = spike_events_layer2['senders']

spike_events_layer3 = nest.GetStatus(spike_recorder_layer3, 'events')[0]
spike_times_layer3 = spike_events_layer3['times']
neuron_ids_layer3 = spike_events_layer3['senders']

print("Recorded spike times from generator:", spike_times_generator)
print("Recorded spike times from first layer:", spike_times_layer1)
print("Recorded spike times from second layer:", spike_times_layer2)
print("Recorded spike times from third layer:", spike_times_layer3)

# Convert spike times from the third layer to binary sequence
binary_sequence_output = spike_train_to_binary(spike_times_layer3, dt, start_time)

# Convert binary sequence back to text
text_output = binary_to_text(binary_sequence_output)

print("Decoded text:", text_output)

# Plotting the raster plot
plt.figure(figsize=(12, 12))

# Raster plot for spike generator
plt.subplot(4, 1, 1)
plt.scatter(spike_times_generator, neuron_ids_generator, s=10, color='blue')
plt.title('Spike Generator Output')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')

# Raster plot for the first layer output
plt.subplot(4, 1, 2)
plt.scatter(spike_times_layer1, neuron_ids_layer1, s=10, color='red')
plt.title('First Layer Output')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')

# Raster plot for the second layer output
plt.subplot(4, 1, 3)
plt.scatter(spike_times_layer2, neuron_ids_layer2, s=10, color='green')
plt.title('Second Layer Output')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')

# Raster plot for the third layer output
plt.subplot(4, 1, 4)
plt.scatter(spike_times_layer3, neuron_ids_layer3, s=10, color='purple')
plt.title('Third Layer Output')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')

plt.tight_layout()
plt.show()
