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

def text_to_binary(text):
    binary_text = ''.join(format(ord(char), '08b') for char in text)
    return binary_text

def binary_to_spike_train(binary_sequence, dt, start_time=1.0):
    spike_times = [i * dt + start_time for i, bit in enumerate(binary_sequence) if bit == '1']
    return spike_times

def spike_train_to_binary(spike_times, dt, start_time, length, total_propagation_delay):
    binary_sequence = ["0"] * length
    for spike_time in spike_times:
        adjusted_spike_time = spike_time - total_propagation_delay  # Adjust for propagation delay
        index = int((adjusted_spike_time - start_time) / dt)
        if 0 <= index < length:
            binary_sequence[index] = "1"
    return ''.join(binary_sequence)

def binary_to_text(binary_sequence):
    text = ""
    for i in range(0, len(binary_sequence), 8):
        byte = binary_sequence[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text

# Parameters
text_block = "A"
binary_sequence = text_to_binary(text_block)
dt = 1.0
start_time = 1.0

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
neuron_layer2 = nest.Create('iaf_psc_alpha', 10)
neuron_layer3 = nest.Create('iaf_psc_alpha', 1)

# Set an external current to the neurons to help them reach the threshold
nest.SetStatus(neuron_layer1, {'I_e': 400.0})  # Adjust this value as needed
nest.SetStatus(neuron_layer2, {'I_e': 400.0})  # Adjust this value as needed
nest.SetStatus(neuron_layer3, {'I_e': 400.0})  # Adjust this value as needed

# Create spike recorders for the spike generator and neuron layers
spike_recorder_generator = nest.Create('spike_recorder')
spike_recorder_layer1 = nest.Create('spike_recorder')
spike_recorder_layer2 = nest.Create('spike_recorder')
spike_recorder_layer3 = nest.Create('spike_recorder')

# Connect spike generator to the first layer of neurons with increased synaptic weight
nest.Connect(spike_generator, neuron_layer1, syn_spec={'weight': 2000.0, 'delay': 1.0})

# Connect the first layer of neurons to the second layer
nest.Connect(neuron_layer1, neuron_layer2, syn_spec={'weight': 1500.0, 'delay': 1.0})

# Connect the second layer of neurons to the third layer
nest.Connect(neuron_layer2, neuron_layer3, syn_spec={'weight': 1500.0, 'delay': 1.0})

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

# Calculate propagation delays
prop_delay_sg_to_1 = spike_times_layer1[0] - spike_times_generator[0]
prop_delay_1_to_2 = spike_times_layer2[0] - spike_times_layer1[0]
prop_delay_2_to_3 = spike_times_layer3[0] - spike_times_layer2[0]
total_propagation_delay = prop_delay_1_to_2 + prop_delay_2_to_3 + prop_delay_sg_to_1
prop_delay_by_L1 = prop_delay_sg_to_1
prop_delay_by_L2 = prop_delay_sg_to_1 + prop_delay_1_to_2

print("Propagation delay between Spike Gen and L1:", (spike_times_layer1[0] - spike_times_generator[0]), "ms.")
print("Propagation delay between L1 and L2:", prop_delay_1_to_2, "ms.")
print("Propagation delay between L2 and L3:", prop_delay_2_to_3, "ms.")
print("Total propagation delay:", total_propagation_delay, "ms.")

print("Recorded spike times from generator:", spike_times_generator)
print("Recorded spike times from first layer:", spike_times_layer1)
print("Recorded spike times from second layer:", spike_times_layer2)
print("Recorded spike times from third layer:", spike_times_layer3)

# Convert spike times from the third layer to binary sequence
binary_sequence_output = spike_train_to_binary(spike_times_layer3, dt, start_time, len(binary_sequence), total_propagation_delay)
binary_sequence_output_L2 = spike_train_to_binary(spike_times_layer2, dt, start_time, len(binary_sequence), prop_delay_by_L2)
binary_sequence_output_L1 = spike_train_to_binary(spike_times_layer1, dt, start_time, len(binary_sequence), prop_delay_by_L1)


# Convert binary sequence back to text
text_output = binary_to_text(binary_sequence_output)
text_output2 = binary_to_text(binary_sequence_output_L2)
text_output1 = binary_to_text(binary_sequence_output_L1)


print("Decoded text:", text_output)
print("Decoded text L1:", text_output1)
print("Decoded text L2:", text_output2)


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

def string_to_ascii(s):
    return [ord(char) for char in s]

ascii_representation = string_to_ascii(text_output)
print(ascii_representation)
