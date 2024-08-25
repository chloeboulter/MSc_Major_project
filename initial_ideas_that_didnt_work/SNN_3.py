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

def string_to_ascii(s):
    return [ord(char) for char in s]

# Parameters
text_block = "A"
binary_sequence = text_to_binary(text_block)
print(f"Binary sequence of the text '{text_block}': {binary_sequence}")
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

# Create neurons for the first layer
neuron_layer1 = nest.Create('iaf_psc_alpha', 1)

# Set an external current to the neurons to help them reach the threshold
nest.SetStatus(neuron_layer1, {'I_e': 400.0})  # Adjust this value as needed

# Create spike recorders for the spike generator and neuron layers
spike_recorder_generator = nest.Create('spike_recorder')
spike_recorder_layer1 = nest.Create('spike_recorder')

# Store results for each weight increment
results = []

# Connect spike generator to the first layer of neurons and increase weights incrementally
weights = np.arange(1000.0, 5000.0, 1000.0)

for weight in weights:
    # Connect spike generator to the neuron layer with the specified weight
    nest.Connect(spike_generator, neuron_layer1, syn_spec={'weight': weight, 'delay': 1.0})

    # Connect spike recorders to the spike generator and neuron layers
    nest.Connect(spike_generator, spike_recorder_generator)
    nest.Connect(neuron_layer1, spike_recorder_layer1)

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

    # Calculate propagation delays
    if spike_times_layer1.size > 0:
        prop_delay_sg_to_1 = spike_times_layer1[0] - spike_times_generator[0]
        prop_delay_by_L1 = prop_delay_sg_to_1

        # Convert spike times from the first layer to binary sequence
        binary_sequence_output_L1 = spike_train_to_binary(spike_times_layer1, dt, start_time, len(binary_sequence), prop_delay_by_L1)

        # Convert binary sequence back to text
        text_output1 = binary_to_text(binary_sequence_output_L1)

        # ASCII representation
        ascii_representation = string_to_ascii(text_output1)

        # Store the results
        results.append({
            'weight': weight,
            'spike_times_generator': spike_times_generator,
            'spike_times_layer1': spike_times_layer1,
            'prop_delay_sg_to_1': prop_delay_sg_to_1,
            'decoded_text': text_output1,
            'ascii_representation': ascii_representation
        })
    else:
        results.append({
            'weight': weight,
            'spike_times_generator': spike_times_generator,
            'spike_times_layer1': np.array([]),
            'prop_delay_sg_to_1': None,
            'decoded_text': None,
            'ascii_representation': None
        })

# Present the results
for result in results:
    print(f"Weight: {result['weight']}")
    print(f"Propagation delay between Spike Gen and L1: {result['prop_delay_sg_to_1']} ms")
    print(f"Recorded spike times from generator: {result['spike_times_generator']}")
    print(f"Recorded spike times from first layer: {result['spike_times_layer1']}")
    print(f"Decoded text L1: {result['decoded_text']}")
    print(f"ASCII representation: {result['ascii_representation']}")
    print("-----")

    if len(result['spike_times_layer1']) > 0:
        # Plotting the raster plot
        plt.figure(figsize=(12, 12))

        # Raster plot for spike generator
        plt.subplot(4, 1, 1)
        plt.scatter(result['spike_times_generator'], [1]*len(result['spike_times_generator']), s=10, color='blue')
        plt.title(f'Spike Generator Output (Weight: {result["weight"]})')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')

        # Raster plot for the first layer output
        plt.subplot(4, 1, 2)
        plt.scatter(result['spike_times_layer1'], [1]*len(result['spike_times_layer1']), s=10, color='red')
        plt.title(f'First Layer Output (Weight: {result["weight"]})')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')

        plt.tight_layout()
        plt.show()
