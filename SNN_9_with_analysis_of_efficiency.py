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

    # Measure CPU usage before simulation
    cpu_usage_before = psutil.cpu_percent(interval=None)

    # Measure memory usage before simulation
    mem_usage_before = memory_usage()[0]

    # Start timing the simulation
    start_time = time.time()

    # Simulate the network
    nest.Simulate(sim_time)

    # End timing the simulation
    end_time = time.time()

    # Measure CPU usage after simulation
    cpu_usage_after = psutil.cpu_percent(interval=None)

    # Measure memory usage after simulation
    mem_usage_after = memory_usage()[0]

    # Calculate and print profiling data
    execution_time = end_time - start_time
    cpu_usage = cpu_usage_after - cpu_usage_before
    mem_usage = mem_usage_after - mem_usage_before

    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"CPU Usage: {cpu_usage:.2f} %")
    print(f"Memory Usage: {mem_usage:.2f} MiB")

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

    return senders3, ts3_adjusted


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


# Example usage
# embed_spike_data_in_image("original_image.png", senders3, ts3_adjusted, "image_with_hidden_data.png")

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


# Example usage
# extracted_senders, extracted_times = extract_spike_data_from_image("image_with_hidden_data.png", len(senders3))

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


# Example usage after extracting spike data
# decoded_text = decode_text(ascii_values, extracted_senders, extracted_times)


# Example text to hide in an image
text = """
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

Curabitur vitae ligula augue. Nulla ut felis eget sapien efficitur fermentum. Sed nec justo 
libero. Etiam sit amet metus vitae magna accumsan sollicitudin in nec sapien. Vestibulum 
dignissim nec nulla in ultricies. Phasellus eget tristique nulla. Pellentesque sit amet arcu 
id orci egestas malesuada. Suspendisse id velit et libero consequat gravida id sit amet 
augue. Praesent lacinia lectus metus, sit amet pharetra sem lacinia quis. Integer viverra 
sem in vehicula convallis. Ut consequat sollicitudin felis non ornare. Vivamus fermentum 
massa ac nisi pretium, sed porttitor ligula ultrices. Nam vitae ante vitae eros laoreet 
volutpat eget id dolor. Curabitur vestibulum, ligula vitae luctus tempus, felis nisi lacinia 
elit, a posuere erat quam non nisi. Duis a malesuada arcu. 

Nunc ultrices sem sed dui dapibus, quis malesuada orci auctor. Vivamus suscipit elit 
nibh, et malesuada orci aliquam sed. Nulla vehicula nisi vel turpis tincidunt, sed feugiat 
tellus pharetra. Nulla vehicula ante non ex fermentum, a porttitor turpis tempor. Etiam ut 
neque eu arcu blandit dignissim et et sapien. Mauris sagittis leo id turpis dignissim, eget 
accumsan libero volutpat. Sed elementum interdum magna, in molestie felis. In id mauris 
quam. Fusce at metus sit amet ex lacinia interdum nec sit amet velit. Vestibulum varius 
semper augue, vel pharetra nisi pharetra sed. Phasellus id volutpat mi. Nulla vel nisi ac 
dolor varius dapibus. Nullam et lacus a metus eleifend scelerisque non sit amet nunc. Donec 
egestas, sem at gravida porttitor, sapien orci feugiat ante, sed dictum ligula nunc quis 
arcu. Integer tempor vel sapien sit amet egestas. Cras rutrum mi quis orci euismod dapibus.

Fusce malesuada, arcu id scelerisque accumsan, nisi nisi consectetur odio, ac condimentum 
sem velit eu nisl. Nullam aliquam dolor quis nunc vestibulum, a viverra magna scelerisque. 
Mauris aliquam libero at viverra tincidunt. Pellentesque nec nisi ligula. Etiam gravida 
eleifend posuere. Aenean finibus ligula eu justo condimentum, non varius dolor volutpat. 
Nullam malesuada tellus eu velit pretium tristique. Suspendisse vitae ligula tortor. 
Curabitur bibendum turpis nec mi fermentum dictum. Donec quis turpis consectetur, tincidunt 
libero vel, malesuada lacus. Nulla vehicula, ipsum sit amet tincidunt lacinia, felis libero 
pharetra dolor, ac elementum felis ex eget odio. Duis luctus condimentum augue sit amet 
viverra. Nulla sed sagittis dolor. Morbi ut felis vitae erat feugiat facilisis sed id magna. 
Phasellus dignissim ultricies dolor, ut pellentesque orci venenatis sed. Nam laoreet justo 
ac elit laoreet vehicula. 

Aliquam rutrum orci in eros laoreet, vel luctus neque feugiat. Fusce sit amet risus vel 
dolor pharetra cursus. Curabitur tincidunt orci quam, ac consequat tortor egestas non. 
Integer laoreet ligula non sapien varius, sed hendrerit libero posuere. Vivamus euismod 
malesuada justo, vel scelerisque risus blandit in. Integer et vehicula elit. Suspendisse 
nec mauris a lectus fermentum fringilla. Phasellus dictum suscipit turpis non ultrices. 
Aliquam erat volutpat. Cras egestas lectus risus, eu lobortis ligula vulputate nec. 
Mauris ac vehicula lorem. Nullam elementum erat in feugiat dapibus. In hac habitasse platea 
dictumst. Sed viverra sapien sed nunc varius, sed interdum purus tempor. Nullam non sem 
vel mauris fringilla dictum ac sit amet tortor. Etiam ut justo ullamcorper, interdum arcu 
a, luctus sem. 

Duis dapibus sollicitudin metus, ac vehicula nisi laoreet nec. Praesent fermentum tortor 
elit, eget molestie erat aliquam sed. Integer eu lacinia magna. Integer pharetra, orci 
eget suscipit fermentum, lorem dolor hendrerit nisl, eget suscipit justo sem vel enim. 
Ut porttitor ligula vitae pharetra viverra. Duis sodales ligula ac ex posuere, eu luctus 
leo volutpat. Pellentesque vehicula, libero a vehicula suscipit, erat purus vulputate urna, 
eu vestibulum odio sapien id elit. Integer eget lectus sed orci malesuada gravida non nec 
est. Aenean porta nisl sed lacus viverra tincidunt. Sed elementum, odio et malesuada sodales, 
mauris felis fringilla enim, a vehicula risus nulla sed purus. Proin lacinia libero ac metus 
tempus, in molestie ante luctus. Nam lobortis augue sit amet libero porttitor, eu faucibus 
nulla vehicula. Aenean viverra vestibulum justo ut hendrerit. Integer vulputate ex nunc, 
at venenatis sapien fermentum in. Aenean iaculis tellus et nibh aliquam, vel pharetra purus 
mollis. Nam vel leo nec nunc interdum interdum a sit amet urna.

Suspendisse id eros sit amet justo sodales tristique. Cras faucibus, velit id dictum 
condimentum, tortor urna venenatis dui, nec posuere neque eros eu dolor. Ut consectetur 
nunc non erat ultricies gravida. Vivamus posuere lacus at leo ullamcorper, vel feugiat 
purus sodales. Etiam ornare, orci vel scelerisque lacinia, orci lectus ultrices mi, 
eget elementum purus arcu in dolor. Duis varius magna et efficitur sollicitudin. Sed 
iaculis viverra nisi a iaculis. Cras volutpat, nunc sit amet venenatis iaculis, purus 
ante dictum enim, sit amet malesuada arcu enim sed odio. Mauris vestibulum nisi ac diam 
euismod, at pellentesque dolor ornare. Morbi feugiat luctus lorem, nec dapibus nisl 
sodales sed. Aenean pharetra, felis in sodales facilisis, leo eros molestie ipsum, 
vulputate varius erat odio at libero. Nulla facilisi. Integer at gravida risus. Nulla 
sit amet volutpat purus. Nullam nec dignissim lectus, eget facilisis dui. Nam in dolor 
ante.

Vivamus vulputate volutpat erat ac fringilla. Cras id scelerisque justo. Cras id 
vulputate ante. Curabitur gravida dui orci, ac dapibus nunc dapibus et. Nulla non diam 
nec dolor ultricies pretium a et augue. Nulla vel dapibus est. Ut sodales, leo sed 
mattis volutpat, tortor orci porta nulla, at volutpat orci turpis ac orci. Morbi 
tristique, nulla sit amet suscipit lobortis, nisi turpis vulputate justo, vel tristique 
libero leo et urna. Sed tincidunt quam ut ex volutpat, ut congue lacus venenatis. Cras 
sed est gravida, pharetra justo vel, vestibulum magna. Donec sed nunc risus. Donec maximus 
auctor ante, nec facilisis ante elementum non. Fusce ornare turpis justo, eget tincidunt 
ipsum viverra quis. Vivamus a augue justo. Aenean ultrices bibendum justo, sed dictum 
ipsum tincidunt at. Nullam a risus libero.

Ut in vestibulum lorem. Donec consectetur quam non interdum ultricies. Phasellus sit 
amet purus ultricies, vehicula libero in, malesuada libero. Donec sagittis neque sit 
amet nisi laoreet suscipit. Integer bibendum consequat nisi, et laoreet nisl tempor at. 
Ut fringilla dui at odio faucibus, sit amet dapibus ligula pharetra. Integer viverra 
ante arcu, at fermentum risus faucibus eu. Donec suscipit posuere libero, id pharetra 
libero volutpat quis. Integer pellentesque scelerisque tortor, quis eleifend lacus. 
Phasellus ac tortor at velit fermentum ultrices. Quisque sit amet ligula ipsum. Nulla 
laoreet justo vel dui venenatis, eget bibendum nisi varius. Morbi tristique tincidunt 
nunc, id consequat ex pulvinar nec. Nullam sollicitudin tellus id risus malesuada, ac 
vulputate tortor tristique. Nam pulvinar ligula nec nisi iaculis faucibus. Integer eu 
diam risus.

Pellentesque scelerisque feugiat lectus sit amet vehicula. Sed id feugiat metus. Donec 
ut augue tempor, mollis neque sed, consequat felis. Nulla efficitur gravida sapien nec 
porta. Phasellus interdum turpis id nisi interdum, quis dapibus lacus fermentum. Morbi 
ornare ultricies nibh, in venenatis dui dignissim vel. Donec malesuada leo in libero 
elementum convallis. Nulla facilisi. Vestibulum congue nulla vitae ligula accumsan, 
at auctor nunc venenatis. Nulla eget erat metus. Suspendisse laoreet ornare eros, ac 
volutpat sapien euismod ac. Donec molestie risus ac ex pellentesque, sit amet volutpat 
arcu tristique. Duis volutpat, arcu nec posuere hendrerit, sapien nulla dictum justo, 
in feugiat libero risus non libero. Nullam venenatis et nulla sed convallis. Cras sed 
justo nec tortor efficitur feugiat. Ut ac neque fermentum, cursus orci ut, suscipit 
dui. Suspendisse potenti.

Aliquam interdum ligula in auctor varius. Pellentesque non arcu non quam varius luctus. 
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis 
egestas. Sed posuere enim quis consectetur facilisis. Proin dictum bibendum tortor, 
ac varius elit vehicula at. Vestibulum non libero nec leo dapibus facilisis. Curabitur 
viverra eros eu ligula scelerisque congue. Vestibulum eget diam suscipit, cursus odio 
at, sodales risus. Etiam fringilla pharetra ligula, ac aliquam ex tristique in. Nulla 
at arcu et nisi dignissim vulputate. Proin dictum urna metus, id auctor neque mollis 
in. Donec nec fringilla nulla, vel aliquet urna. Nam luctus, ipsum nec convallis 
porttitor, nisi justo ultrices ligula, eu vehicula leo sem at enim. Vivamus sodales 
risus turpis, a tincidunt metus mollis a. In sagittis viverra egestas.

Etiam pretium pharetra mauris, at placerat lorem pharetra eu. Nullam mollis urna vel 
ligula malesuada condimentum. Cras dapibus nulla eget leo scelerisque, ac feugiat sapien 
feugiat. Ut ac lacus velit. Vivamus tempus arcu magna, sed lobortis erat iaculis sed. 
Donec lacinia velit et ante ultricies bibendum. Quisque maximus consectetur velit, vel 
sollicitudin mi euismod eget. Pellentesque sit amet mi volutpat, fringilla mi nec, 
sodales orci. Aenean dapibus est sapien, in tincidunt orci blandit sed. Nulla dictum 
velit nulla, in faucibus odio rutrum a. Pellentesque habitant morbi tristique senectus 
et netus et malesuada fames ac turpis egestas. Quisque id justo viverra, iaculis nulla 
eget, dapibus orci. Aenean interdum justo at leo molestie malesuada. Ut cursus, quam 
et iaculis eleifend, sapien nisl rhoncus dolor, non sagittis magna nisl ac sapien. 
Praesent et tincidunt lectus. Nullam consectetur justo at tellus eleifend aliquet.

Proin aliquet, urna nec laoreet dapibus, mi sapien cursus libero, non pulvinar nulla 
felis a purus. Cras convallis ex sed cursus iaculis. Sed vestibulum turpis ut risus 
condimentum, in pretium quam scelerisque. Suspendisse potenti. Maecenas ultricies 
velit quis ligula tempus, non fermentum sapien faucibus. Nulla facilisi. Praesent nec 
ultricies magna, sit amet lacinia purus. Aenean aliquam, nisi ut feugiat vulputate, 
turpis orci fermentum nulla, in laoreet sem turpis sit amet neque. Integer porttitor 
sapien nec quam aliquet, id volutpat risus venenatis. Donec euismod fringilla orci 
non commodo. In hac habitasse platea dictumst. Sed pharetra purus at magna malesuada, 
non suscipit ante eleifend. Etiam non tempor sapien. Proin sit amet auctor odio, a 
varius erat. Suspendisse pharetra feugiat magna non malesuada. Aliquam erat volutpat.
"""
ascii_values = text_to_ascii(text)

# Simulate the encoding to get the spike data
senders3, ts3_adjusted = simulate_text_encoding(text)

# Embed the spike data into an image
embed_spike_data_in_image("dalle_image.png", senders3, ts3_adjusted, "image_with_hidden_data_efficiency_test.png")

# Extract the spike data back from the image
extracted_senders, extracted_times = extract_spike_data_from_image("image_with_hidden_data_efficiency_test.png", len(senders3))

# Decode the text from the extracted spike data
decoded_text = decode_text(ascii_values, extracted_senders, extracted_times)
print(f"Decoded Text: {decoded_text}, Length of string: {len(decoded_text)}, Length of Original String: {len(text)}")
