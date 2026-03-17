import random
import math


class Neuron:
    def __init__(self):
        self.id = None
        self.type = None
        self.bias = None
        self.depth = None

class Synapse:
    def __init__(self):
        self.input_id = None
        self.output_id = None
        self.enabled = None
        self.weight = None
        self.innovation = None

class Network:
    def __init__(self, neurons_list, connections_list):
        self.neurons = neurons_list
        self.connections = connections_list

        self.neuron_dict = {n.id: n for n in self.neurons}
        self.execution_order = self.compute_order()

    def compute_order(self):
        # depth approach
        execution_order = [n.id for n in self.neurons if n.depth > 0.0]
        execution_order.sort(key=lambda n_id: self.neuron_dict[n_id].depth)

        # Kahn's Algorithm ########
        # in_degree = {n.id: 0 for n in self.neurons}
        # for synapse in self.connections:
        #     if synapse.enabled:
        #         in_degree[synapse.output_id] += 1
        #
        # queue = []
        # for key, value in in_degree.items():
        #     if value == 0:
        #         queue.append(key)
        #
        # sorted_neuron_ids = []
        # while len(queue) != 0:
        #     current_neuron_id = queue.pop(0)
        #     sorted_neuron_ids.append(current_neuron_id)
        #
        #     for synapse in self.connections:
        #         if synapse.enabled and synapse.input_id == current_neuron_id:
        #             in_degree[synapse.output_id] -= 1
        #             if in_degree[synapse.output_id] == 0:
        #                 queue.append(synapse.output_id)
        #
        # execution_order = []
        # for neuron_id in sorted_neuron_ids:
        #     if neuron_id >= 6: # 0 to 5 reserved for inputs
        #         execution_order.append(neuron_id)

        return execution_order

    def forward_pass(self, observations):
        neuron_values = {
            0: observations[0],  # cart x
            1: observations[1],  # cart v
            2: observations[2],  # theta
            3: observations[3],  # v
            4: observations[4],  # phi
            5: observations[5]   # w
        }

        for neuron_id in self.execution_order:
            neuron = self.neuron_dict[neuron_id]

            incoming_sum = 0.0
            for synapse in self.connections:
                if synapse.output_id == neuron.id and synapse.enabled:
                    parent_value = neuron_values.get(synapse.input_id, 0.0)
                    incoming_sum += parent_value * synapse.weight

            incoming_sum += neuron.bias
            final_value = math.tanh(incoming_sum)

            neuron_values[neuron.id] = final_value

        return neuron_values[6]

def gen0_network():
    neurons = []
    synapses = []

    # 1. Create 6 Input Neurons (IDs 0-5)
    for i in range(6):
        n = Neuron()
        n.id = i
        n.type = "INPUT"
        n.bias = 0.0  # Inputs do not use bias
        n.depth = 0.0
        neurons.append(n)

    # 2. Create 1 Output Neuron (ID 6)
    out_n = Neuron()
    out_n.id = 6
    out_n.type = "OUTPUT"
    out_n.bias = random.uniform(-1.0, 1.0) # Random bias
    out_n.depth = 1.0
    neurons.append(out_n)

    # 3. Create Synapses connecting every input to the output
    for i in range(6):
        s = Synapse()
        s.input_id = i
        s.output_id = 6
        s.weight = random.uniform(-2.0, 2.0) # Random weight
        s.enabled = True
        s.innovation = i # Just use the loop index for Gen 0 innovations
        synapses.append(s)

    # 4. Initialize and return the compiled Network
    return Network(neurons, synapses)