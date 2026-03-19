import random
import math

class Neuron:
    def __init__(self, id, type, bias, depth):
        self.id = id
        self.type = type
        self.bias = bias
        self.depth = depth

class Synapse:
    def __init__(self, input_id, output_id, weight, innovation, enabled=True):
        self.input_id = input_id
        self.output_id = output_id
        self.enabled = enabled
        self.weight = weight
        self.innovation = innovation


    def __eq__(self, other):
        return True if (self.input_id == other.input_id and self.output_id == other.output_id) else False

class InnovationManager:
    def __init__(self):
        self.innovation_ct = 10 # gen0 syanpses already have their innovation numbers
        self.neuron_ct = 11 # gen0 inputs + outputs

        # this generation only; manage new innovations
        self.innovationDict = {}
        self.nodeDict = {}

    def start_new_generation(self):
        self.innovationDict.clear()
        self.nodeDict.clear()

    def get_synapse_innovation(self, input_id, output_id):
        key = (input_id, output_id)

        # already created
        if key in self.innovationDict:
            return self.innovationDict[key]

        #new discovery
        new_id = self.innovation_ct
        self.innovationDict[key] = new_id
        self.innovation_ct += 1

        return new_id

    def get_new_node_id(self, input_id, output_id):
        key = (input_id, output_id)
        if key in self.nodeDict:
            return self.nodeDict[key]
            
        new_id = self.neuron_ct
        self.nodeDict[key] = new_id
        self.neuron_ct += 1
        return new_id


class Network:
    def __init__(self, neurons_list, connections_list):
        self.neurons = neurons_list
        self.connections = connections_list
        self.fitness = None
        self.adjusted_fitness = None
        self.species_id = None
        self.frames = None

        self.neuron_dict = {n.id: n for n in self.neurons}
        self.execution_order = self.compute_order()
        self.incoming_synapses = self.compute_incoming_synapses()
        self.flattened_execution = self.compute_flattened_execution()

    def clone(self):
        new_neurons = [Neuron(n.id, n.type, n.bias, n.depth) for n in self.neurons]
        new_connections = [Synapse(s.input_id, s.output_id, s.weight, s.innovation, s.enabled) for s in self.connections]
        
        clone_net = Network.__new__(Network)
        clone_net.neurons = new_neurons
        clone_net.connections = new_connections
        clone_net.fitness = self.fitness
        clone_net.adjusted_fitness = self.adjusted_fitness
        clone_net.species_id = self.species_id
        
        clone_net.neuron_dict = {n.id: n for n in clone_net.neurons}
        
        # Pass by reference to save computation time
        clone_net.execution_order = self.execution_order
        
        # Must recompute so flattened representation uses the newly referenced neurons and synapses
        clone_net.incoming_synapses = clone_net.compute_incoming_synapses()
        clone_net.flattened_execution = clone_net.compute_flattened_execution()
        
        return clone_net
    def compute_flattened_execution(self):
        instructions = []
        for n_id in self.execution_order:
            neuron = self.neuron_dict[n_id]
            syns = self.incoming_synapses.get(n_id, [])
            in_syn_weights = [(s.input_id, s.weight) for s in syns]
            instructions.append((n_id, neuron.bias, in_syn_weights))
        return instructions

    def distance_to(self, other_network, c1=1.0, c3=0.4):
        my_synapses = {s.innovation: s for s in self.connections}
        other_synapses = {s.innovation: s for s in other_network.connections}

        all_innovations = set(my_synapses.keys()) | set(other_synapses.keys())
        
        N = max(len(my_synapses), len(other_synapses))
        if N < 20: # unmutated entirely
            N = 1.0 # Standard NEAT logic, don't normalize small networks

        disjoint_excess = 0
        weight_diff_sum = 0.0
        matching = 0

        for inno in all_innovations:
            if inno in my_synapses and inno in other_synapses:
                matching += 1
                weight_diff_sum += abs(my_synapses[inno].weight - other_synapses[inno].weight)
            else:
                disjoint_excess += 1

        avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0.0

        return (c1 * disjoint_excess / N) + (c3 * avg_weight_diff)

        
    @classmethod
    def crossover(cls, parent1, parent2):
        # 1. Ensure parent1 is always the more fit parent
        if parent2.fitness is not None and parent1.fitness is not None and parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1

        # FIX 1: Properly cross over node biases for shared topology
        p2_neuron_dict = {n.id: n for n in parent2.neurons}
        new_neurons = []
        for n in parent1.neurons:
            bias = n.bias
            # If both parents share this node, randomly inherit the bias
            if n.id in p2_neuron_dict and random.random() < 0.5:
                bias = p2_neuron_dict[n.id].bias
            new_neurons.append(Neuron(n.id, n.type, bias, n.depth))

        p1_synapses = {s.innovation: s for s in parent1.connections}
        p2_synapses = {s.innovation: s for s in parent2.connections}

        all_innovations = set(p1_synapses.keys()) | set(p2_synapses.keys())

        new_synapses = []
        for inno in sorted(list(all_innovations)):
            # Matching genes
            if inno in p1_synapses and inno in p2_synapses:
                chosen = random.choice([p1_synapses[inno], p2_synapses[inno]])

                # FIX 2: Prevent severed networks by inheriting structure state from the fit parent
                enabled = p1_synapses[inno].enabled

                # Small chance to re-enable a previously disabled structural path
                if not enabled and random.random() < 0.25:
                    enabled = True

                new_synapses.append(
                    Synapse(chosen.input_id, chosen.output_id, chosen.weight, chosen.innovation, enabled))

            # Disjoint/Excess genes from the fitter parent only
            elif inno in p1_synapses:
                s = p1_synapses[inno]
                new_synapses.append(Synapse(s.input_id, s.output_id, s.weight, s.innovation, s.enabled))

        return cls(new_neurons, new_synapses)

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

    def compute_incoming_synapses(self):
        incoming = {n.id: [] for n in self.neurons}
        for synapse in self.connections:
            if synapse.enabled:
                incoming[synapse.output_id].append(synapse)
        return incoming

    def export_flat(self):
        return tuple(self.flattened_execution)


    def forward_pass(self, observations):
        neuron_values = {
            0: observations[0],  # cart x
            1: observations[1],  # cart v
            2: observations[2],  # sin_theta
            3: observations[3],  # cos_theta
            4: observations[4],  # v_x
            5: observations[5],  # v_y
            6: observations[6],  # sin_phi
            7: observations[7],  # cos_phi
            8: observations[8],  # w_x
            9: observations[9]   # w_y
        }

        for n_id, bias, syns in self.flattened_execution:
            incoming_sum = bias
            for in_id, weight in syns:
                incoming_sum += neuron_values.get(in_id, 0.0) * weight
            neuron_values[n_id] = math.tanh(incoming_sum)

        for n in self.neurons:
            n.last_activation = neuron_values.get(n.id, 0.0)

        return neuron_values[10]

    def mutate_add_neuron(self, innovation_tracker):
        # split a random  existing synapse with a nueron in between
        # only split active synapses
        # important: new neuron has bias of 0, first synapse has weight 1.0, second synapse original weight

        # always find a random enabled synapse without bias or redundant checks
        enabled_synapses = [s for s in self.connections if s.enabled]
        if not enabled_synapses:
            return  # Safety check just in case the brain has no active wires
        random_synapse = random.choice(enabled_synapses)

        # find the random synapse in network
        random_synapse.enabled = False

        in_depth = self.neuron_dict[random_synapse.input_id].depth
        out_depth = self.neuron_dict[random_synapse.output_id].depth

        # new neuron
        new_depth = (in_depth + out_depth) / 2.0
        new_id = innovation_tracker.get_new_node_id(random_synapse.input_id, random_synapse.output_id)
        new_neuron = Neuron(new_id, "HIDDEN", 0.0, new_depth)
        self.neurons.append(new_neuron)
        self.neuron_dict[new_id] = new_neuron

        # check for existing innovation on the synapses
        synapse1_inno = innovation_tracker.get_synapse_innovation(random_synapse.input_id, new_id)
        synapse2_inno = innovation_tracker.get_synapse_innovation(new_id, random_synapse.output_id)

        new_synapse1 = Synapse(random_synapse.input_id, new_id, 1.0, synapse1_inno, True)
        new_synapse2 = Synapse(new_id, random_synapse.output_id, random_synapse.weight, synapse2_inno, True)
        self.connections.append(new_synapse1)
        self.connections.append(new_synapse2)

        # update execution order & incoming synapses cache
        self.execution_order = self.compute_order()
        self.incoming_synapses = self.compute_incoming_synapses()
        self.flattened_execution = self.compute_flattened_execution()

    def mutate_add_synapse(self, innovation_tracker):
        existing_synapses = set((s.input_id, s.output_id) for s in self.connections)

        possible_new = []
        for sender in self.neurons:
            if sender.type == "OUTPUT":
                continue

            for reciever in self.neurons:
                if reciever.type == "INPUT":
                    continue

                if sender.depth >= reciever.depth:
                    continue

                if (sender.id, reciever.id) not in existing_synapses:
                    possible_new.append((sender, reciever))

        if not possible_new:
            return #could not find a valid connection to build

        chosen_sender, chosen_receiver = random.choice(possible_new)
        synapse_inno = innovation_tracker.get_synapse_innovation(chosen_sender.id, chosen_receiver.id)

        new_synapse = Synapse(chosen_sender.id,
                              chosen_receiver.id,
                              random.uniform(-2.0, 2.0),
                              synapse_inno,
                              True)

        self.connections.append(new_synapse)
        self.incoming_synapses = self.compute_incoming_synapses()
        self.flattened_execution = self.compute_flattened_execution()

    def mutate_weights(self):
        for synapse in self.connections:
            # 90% chance for a gene to be mutated
            if random.random() < 0.8:
                if random.random() < 0.9:
                    # Perturb the weight with a larger power (e.g. 0.5 is standard)
                    synapse.weight += random.uniform(-0.1, 0.1)
                else:
                    # Totally entirely new weight 10% of time
                    synapse.weight = random.uniform(-2.0, 2.0)

        for neuron in self.neurons:
            if neuron.type != "INPUT":
                if random.random() < 0.9:
                    if random.random() < 0.9:
                        neuron.bias += random.uniform(-0.5, 0.5)
                    else:
                        neuron.bias = random.uniform(-2.0, 2.0)

        self.flattened_execution = self.compute_flattened_execution()


def gen0_network():
    neurons = []
    synapses = []

    # 1. Create 10 Input Neurons (IDs 0-9)
    for i in range(10):
        n = Neuron(i, "INPUT", 0.0, 0.0)
        neurons.append(n)

    # 2. Create 1 Output Neuron (ID 10)
    out_n = Neuron(10, "OUTPUT", random.uniform(-1.0, 1.0), 1.0)
    neurons.append(out_n)

    # 3. Create Synapses connecting every input to the output
    for i in range(10):
        s = Synapse(i, 10, random.uniform(-2.0, 2.0), i, True)
        synapses.append(s)

    # 4. Initialize and return the compiled Network
    return Network(neurons, synapses)

def fast_forward_pass_flat(flat_execution, obs):
    neuron_values = {
        0: obs[0],  # cart x
        1: obs[1],  # cart v
        2: obs[2],  # sin_theta
        3: obs[3],  # cos_theta
        4: obs[4],  # v_x
        5: obs[5],  # v_y
        6: obs[6],  # sin_phi
        7: obs[7],  # cos_phi
        8: obs[8],  # w_x
        9: obs[9]   # w_y
    }

    for n_id, bias, syns in flat_execution:
        incoming_sum = bias
        for in_id, weight in syns:
            incoming_sum += neuron_values[in_id] * weight
        neuron_values[n_id] = math.tanh(incoming_sum)

    return neuron_values[10]