import concurrent.futures.process
import random
from copy import deepcopy
import pickle
import os
from agent import *
from physics import *
import concurrent.futures
from functools import partial
import copy


# This runs in a completely separate background process!
def evaluate_single_network(network, run_steps):
    env = DoublePendulumEnv()
    obs = env.reset()
    fitness = 0.0

    # 2. Run the simulation
    for _ in range(run_steps):
        action = network.forward_pass(obs)
        obs, reward = env.step(action)
        fitness += reward

    # 3. We ONLY return the score.
    # (Returning the whole brain is heavy and slows down the pipe)
    return fitness



def run_simulation(num_generations, pop_size):
    inno_tracker = InnovationManager()
    population = [gen0_network() for _ in range(pop_size)]
    steps = int(SIM_TIME / DT)

    if not os.path.exists("saved_networks"):
        os.makedirs("saved_networks")


    with concurrent.futures.process.ProcessPoolExecutor() as executor:
        for generation in range(num_generations + 1):

            eval_func = partial(evaluate_single_network, run_steps=steps)
            scores = list(executor.map(eval_func, population))

            for i in range(pop_size):
                population[i].fitness = scores[i]

            # all networks have a fitness score now, copy elites exactly, mutate some elites, mutate some commoners
            population.sort(key=lambda n: n.fitness, reverse=True)

            champion_network = population[0]

            print(f"Generation {generation:>4} Distribution :{population[-1].fitness:>5.0f} "
                  f"{population[int(0.75 * pop_size)].fitness:>5.0f} "
                  f"{population[int(0.5 * pop_size)].fitness:>5.0f} "
                  f"{population[int(0.25 * pop_size)].fitness:>5.0f} "
                  f" {population[0].fitness:>5.0f}"
                  f"{" *" if ((generation % 10 == 0) or (champion_network.fitness > 3000)) else ""}")


            if (generation % 10 == 0) or (champion_network.fitness > 2000):
                filename = f"saved_networks/champion_gen_{generation}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(champion_network, f)


            elite_ct = int(ELITE_PERCENTILE * pop_size)
            next_gen = population[:elite_ct]

            while len(next_gen) < ELITE_MUTATE * pop_size:
                parent = random.choice(population[:elite_ct])
                child = deepcopy(parent)

                next_gen.append(child)

            while len(next_gen) < pop_size:
                parent = random.choice(population)
                child = deepcopy(parent)

                next_gen.append(child)

            inno_tracker.start_new_generation()

        for network in next_gen[elite_ct:]:
            network.mutate_weights()

            if random.random() <= 0.05:
                network.mutate_add_neuron(inno_tracker)
            if random.random() <= 0.1:
                network.mutate_add_synapse(inno_tracker)

        population = next_gen

    return population[0] # the best network of all time!!!!

POPULATION = 100
GENERATIONS = 1000
SIM_TIME = 15
DT = 1/60.0
ELITE_PERCENTILE = 0.1 # top creatures always advance
ELITE_MUTATE = 0.8 # fill most of the population with mutations of elites, rest with mutations of commoners

if __name__ == "__main__":
    run_simulation(GENERATIONS, POPULATION)

