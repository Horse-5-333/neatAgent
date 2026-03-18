import concurrent.futures.process
import concurrent.futures
import random
from copy import deepcopy
import pickle
import os
from functools import partial
from agent import gen0_network, InnovationManager, Network, fast_forward_pass_flat
from physics import DoublePendulumEnv

POPULATION = 96
GENERATIONS = 1000
SIM_TIME = 20
DT = 1/60.0
ELITE_PERCENTILE = 0.1 # top creatures always advance
ELITE_MUTATE = 0.8 # fill most of the population with mutations of elites, rest with mutations of commoners
CURRICULUM_STEP = 0.0025 # parameter to control difficulty progression speed
NEXT_STAGE_CUTOFF = 1200 # reward required for 95% percentile, to continue cirriculum

def evaluate_single_network(network_flat, run_steps, generation_seed, start_var):
    # Ensure all networks in a generation face the exact same random environmental start
    random.seed(generation_seed)
    
    env = DoublePendulumEnv(start_var=start_var)
    obs = env.reset()
    fitness = 0.0

    # 2. Run the simulation
    for _ in range(run_steps):
        action = fast_forward_pass_flat(network_flat, obs)
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

    current_gravity = 9.81
    current_friction = 0.05
    current_variance = 0.01

    with concurrent.futures.process.ProcessPoolExecutor(max_workers=8) as executor:
        for generation in range(num_generations + 1):
            
            gen_seed = random.randint(0, 100000000)
            eval_func = partial(evaluate_single_network, run_steps=steps,
                                generation_seed=gen_seed,
                                start_var=current_variance)
            flat_pop = [bench.export_flat() for bench in population]
            scores = list(executor.map(eval_func, flat_pop, chunksize=12))

            for i in range(pop_size):
                population[i].fitness = scores[i]

            # all networks have a fitness score now, copy elites exactly, mutate some elites, mutate some commoners
            population.sort(key=lambda n: n.fitness, reverse=True)

            good_performer = population[int(0.90 * pop_size)]

            if good_performer.fitness > NEXT_STAGE_CUTOFF:
                current_variance = min(current_variance * 1.01, 1)
                print(f"Variance updated to {current_variance:>5.5f} in Gen {generation}")


            if generation % 100 == 0:
                print(f"Generation {generation:>4} Distribution :{population[-1].fitness:>5.0f} "
                      f"{population[int(0.75 * pop_size)].fitness:>5.0f} "
                      f"{population[int(0.5 * pop_size)].fitness:>5.0f} "
                      f"{population[int(0.25 * pop_size)].fitness:>5.0f} "
                      f" {population[0].fitness:>5.0f}")

                filename = f"saved_networks/champion_gen_{generation}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(population, f)


            elite_ct = int(ELITE_PERCENTILE * pop_size)
            next_gen = [deepcopy(p) for p in population[:elite_ct]]

            while len(next_gen) < ELITE_MUTATE * pop_size:
                if random.random() < 0.75:
                    parent1 = random.choice(population[:elite_ct])
                    parent2 = random.choice(population[:elite_ct])
                    child = Network.crossover(parent1, parent2)
                else:
                    parent = random.choice(population[:elite_ct])
                    child = deepcopy(parent)
                next_gen.append(child)

            while len(next_gen) < pop_size:
                if random.random() < 0.75:
                    parent1 = random.choice(population[:elite_ct])
                    parent2 = random.choice(population)
                    child = Network.crossover(parent1, parent2)
                else:
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

if __name__ == "__main__":
    run_simulation(GENERATIONS, POPULATION)

