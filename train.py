import concurrent.futures.process
import concurrent.futures
import random
from copy import deepcopy
import pickle
import os
from functools import partial
import collections
import cProfile
import pstats
import io
from agent import gen0_network, InnovationManager, Network, fast_forward_pass_flat
from physics import DoublePendulumEnv

POPULATION = 256
GENERATIONS = 50
SIM_TIME = 20
DT = 1/60.0
ELITE_PERCENTILE = 0.1 # top creatures always advance
ELITE_MUTATE = 0.8 # fill most of the population with mutations of elites, rest with mutations of commoners
CURRICULUM_STEP = 0.005 # parameter to control difficulty progression speed
NEXT_STAGE_CUTOFF = 400 # upright frames required for 90% percentile, to continue curriculum
COMPATIBILITY_THRESHOLD = 15.0

def evaluate_single_network(network_flat, run_steps, generation_seed, start_var):
    # Ensure all networks in a generation face the exact same random environmental start
    random.seed(generation_seed)
    
    env = DoublePendulumEnv(start_var=start_var)
    obs = env.reset()
    fitness = 0.0
    frames = 0

    for _ in range(run_steps):
        action = fast_forward_pass_flat(network_flat, obs)
        obs, reward, frame = env.step(action)
        fitness += reward
        if frame:
            frames += 1

    # 3. We ONLY return the score.
    # (Returning the whole brain is heavy and slows down the pipe)
    return max(0.0, fitness), frames



def run_simulation(num_generations, pop_size):
    inno_tracker = InnovationManager()
    population = [gen0_network() for _ in range(pop_size)]
    steps = int(SIM_TIME / DT)

    if not os.path.exists("saved_networks"):
        os.makedirs("saved_networks")

    current_gravity = 9.81
    current_friction = 0.05
    current_variance = 0.05
    compatibility_threshold = COMPATIBILITY_THRESHOLD

    with concurrent.futures.process.ProcessPoolExecutor(max_workers=8) as executor:
        for generation in range(num_generations + 1):
            
            gen_seed = random.randint(0, 100000000)
            eval_func = partial(evaluate_single_network, run_steps=steps,
                                generation_seed=gen_seed,
                                start_var=current_variance)
            flat_pop = [bench.export_flat() for bench in population]
            eval_results = list(executor.map(eval_func, flat_pop, chunksize=32))

            for i in range(pop_size):
                population[i].fitness, population[i].frames = eval_results[i]

            # Speciation
            species_reps = []
            species_members = collections.defaultdict(list)
            
            for network in population:
                found_species = False
                for s_idx, rep in enumerate(species_reps):
                    if network.distance_to(rep) < compatibility_threshold:
                        species_members[s_idx].append(network)
                        network.species_id = s_idx
                        found_species = True
                        break
                if not found_species:
                    s_idx = len(species_reps)
                    species_reps.append(network)
                    species_members[s_idx].append(network)
                    network.species_id = s_idx

            target_species = 15
            if len(species_reps) > target_species:
                compatibility_threshold += 0.5
            elif len(species_reps) < target_species:
                compatibility_threshold -= 0.5
            compatibility_threshold = max(0.5, compatibility_threshold)

            # Calculate adjusted fitness
            for network in population:
                species_size = len(species_members[network.species_id])
                network.adjusted_fitness = network.fitness / species_size

            # all networks have a fitness score now, copy elites exactly, mutate some elites, mutate some commoners
            population.sort(key=lambda n: n.adjusted_fitness, reverse=True)
            best_raw = max(population, key=lambda n: n.fitness)

            good_performer_raw = sorted([n.frames for n in population], reverse=True)[int(0.10 * pop_size)]

            if good_performer_raw > NEXT_STAGE_CUTOFF:
                current_variance = min(current_variance + CURRICULUM_STEP, 1.0)
                print(f"Variance updated to {current_variance:>5.5f} in Gen {generation}")

            if generation % 1 == 0:
                print(f"Generation {generation:>4} | Species: {len(species_reps)} | Best Raw Fitness: {best_raw.fitness:>5.0f}")
                print(f"Top Adj. Fit: {population[0].adjusted_fitness:>5.0f} | Median Adj. Fit: {population[int(0.5 * pop_size)].adjusted_fitness:>5.0f}")

                filename = f"saved_networks/champion_gen_{generation}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump(population, f)

            species_avg_adj_fitness = {}
            for s_idx, members in species_members.items():
                species_avg_adj_fitness[s_idx] = sum(m.adjusted_fitness for m in members) / len(members)

            total_avg = sum(species_avg_adj_fitness.values())
            
            species_slots = {}
            fractional_parts = {}
            
            for s_idx, avg in species_avg_adj_fitness.items():
                exact = (avg / total_avg) * pop_size if total_avg > 0 else pop_size / len(species_members)
                species_slots[s_idx] = int(exact)
                fractional_parts[s_idx] = exact - int(exact)
                
            remaining = pop_size - sum(species_slots.values())
            for s_idx, _ in sorted(fractional_parts.items(), key=lambda x: x[1], reverse=True)[:remaining]:
                species_slots[s_idx] += 1

            next_gen_elites = []
            next_gen_children = []
            
            for s_idx, slots in species_slots.items():
                if slots == 0:
                    continue
                members = species_members[s_idx]
                members.sort(key=lambda n: n.adjusted_fitness, reverse=True)
                
                clones_to_make = min(2, len(members), slots)
                for i in range(clones_to_make):
                    next_gen_elites.append(members[i].clone())
                    
                slots_remaining = slots - clones_to_make
                
                if slots_remaining <= 0:
                    continue

                mating_pool_size = max(1, len(members) // 2)
                mating_pool = members[:mating_pool_size]

                pool_fitness = [m.adjusted_fitness for m in mating_pool]
                min_fit = min(pool_fitness)
                
                # Shift to be > 0 for weights
                weights = [(f - min_fit + 0.001) for f in pool_fitness] 
                
                while slots_remaining > 0:
                    if random.random() < 0.75 and len(mating_pool) > 1:
                        parent1, parent2 = random.choices(mating_pool, weights=weights, k=2)
                        
                        # ensure different parents if possible
                        attempts = 0
                        while parent1 is parent2 and attempts < 10:
                             parent2 = random.choices(mating_pool, weights=weights, k=1)[0]
                             attempts += 1
                             
                        child = Network.crossover(parent1, parent2)
                    else:
                        parent = random.choices(mating_pool, weights=weights, k=1)[0]
                        child = parent.clone()
                        
                    next_gen_children.append(child)
                    slots_remaining -= 1

            next_gen = next_gen_elites + next_gen_children

            inno_tracker.start_new_generation()

            for network in next_gen_children:
                network.mutate_weights()

                if random.random() <= 0.05:
                    network.mutate_add_neuron(inno_tracker)
                if random.random() <= 0.1:
                    network.mutate_add_synapse(inno_tracker)

            population = next_gen

    return population[0] # the best network of all time!!!!

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run_simulation(GENERATIONS, POPULATION)
    pr.disable()
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(25)
    print(s.getvalue())
