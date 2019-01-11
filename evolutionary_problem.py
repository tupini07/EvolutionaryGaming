import itertools
import math
import multiprocessing
import os
import random
import sys
from typing import Dict, List

import gym
import numpy as np
from inspyred import ec
from inspyred.benchmarks import Benchmark

import constants as cc
from CGP_program import CGP_program

bounder = ec.Bounder(0.0, 1.0)


# Util functions for GC

def generator(random: random.Random, args: Dict) -> List:
    """
    Generate an individual of length `cc.N_EVOLVABLE_GENES` where every element in it is a random number x 
    where 0 <= x <= 1 

    But only the outputs and C (number of inner nodes) will be evolved, so the inputs are not considered part of the genome 

    Parameters
    ----------
    random : random.Random
        The random generator passed to inspyred
    args : Dict
        Dictionary of arguments passed to inspyred

    Returns
    -------
    List
        An individual of length `cc.N_EVOLVABLE_GENES` where every element in it is a random number x 
        where 0 <= x <= 1 
    """

    return [random.uniform(0.0, 1.0) for _ in range(cc.N_EVOLVABLE_GENES)]


def observer(population, num_generations, num_evaluations, args):
    """
    Print information of the population at the end of every generation.
    """

    best = max(population)

    # Print information to console
    print(f"GEN: {num_generations} \t Best fitness: {best.fitness}")
    print("Fitnesses of complete population:")

    for p in sorted(population, key=lambda x: x.fitness, reverse=True):
        print("\t" + str(p.fitness))

    print()

    # Write results to file

    # create results file if it doesn't exist already
    if not os.path.isfile("./results.csv"):
        with open("./results.csv", "w+") as ff:
            ff.write("Game Name,Generation,Fitness,Individual\n")

    with open("./results.csv", "a+") as ff:
        # append information on the current individuals in the population to the results file
        for p in population:
            ff.write(
                f"{cc.ATARI_GAME},{num_generations},{p.fitness},{' '.join(str(x) for x in p.candidate)}\n")


@ec.evaluators.evaluator
def evaluator(candidate: List, args: Dict) -> float:

    # the [0] *3*4 represent the genome for the input cells
    candidate = ([0]*3*4) + candidate
    cpg_genome = [candidate[i:i+4]
                  for i in range(0, len(candidate), 4)]  # split into chunks of 4

    program = CGP_program(cpg_genome)

    program.draw_function_graph("currently_evaluating")  # TODO: remove when doing final evaluation
    with open("currently_evaluating.txt", "w+") as ff:
        ff.write(str(program))

    # if current program doesn't make use of input nodes then just return -inf as firtness
    if not any(x.active for x in program.input_cells):
        return np.NINF

    # else just proceed to make the evaluation
    env = gym.make(cc.ATARI_GAME)

    if len(sys.argv) > 1:
        env.seed(int(sys.argv[1]))

    observation = env.reset()
    total_score = 0.0

    for _ in range(10_000):

        if args.get("render", False):
            env.render()

        try:

            observation = ((observation)/255.0) * (1 - -1) + -1
            action = program.evaluate(np.transpose(observation, [2, 0, 1]))

        except MemoryError:
            return np.NINF

        except Exception as err:
            print("Individual:")
            print(str(program))
            program.draw_function_graph("problamtic_CGP_program")
            raise err

        assert env.action_space.contains(
            action), "CGP suggested an illegal action: " + action + "\nAvailable actions are: " + env.action_space

        observation, reward, done, info = env.step(action)

        total_score += reward

        if done:
            break

    env.close()

    import datetime
    # TODO: remove
    print(f"[{datetime.datetime.now()}]\tEvaluated individual. Score: {total_score}\n")

    return total_score


@ec.variators.crossover
def crossover(random: random.Random, mom: List, dad: List, args: Dict) -> List[List]:
    # still need to create doc

    # this seems to be very similar to : ec.variators.arithmetic_crossover
    # see for implementation https://github.com/aarongarrett/inspyred/blob/master/inspyred/ec/variators/crossovers.py#L216

    # using crossover from paper: A New Crossover Technique for Cartesian Genetic Programming

    bounder = args["_ec"].bounder

    def gen_offspring():
        ri = random.uniform(0.0, 1.0)
        iri = 1 - ri  # inverse of ri

        p1 = [iri * g for g in mom]
        p2 = [ri * g for g in dad]

        return bounder([p1[i] + p2[i] for i in range(len(mom))], args)

    return [gen_offspring(), gen_offspring()]


@ec.variators.mutator
def mutate(random: random.Random, candidate: List, args: Dict) -> List:

    # not implemented yet, leaving below as reference
    # note that this acts on a single candidate, so example below is not 100% correct

    # atari paper details in section 3.2 : Evolution

    # if we don't 'hit' the probability of mutation then don't mutate
    # and just return plain individual
    if random.random() > args["mutation_rate"]:
        return candidate

    output_nodes = candidate[-cc.N_OUTPUT_NODES*4:]
    inner_nodes = candidate[:-cc.N_OUTPUT_NODES*4]

    # For both output and inner nodes the mutation selects MUTP_{NODES || OUTPUT} nodes from the sets of output
    # or inner nodes, and new radom values in [0, 1] are assigned to that node

    def mutate_nodes_in_set(st, mut_prob):
        n_nodes = len(st)/4 - 1
        n_nodes_to_pick = round(n_nodes * mut_prob)

        indices_to_mutate = [random.randint(
            0, n_nodes) for _ in range(n_nodes_to_pick)]

        for i_n in indices_to_mutate:
            st[i_n*4:i_n*4+4] = [random.uniform(0.0, 1.0) for _ in range(4)]

        return st

    output_nodes = mutate_nodes_in_set(output_nodes, cc.MUTP_OUTPUT)
    inner_nodes = mutate_nodes_in_set(inner_nodes, cc.MUTP_NODES)

    candidate = inner_nodes + output_nodes

    bounder = args["_ec"].bounder

    return bounder(candidate, args)
