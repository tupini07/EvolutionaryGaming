import itertools
import math
import random
from typing import Dict, List

import numpy as np
from inspyred import ec
from inspyred.benchmarks import Benchmark

import constants as cc
import gym
from CGP_program import CGP_program

bounder = ec.Bounder(0.0, 1.0)

maximize = True


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
    best = max(population)
    print(f"GEN: {num_generations} \t Best fitness: {best.fitness}")

@ec.evaluators.evaluator
def evaluator(candidate: List, args: Dict, render=True) -> float:
    # not implemented yet, but the flow will be something like this
    # note that this operates on one candidate at a time

    # the [0] *3*4 represent the genome for the input cells
    candidate = ([0]*3*4) + candidate
    cpg_genome = [candidate[i:i+4]
                  for i in range(0, len(candidate), 4)]  # split into chunks of 4

    program = CGP_program(cpg_genome)

    env = gym.make(cc.ATARI_GAME)
    observation = env.reset()
    total_score = 0.0

    # initialize random action for frame skip
    action = env.action_space.sample()

    for _ in range(10_000):

        if render:
            env.render()

        assert env.action_space.contains(
            action), "CGP suggested an illegal action: " + action + "\nAvailable actions are: " + env.action_space

        observation, reward, done, info = env.step(action)

        total_score += reward

        if done:
            break

    env.close()

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

        print(len(p1), " -- ", len(p2))
        return bounder([p1[i] + p2[i] for i in range(len(mom))], args)

    if random.random() < args["crossover_rate"]:
        return [gen_offspring(), gen_offspring()]

    else:
        return [mom, dad]


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
        n_nodes = len(st)/4
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
