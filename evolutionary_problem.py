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

    #initialize random action for frame skip
    action = env.action_space.sample()

    for _ in range(10_000):

        if render:
            env.render()

        #frame skip
        skip = np.random.choice([True, False], p=[0.25,0.75])
        
        if not skip:
            action = program.evaluate(np.transpose(observation, [2, 0, 1]))

        assert env.action_space.contains(
            action), "CGP suggested an illegal action: " + action + "\nAvailable actions are: " + env.action_space

        observation, reward, done, info = env.step(action)

        total_score += reward 

        if done:
            break

    env.close()

    return total_score


@ec.variators.mutator
def mutate(random: random.Random, candidate: List, args: Dict) -> List:

    # not implemented yet, leaving below as reference
    # note that this acts on a single candidate, so example below is not 100% correct

    # atari paper dateils in section 3.2 : Evolution

    # see `constants` module for mutation probabilities

    # TODO mutation should be done with different probabilities on output nodes and on inner nodes, so they need to be separated
    # note that here we also need to take into account the probability of creating a recurrent connection
    output_nodes = candidate[-cc.N_OUTPUT_NODES:]
    inner_nodes = candidate[:-cc.N_OUTPUT_NODES]

    for f, _ in enumerate(candidate):
        candidate[f] += random.uniform(-0.01, 0.01)
  
    bounder = args["_ec"].bounder

    return bounder(candidate, args)
