import itertools
import math
import random
from typing import Dict, List

import numpy as np
from inspyred import ec
from inspyred.benchmarks import Benchmark

import constants
from CGP_program import CGP_program

bounder = ec.Bounder(0.0, 1.0)

maximize = True


# Util functions for GC

def generator(random: random.Random, args: Dict) -> List:
    """
    Generate an individual of length `constants.N_EVOLVABLE_GENES` where every element in it is a random number x 
    where 0 <= x <= 1 

    The structure of the genome is as follos
       [ (16 real numbers, one for each output), (3 placeholder numbers representing the indexes of inputs), (4 numbers per each node), ]

    But only the 16 inputs and C (4 numbers per output) will be evolved, so the inputs are not considered part of t genome 

    Parameters
    ----------
    random : random.Random
        The random generator passed to inspyred
    args : Dict
        Dictionary of arguments passed to inspyred

    Returns
    -------
    List
        An individual of length `constants.N_EVOLVABLE_GENES` where every element in it is a random number x 
        where 0 <= x <= 1 
    """

    return [random.uniform(0.0, 1.0) for _ in range(constants.N_EVOLVABLE_GENES)]


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
def evaluator(candidate: List, args: Dict) -> float:
    # not implemented yet, but the flow will be something like this
    # note that this operates on one candidate at a time

    candidate = ([0]*3*4) + candidate # the [0] *3*4 represent the genome for the input cells
    cpg_genome = [candidate[i:i+4]
                  for i in range(0, len(candidate), 4)]  # split into chunks

    program = CGP_program(cpg_genome)

    # todo this is just a test input, need to connect with atari simulator.
    # remember that the evaluation is done accross an entire game. So we evaluate and play the game until
    # it ends, the final score is what we return from this function

    # Note: that currently we're always getting 15 as the output. This is just because 15 is the "highest" action index
    # and the program is maximizing
    return program.evaluate(np.array([[[1, 2, 3, 4],[1, 2, 3, 4]], [[6, 6, 5, 4],[6, 6, 5, 4]], [[3, 4, 2, 1],[3, 4, 2, 1]]]))


@ec.variators.mutator
def mutate(random: random.Random, candidate: List, args: Dict) -> List:

    # not implemented yet, leaving below as reference
    # note that this acts on a single candidate, so example below is not 100% correct

    # atari paper dateils in section 3.2 : Evolution

    # see `constants` module for mutation probabilities

    # todo mutation should be done with different probabilities on output nodes and on inner nodes, so they need to be separated
    output_nodes = candidate[-16:]
    inner_nodes = candidate[:-16]

    for f, _ in enumerate(candidate):
        candidate[f] += random.uniform(-0.01, 0.01)

    bounder = args["_ec"].bounder

    return bounder(candidate, args)
