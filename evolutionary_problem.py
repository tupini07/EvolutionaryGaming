"""Here we define the helper operators for inspyred: mutation, crossover, and selection

Note that we can't place these in a class because the function annotations don't like the (self) parameter
So for the moment they can stay as a module
"""

import itertools
import math
import random

from inspyred import ec
from inspyred.benchmarks import Benchmark
from constants import N_TOTAL_GENES


bounder = ec.Bounder([0.0] * N_TOTAL_GENES, [1.0] * N_TOTAL_GENES)

maximize = True



## Util functions for GC

def generator(random, args):
    return [random.uniform(0.0, 1.0) for _ in range(N_TOTAL_GENES)]


def observer(population, num_generations, num_evaluations, args):
    best = max(population)
    print(f"GEN: {num_generations} \t Best fitness: {best.fitness} \t Pop size {len(population)}")


@ec.variators.crossover
def crossover(random, mom, dad, args):
    # not implemented yet

    # operates on only 2 parent candidates at a time. It should return an iterable sequence of offspring (typically two)

    return [mom, dad]


@ec.evaluators.evaluator
def evaluator(candidate, args):
    # not implemented yet, but the flow will be something like this
    # note that this operates on one candidate at a time

    fitness = sum(candidate)

    return fitness


@ec.variators.mutator
def mutate(random, candidate, args):
    
    # not implemented yet, leaving below as reference
    # note that this acts on a single candidate, so example below is not 100% correct

    # atari paper dateils in section 3.2 : Evolution
    
    # see `constants` module for mutation probabilities

    # mut_rate = args.setdefault('mutation_rate', 0.1)
    # bounder = args['_ec'].bounder
    # for i, cs in enumerate(candidates):
    #     for j, (c, lo, hi) in enumerate(zip(cs, bounder.lower_bound, bounder.upper_bound)):
    #         if random.random() < mut_rate:
    #             x = c[0] + random.gauss(0, 1) * (hi - lo)
    #             y = c[1] + random.gauss(0, 1) * (hi - lo)
    #             candidates[i][j] = (x, y)
    #     candidates[i] = bounder(candidates[i], args)

    return candidate
