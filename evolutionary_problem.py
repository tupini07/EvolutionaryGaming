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
    print(f"GEN: {num_generations} \t Best fitness: {best.fitness}")


@ec.variators.crossover
def crossover(random, mom, dad, args):
    # still need to create doc

    # this seems to be very similar to : ec.variators.arithmetic_crossover
    # see for implementation https://github.com/aarongarrett/inspyred/blob/master/inspyred/ec/variators/crossovers.py#L216

    # using crossover from paper: A New Crossover Technique for Cartesian Genetic Programming
    
    bounder = args["_ec"].bounder

    def gen_offspring():
        ri = random.uniform(0.0, 1.0)
        iri = 1 - ri # inverse of ri

        p1 = [iri * g for g in mom]
        p2 = [ri * g for g in dad]

        return bounder([p1[i] + p2[i] for i in range(len(mom))], args)

    return [gen_offspring(), gen_offspring()]


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

    for f, _ in enumerate(candidate):
        candidate[f] += random.uniform(-0.2, 0.2)

    bounder = args["_ec"].bounder

    return bounder(candidate, args)
