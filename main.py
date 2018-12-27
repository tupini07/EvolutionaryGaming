import random
from time import time
import inspyred
import evolutionary_problem as problem
import constants as cc

from inspyred import ec


def main(prng=None, display=False):
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.GA(prng)

    ea.selector = inspyred.ec.selectors.rank_selection

    ea.variator = [problem.mutate,
                   problem.crossover]

    # ea.replacer = inspyred.ec.replacers.steady_state_replacement
    # ea.replacer = inspyred.ec.replacers.plus_replacement
    ea.replacer = inspyred.ec.replacers.generational_replacement

    ea.terminator = inspyred.ec.terminators.generation_termination

    ea.observer = problem.observer

    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          pop_size=9, 
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          tournament_size=3,
                          max_generations=120,
                          num_selected=6, # for crossover, only this # of parents are chosen 
                          num_elites=2, # number of elites for generaitonal replacement
                          mutation_rate=0.7, 
                          render=False)  

    if display:
        best = max(final_pop)
        print('Best Solution: \n{0}'.format(str(best)))
    return ea


if __name__ == '__main__':
    main(display=True)
