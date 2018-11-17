import random
from time import time
import inspyred
import evolutionary_problem as problem


from inspyred import ec


@ec.evaluators.evaluator
def evaluate(candidate, args):
    # not implemented yet, but the flow will be something like this
    # note that this operates on one candidate at a time

    fitness =  random.random()

    return fitness


def main(prng=None, display=False):
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.EvolutionaryComputation(prng)

    ea.selector = inspyred.ec.selectors.tournament_selection

    ea.variator = [# problem.mutate, # gaussian mutation should be enough
                   problem.crossover,
                   inspyred.ec.variators.gaussian_mutation]
    
    ea.replacer = inspyred.ec.replacers.steady_state_replacement
    ea.terminator = inspyred.ec.terminators.generation_termination
    
    ea.observer = problem.observer

    final_pop = ea.evolve(generator=problem.generator,
                          evaluator=problem.evaluator,
                          pop_size=100,
                          bounder=problem.bounder,
                          maximize=problem.maximize,
                          tournament_size=7,
                          num_selected=2,
                          max_generations=300,
                          mutation_rate=0.2)

    if display:
        best = max(final_pop)
        print('Best Solution: \n{0}'.format(str(best)))
    return ea


if __name__ == '__main__':
    main(display=True)
