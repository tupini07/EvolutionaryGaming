"""Here we define constants to be used in the algorithm"""

# these are taken from the Atari paper

N_INNER_NODES = 40 # aka C

# a note on node representation:
#   while reading the atari paper it seems to me that the structure of each genome is as follows
#   [ (16 real numbers, one for each output), (3 placeholder numbers representing the indexes of inputs), (4 numbers per each node), ]
N_TOTAL_GENES = 16 + 3 + (4 * N_INNER_NODES)
N_EVOLVABLE_GENES = N_TOTAL_GENES - 3 # we don't evolve input genes, so they won't be part of our genome. Placeholder values will be passed during evaluation

MUTP_NODES = 0.1 # mutation probability for inner nodes
MUTP_OUTPUT = 0.6 # mutation probability for output nodes

POPULATION_SIZE = 9

G_R = 0.1 # regulates how far back from a node a connection can be made  `r`