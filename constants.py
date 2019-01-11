"""Here we define constants to be used in the algorithm"""
import gym

# these are taken from the Atari paper

# ATARI_GAME = "SpaceInvaders-v0"
ATARI_GAME = "Boxing-v0"
EA_MAXIMIZE = True

_env = gym.make(ATARI_GAME)
_env.reset()
N_OUTPUT_NODES = _env.action_space.n
_env = None # free variable so that atari gets GC

N_INNER_NODES = 40 # aka C

# a note on node representation:
#   while reading the atari paper it seems to me that the structure of each genome is as follows
#   [ 3 input cells, +  N_INNER_NODES + 14 ouput cells]

N_EVOLVABLE_GENES = (N_INNER_NODES * 4) + (N_OUTPUT_NODES * 4) # we don't evolve input genes, so they won't be part of our genome. Placeholder values will be passed during evaluation

MUTP_NODES = 0.1 # mutation probability for inner nodes
MUTP_OUTPUT = 0.6 # mutation probability for output nodes

G_R = 0.1 # regulates how far back from a node a connection can be made  `r`