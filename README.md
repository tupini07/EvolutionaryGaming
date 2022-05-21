# EvolutionaryGaming

Playing games has long been a popular way to test the
performance of automatic AI. This field has lately been
dominated by Generative Adversarial Networks, but other
techniques have proven quite effective as well. In this paper,
we apply a specific evolutionary technique called Cartesian
Genetic Programming (CGP) which will let us evolve
programs to play Atari games. The performance of our method
is then be compared with the state of the art in automatic
game playing.

The report for this project can be seen [here](/Report.pdf)


## Files

* `main.py` calls the main evolution algorithm. It can be given command line arguments: python3 main.py [random seed] [Atari game] [log filename]
* `evolutionary_problem.py` contains the functions required by the algorithm: generator, observer, evaluator, mutate, crossover
* `function_set.py` contains the function set.
* `constants.py` contains constants for the algorithm and the CGP programs.
* `CGP_program.py` contains the code of for a CGP program.
* `CGP_cell.py` contains the code for a cell in a CGP_program.


## Video

A video of the best individual produced by the algorithm playing the `Boxing` game can be seen below:

[![Youtube video for the Boxing game)](https://youtube-md.vercel.app/vE1OzXCKFUQ/640/360)](https://www.youtube.com/watch?v=vE1OzXCKFUQ)
