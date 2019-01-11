# This script runs the EA for every game in the "games" array, with every seed in the "seed array"
# and the results of all these runs are saved to the "results.csv" file

games=("Asteroids-v0" "Boxing-v0" "SpaceInvaders-v0" "Gravitar-v0" "MsPacman-v0")
seeds=(231 84 111)

for g in ${games[@]}; do for s in ${seeds[@]}; do

    echo
    echo ---------------------------------------
    echo running game $g with seed $s
    echo ---------------------------------------
    echo

    python main.py $s $g

done; done