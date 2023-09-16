from __future__ import print_function, division
from builtins import range

# import numpy as np
from grid_class import standard_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3


def print_values(V, g):
    for i in range(g.rows):
        print("------------------------")
        print("|", end="")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")
    print("------------------------")


def print_policy(P, g):
    for i in range(g.rows):
        print("------------------------")
        print("|", end="")
        for j in range(g.cols):
            a = P.get((i, j), " ")
            print("  %s  |" % a, end="")
        print("")
    print("------------------------")


def main():
    transition_probs = {}
    rewards = {}

    g = standard_grid()

    for i in range(g.rows):
        for j in range(g.cols):
            s = (i, j)
            if not g.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = g.get_next_state(s, a)
                    transition_probs[(s, a, s2)] = 1
                    if s2 in g.rewards:
                        rewards[(s, a, s2)] = g.rewards[s2]

    print(transition_probs)

    # print(rewards)

    ### fixed policy ###
    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "U",
        (2, 1): "R",
        (2, 2): "U",
        (2, 3): "L",
    }
    print_policy(policy, g)


if __name__ == "__main__":
    main()
