from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_class import standard_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3


def print_values(V, g):
    for i in range(g.rows):
        print("----------------------------------")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                print(" %.2f|" % v, end="")
            else:
                print("%.2f|" % v, end="")
        print("")


def print_policy(P, g):
    for i in range(g.rows):
        print("----------------------------------")
        for j in range(g.cols):
            a = P.get((i, j), " ")
            print(" %s  |" % a, end="")
        print("")


def main():
    g = standard_grid


if __name__ == "__main__":
    main()
