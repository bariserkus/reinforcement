from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_class import standard_grid, ACTION_SPACE, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def get_tp_and_rew(g):
    tp = {}
    rew = {}

    for i in range(g.rows):
        for j in range(g.cols):
            s = (i, j)
            if not g.is_terminal(s):
                for a in ACTION_SPACE:
                    s2 = g.get_next_state(s, a)
                    tp[(s, a, s2)] = 1
                    if s2 in g.rewards:
                        rew[(s, a, s2)] = g.rewards[s2]

    return tp, rew


def eval_pol(g, pol, initV=None):
    if initV is None:
        V = {}
        for s in g.all_states():
            V[s] = 0
    V = initV

    # Repeat until convergence
    it = 0
    while True:
        biggest_change = 0
        for s in g.all_states():
            if not g.is_terminal(s):
                old_v = V[s]
                new_v = 0
                for a in ACTION_SPACE:
                    for s2 in g.all_states():
                        # action probability is deterministic
                        action_prob = 1 if policy.get(s) == a else 0

                        # reward is a function of (s, a, s_prime), 0 if not specified
                        r = rewards.get((s, a, s2), 0)
                        new_v += (
                            action_prob
                            * transition_probs.get((s, a, s2), 0)
                            * (r + gamma * V[s2])
                        )

                # after getting the new value, update the value table
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))

        print("Iteration: ", it, "Biggest Change:", biggest_change)
        print_values(V, g)
        it += 1
        if biggest_change < SMALL_ENOUGH:
            break
    print("\n\n")
