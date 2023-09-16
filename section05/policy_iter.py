from __future__ import print_function, division
from builtins import range

import numpy as np
from grid_class import standard_grid, ACTION_SPACE, print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9


def get_tp_and_rew(grid):
    tp_ = {}
    rew_ = {}

    for i in range(grid.rows):
        for j in range(grid.cols):
            s_ = (i, j)
            if not grid.is_terminal(s_):
                for a_ in ACTION_SPACE:
                    s2_ = g.get_next_state(s_, a_)
                    tp_[(s_, a_, s2_)] = 1
                    if s2_ in grid.rewards:
                        rew_[(s_, a_, s2_)] = grid.rewards[s2_]

    return tp_, rew_


def eval_pol(grid, pol_, initV=None):
    if initV is None:
        V_ = {}
        for s_ in grid.all_states():
            V_[s_] = 0
    else:
        V_ = initV

    # Repeat until convergence
    it = 0
    while True:
        biggest_change = 0
        for s_ in grid.all_states():
            if not grid.is_terminal(s_):
                old_v = V_[s_]
                new_v = 0
                for a_ in ACTION_SPACE:
                    for s2_ in grid.all_states():
                        # action probability is deterministic
                        action_prob = 1 if pol_.get(s_) == a_ else 0

                        # reward is a function of (s, a, s_prime), 0 if not specified
                        r_ = rew.get((s_, a_, s2_), 0)
                        new_v += (
                            action_prob
                            * tp.get((s_, a_, s2_), 0)
                            * (r_ + GAMMA * V_[s2_])
                        )

                # after getting the new value, update the value table
                V_[s_] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V_[s_]))
        it += 1
        if biggest_change < SMALL_ENOUGH:
            break
    return V_


if __name__ == "__main__":
    g = standard_grid()
    tp, rew = get_tp_and_rew(g)

    policy = {}
    for s in g.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    print("Initial Policy: ")
    print_policy(policy, g)

    # Repeat unitl Convergence
    V = None
    while True:
        # Policy Evaluation
        V = eval_pol(g, policy, initV=V)

        # Policy Improvement
        is_policy_converged = True
        for s in g.actions.keys():
            old_a = policy[s]
            new_a = None
            best_value = float("-inf")

            # Loop through all possible actions to find the best action
            for a in ACTION_SPACE:
                v = 0
                for s2 in g.all_states():
                    r = rew.get((s, a, s2), 0)
                    v += tp.get((s, a, s2), 0) * (r + GAMMA * V[s2])

                if v > best_value:
                    best_value = v
                    new_a = a

            policy[s] = new_a
            if new_a != old_a:
                is_policy_converged = False

        if is_policy_converged:
            break

    print("Values:")
    print_values(V, g)

    print("Final Policy")
    print_policy(policy, g)
