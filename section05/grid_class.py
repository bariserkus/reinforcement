from __future__ import print_function, division

ACTION_SPACE = ("U", "D", "L", "R")


class Grid:  # Environment
    def __init__(self, rows, cols, start_position):
        self.rows = rows
        self.cols = cols
        self.i = start_position[0]
        self.j = start_position[1]
        self.actions = None
        self.rewards = None

    def set(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return self.i, self.j

    def is_terminal(self, s):
        return s not in self.actions

    def reset(self):
        # put agent back to start position
        self.i = 2
        self.j = 0
        return self.i, self.j

    def get_next_state(self, s, a):
        # this function gives the next state, when an action "a" is performed on a given state s
        i, j = s[0], s[1]
        if a in self.actions[(i, j)]:
            if a == "U":
                i -= 1
            elif a == "D":
                i += 1
            elif a == "R":
                j += 1
            elif a == "L":
                j -= 1
        return i, j

    def move(self, a):
        if a in self.actions[self.i, self.j]:
            if a == "U":
                self.i -= 1
            elif a == "D":
                self.i += 1
            elif a == "R":
                self.j += 1
            elif a == "L":
                self.j -= 1
        return self.rewards[(self.i, self.j)]

    def move_opposite(self, a):
        if a == "U":
            self.i += 1
        elif a == "D":
            self.i -= 1
        elif a == "R":
            self.j -= 1
        elif a == "L":
            self.j += 1
        assert self.current_state() in self.all_states()

    def game_over(self):
        return (self.i, self.j) not in self.actions

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


def standard_grid():
    # define a grid that describes the reward for arriving at each state
    # and possible actions at each state
    # the grid looks like this
    # x means you can't go there
    # s means start position
    # number means reward at that state
    # .  .  .  1
    # .  x  . -1
    # s  .  .  .
    g = Grid(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ("D", "R"),
        (0, 1): ("L", "R"),
        (0, 2): ("L", "R", "D"),
        (1, 0): ("U", "D"),
        (1, 2): ("U", "D", "R"),
        (2, 0): ("U", "R"),
        (2, 1): ("L", "R"),
        (2, 2): ("L", "R", "U"),
        (2, 3): ("L", "U"),
    }
    g.set(rewards, actions)
    return g


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
