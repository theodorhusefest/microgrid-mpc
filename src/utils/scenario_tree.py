import time
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pprint import pprint
from casadi import SX, horzcat, vertcat


class Node:
    def __init__(self, id_, level, pv, l):
        self.id = id_
        self.level = level
        self.children = []
        self.scenarios = []
        self.parent = None
        self.pv = np.max([pv, 0])
        self.l = np.max([l, 0])
        self.prob = 1

        self.leaf = False

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self, parent):
        self.parent = parent

    def get_value(self, signal):
        if signal == "pv":
            return self.pv
        elif signal == "l":
            return self.l
        elif signal == "prob":
            return self.prob
        else:
            raise ValueError("Signal not in node")

    def is_leaf(self, N):
        if self.level == N:
            self.leaf = True
            return True

    def print_children(self):
        print("\n************")
        print("ID:", self.id)
        print("PV:", self.pv)
        print("L:", self.l)
        print("Level", self.level)
        print("Probability", self.prob)
        print("Scenarios:", self.scenarios)
        print("************ \n")
        for child in self.children:
            child.print_children()

    def __str__(self):
        return "Node nr" + str(self.id)

    def __eq__(self, other):
        return self.id == other.id


def build_scenario_tree(N, Nr, branching, pv_ref, pv_std, l_ref, l_std):

    assert N >= Nr, "N has to be higher or equal to Nr"

    b_factor = np.append(np.ones(Nr, dtype=int) * branching, np.ones(N - Nr, dtype=int))
    b_factor = np.append(b_factor, 0)

    root = Node(0, 0, pv_ref[0], l_ref[0])

    ids = 1
    to_explore = [root]
    scenario_num = 0
    leaf_nodes = []
    while to_explore:
        current = to_explore.pop(0)
        b = b_factor[current.level]
        assert b <= 3, "Branching factor over three not allowed."
        for j in range(b):
            pv_k = pv_ref[current.level + 1] - pv_ref[current.level]
            l_k = l_ref[current.level + 1] - l_ref[current.level]
            pv = current.pv + pv_k
            l = current.l + l_k

            pv_error = np.abs(np.random.normal(0, pv_std))
            l_error = np.abs(np.random.normal(0, l_std))
            p = 0.68
            if b == 3:
                if j == 2:
                    pv = pv - pv * pv_error
                    l = l + l * l_error
                elif j == 1:
                    pv = pv + pv * pv_error
                    l = l - l * l_error
            elif b == 2:
                if j == 1:
                    pv = pv - pv * pv_error
                    l = l + l * l_error
                elif j == 0:
                    pv = pv + pv * pv_error
                    l = l - l * l_error
            else:
                p = 1

            temp = Node(ids, current.level + 1, pv, l)
            temp.prob = current.prob * p
            temp.set_parent(current)
            if temp.is_leaf(N):
                leaf_nodes.append(temp)
                scenario = "scenario" + str(scenario_num)
                scenario_num += 1
                add_scenario_for_parents(temp, scenario)

            current.add_child(temp)
            to_explore.append(temp)

            ids += 1
    return root, leaf_nodes


def traverse_leaf_to_root(leaf, signal):
    """
    Find path from leaf to root, and returns it reversed
    """
    scenario = []
    current = leaf
    while current:
        scenario.append(current.get_value(signal))
        current = current.parent

    return np.asarray(scenario[::-1])


def get_scenarios(leaf_nodes, signal):
    """
    Extracts all scenarios from leaf to root
    """
    scenarios = []
    for node in leaf_nodes:
        if node.leaf:
            scenarios.append(traverse_leaf_to_root(node, signal))

    return np.asarray(scenarios)


def add_scenario_for_parents(leaf, scenario):
    """
    Starts from leaf, and adds scenario-label to all nodes on path to root
    """
    current = leaf
    while current:
        current.scenarios.append(scenario)
        current = current.parent


if __name__ == "__main__":
    mu = 1
    std = 0.2
    x = np.linspace(mu - std, mu + std, 8)
    plt.plot(x, stats.norm.pdf(x, mu, std))
    plt.show()
    print(stats.norm.pdf(x, mu, std))
