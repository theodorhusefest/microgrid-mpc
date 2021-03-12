import time
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
        self.parent = None
        self.pv = np.max([pv, 0])
        self.l = np.max([l, 0])

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
        else:
            raise ValueError("Signal not in node")

    def is_leaf(self, N):
        if self.level == N:
            self.leaf = True
            return True

    def print_children(self):
        print("ID:", self.id)
        print("PV:", self.pv)
        print("L:", self.l)
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
    nodes = [root]
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

            temp = Node(ids, current.level + 1, pv, l)
            temp.set_parent(current)
            if temp.is_leaf(N):
                leaf_nodes.append(temp)

            current.add_child(temp)
            to_explore.append(temp)

            ids += 1
            nodes.append(temp)
    return leaf_nodes


def traverse_leaf_to_root(node, signal):
    scenario = []
    current = node
    while current:
        scenario.append(current.get_value(signal))
        current = current.parent

    return np.asarray(scenario[::-1])


def get_scenarios(nodes, signal):
    scenarios = []
    for node in nodes:
        if node.leaf:
            scenarios.append(traverse_leaf_to_root(node, signal))

    return np.asarray(scenarios)


def find_n_common_nodes(node1, node2):
    """
    Returns the number of common nodes for two nodes
    """

    common_nodes = 0
    while node1.parent:
        node1 = node1.parent
        node2 = node2.parent
        if node1 == node2:
            common_nodes += 1

    return common_nodes


def Ej_j1(N, Nr, Nu, p, Nc):
    """
    Returns the E matrix for two concecutive scenarios
    """
    base = SX.eye(Nu * Nc)

    zeros = SX.zeros((Nu * Nc, Nu * N - base.shape[1]))
    E = horzcat(base, zeros)
    return E


def build_E_matrix(Ejs, N, Nu, Ns, p):
    """
    Returns the E-matrices as [E1, E_2, ..., E_N]
    """
    ind = 0
    E_bar = []
    for j in range(Ns - 1):
        if j == 0:
            Ej_bar = Ejs[j]
            pad = SX.zeros((p - Ej_bar.shape[0], Nu * N))
            Ej_bar = vertcat(Ej_bar, pad)

        elif j == 1:
            Ej_bar = -Ejs[j - 1]
            Ej_bar = vertcat(-Ejs[j - 1], Ejs[j])
            pad = SX.zeros((p - Ej_bar.shape[0], Nu * N))
            Ej_bar = vertcat(Ej_bar, pad)
        elif j >= 2:
            pad_above = SX.zeros((ind - Ejs[j - 1].shape[0], Nu * N))
            Ej_bar = vertcat(pad_above, -Ejs[j - 1], Ejs[j])
            pad = SX.zeros((p - Ej_bar.shape[0], Nu * N))
            Ej_bar = vertcat(Ej_bar, pad)

        ind += Ejs[j].shape[0]
        E_bar.append(Ej_bar)

    pad_above = np.zeros((ind - Ejs[-1].shape[0], Nu * N))
    Ej_bar = vertcat(pad_above, -Ejs[-1])
    E_bar.append(Ej_bar)

    return E_bar


if __name__ == "__main__":

    N = 2
    Nr = 2
    Nu = 4
    N_scenarios = 2 ** Nr
    p = 20
    E1_2 = Ej_j1(N, Nr, Nu, p, 2)
    E2_3 = Ej_j1(N, Nr, Nu, p, 1)
    E3_4 = Ej_j1(N, Nr, Nu, p, 2)
    Ejs = [E1_2, E2_3, E3_4]
    ind = 0
    E_bar = build_E_matrix(Ejs, N, Nu, N_scenarios, p)
    for E in E_bar:
        print(E)