import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.widgets import Slider


class GraphViz:
    def __init__(self, figsize=(20, 10)):

        self.figsize = figsize
        self.font_size = sum(figsize)

        self.nodes = ["WT", "PV", "G", "T", "B", "L", "L1", "L2", "B1", "B2", "B3", " "]
        self.batteries = [
            ("B1", "B", {"name": "PB1"}),
            ("B2", "B", {"name": "PB2"}),
            ("B3", "B", {"name": "PB3"}),
        ]
        self.topology = [
            ("T", " ", {"name": "T"}),
            ("B", " ", {"name": "B"}),
            ("L", " ", {"name": "L"}),
        ]
        self.res = [
            ("WT", "T", {"name": "WT"}),
            ("PV", "T", {"name": "PV"}),
            ("G", "T", {"name": "PG"}),
        ]
        self.loads = [("L1", "L", {"name": "L1"}), ("L2", "L", {"name": "L2"})]

        self.G = nx.Graph()
        self.G.add_nodes_from(self.nodes)
        self.G.add_edges_from(self.batteries)
        self.G.add_edges_from(self.topology)
        self.G.add_edges_from(self.res)
        self.G.add_edges_from(self.loads)

        self.edges = self.G.edges()

        self.p_color = "red"
        self.n_color = "black"

        self.pos = nx.kamada_kawai_layout(self.G)

        self.edge_colors = [self.p_color] * len(self.edges)
        self.weights = [1] * len(self.edges)

    def update_edges(self, data):

        self.edge_colors = []
        self.weights = []

        edge_dict = nx.get_edge_attributes(self.G, "name")

        for edge, name in edge_dict.items():
            value = data[name]

            if value >= 0:
                color = self.p_color
            else:
                color = self.n_color

            nx.set_edge_attributes(
                self.G, {edge: {"color": color, "weight": np.ceil(abs(value) / 5)}}
            )

        self.edge_colors = list(nx.get_edge_attributes(self.G, "color").values())
        self.weights = list(nx.get_edge_attributes(self.G, "weight").values())

    def draw(self, ax=None):

        nx.draw(
            self.G,
            self.pos,
            with_labels=True,
            edge_color=self.edge_colors,
            width=self.weights,
            font_size=self.font_size,
            node_size=4000,
            node_color="w",
            ax=ax,
        )

    def plot_with_slider(self, df):

        plt.figure(figsize=self.figsize)

        axg = plt.gca()

        self.update_edges(df.iloc[0])
        self.draw(axg)
        axg.set_title("Time: 00.00", fontsize=self.font_size)

        sindex = Slider(
            plt.axes([0.25, 0.03, 0.50, 0.02]),
            "Index",
            0,
            df.shape[0] - 1,
            valinit=0,
            dragging=True,
            valstep=1,
        )

        def update(index):
            plt.pause(0.001)
            i = sindex.val
            datapoint = df.iloc[int(np.floor(i))]
            self.update_edges(datapoint)
            axg.clear()
            self.draw(axg)
            axg.set_title("Time: {}:{}0".format(i // 6, i % 6), fontsize=self.font_size)

        sindex.on_changed(update)
