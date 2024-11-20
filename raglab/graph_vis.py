#! /usr/bin/env python3

import random

import networkx as nx
from pyvis.network import Network

from raglab import ConfigManager


class GraphVis:
    def __init__(self, config_manager: ConfigManager) -> None:
        self.working_dir = config_manager.working_dir
        self.graph_path = config_manager.working_dir + \
            "graph_chunk_entity_relation.graphml"
        self.doc_name = config_manager.doc_file.split(".txt")[0].split("/")[-1]

    def convert(self) -> None:
        # Load the GraphML file
        G = nx.read_graphml(self.graph_path)

        # Create a Pyvis network
        net = Network(height="100vh", notebook=True)

        # Convert NetworkX graph to Pyvis network
        net.from_nx(G)

        # Add colors to nodes
        for node in net.nodes:
            node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        # Save and display the network
        net.show(f"{self.working_dir}{self.doc_name}.html")
