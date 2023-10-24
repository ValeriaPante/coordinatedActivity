import os
import numpy as np
import pandas as pd

import warnings

import networkx as nx
import matplotlib.pyplot as plt

import random

OUTPUT_DIR = "/scratch1/ashwinba/cache/outputs/user_count_countrywise"
GEXF_DIR = "/scratch1/ashwinba/cache/plt_countrywise_fastretweet_v2.gexf"
METHOD_NAME = "fastretweet"

# Generate Random Colors


def random_color():
    levels = range(32, 256, 32)
    return tuple(random.choice(levels)/255 for _ in range(3))


def draw_countrywise(graph, method_name):
    warnings.warn("Entered the method")
    G = nx.Graph()

    countries = set([graph.nodes[node]['country']
                    for node in list(graph.nodes)])

    for country in countries:
        G.add_node(country, node_color=random_color())
    country_colors = [random_color() for _ in countries]

    for edge in graph.edges:
        from1, to = edge
        from_country = graph.nodes[from1]['country']
        to_country = graph.nodes[to]['country']
        if G.has_edge(from_country, to_country):
            # we added this one before, just increase the weight by one
            G[from_country][to_country]['weight'] += 0.000013
        else:
            # new edge. add with weight=1
            G.add_edge(from_country, to_country, weight=0.00009)

    G.remove_edges_from(nx.selfloop_edges(G))

    weights = [G[u][v]['weight'] for u, v in G.edges]
    weights = [(weight/max(weights)) for weight in weights]
    pos = nx.spring_layout(G, k=100.85)
    nx.draw(G, pos, with_labels=True, node_color=country_colors, width=weights)
    plt.savefig(os.path.join(OUTPUT_DIR, method_name+"_image.png"))
    warnings.warn("Completed Generating the output file (png)")
    # plt.show()


graph = nx.read_gexf(GEXF_DIR)
warnings.warn(METHOD_NAME)
# Calling the method
draw_countrywise(graph, method_name=METHOD_NAME)
