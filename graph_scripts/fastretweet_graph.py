import networkx as nx
import os

import matplotlib.pyplot as plt

graph_dir = "fast_retweet.gml.gz"
root_dir = "/scratch1/ashwinba/graphs"
G = nx.read_gml(os.path.join(root_dir,graph_dir))
fig = plt.figure(1, figsize=(200, 80), dpi=60)
nx.draw(G, with_labels=True, font_weight='normal')
plt.savefig("/scratch1/ashwinba/cache/plt_fastretweet_img.png")
