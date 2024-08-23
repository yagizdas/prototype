import networkx as nx
import matplotlib.pyplot as plt
import random

# Number of Donors and Receivers
num_donors = 4
num_receivers = 3

# Create integer-based node names
donors = list(range(num_donors))  # [0, 1, 2, 3]
receivers = list(range(num_donors, num_donors + num_receivers))  # [4, 5, 6]

# Compatibility matrix where True means a match is possible
# This can be based on medical data like blood type, etc.
compatibility = {
    (0, 4): True,
    (0, 5): False,
    (0, 6): True,
    (1, 4): True,
    (1, 5): True,
    (1, 6): True,
    (2, 4): False,
    (2, 5): True,
    (2, 6): False,
    (3, 4): True,
    (3, 5): False,
    (3, 6): True,
}
# Create a bipartite graph
B = nx.Graph()

# Add nodes with the bipartite label
B.add_nodes_from(donors, bipartite=0)
B.add_nodes_from(receivers, bipartite=1)

# Add edges based on compatibility
for donor, receiver in compatibility:
    if compatibility[(donor, receiver)]:
        B.add_edge(donor, receiver)
nx.draw(B, with_labels = True)  

pos = {}

# Assign positions to donors (left side)
for i, donor in enumerate(donors):
    pos[donor] = (-1, i)

# Assign positions to receivers (right side)
for i, receiver in enumerate(receivers):
    pos[receiver] = (1, i)
weighted = True
for (u, v) in B.edges():
    if weighted:
        w = random.uniform(0, 1)
    else:
        w = 1
    B.edges[u, v]['weight'] = w

# Draw the graph
plt.figure(figsize=(10, 7))
edge_labels = nx.get_edge_attributes(B, 'weight')
nx.draw(B, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12, font_weight='bold')
nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, font_color='red')
plt.title("Bipartite Graph of Donors and Receivers with Weights", size=15)
plt.show()