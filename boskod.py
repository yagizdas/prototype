import networkx as nx
import matplotlib.pyplot as plt

# Given bitstring
bitstring = "1011001"

# Split the bitstring into Part A and Part B
part_a = bitstring[:4]
part_b = bitstring[4:]

# Create a bipartite graph
B = nx.Graph()

# Define nodes for Part A and Part B
nodes_a = ['A1', 'A2', 'A3', 'A4']
nodes_b = ['B1', 'B2', 'B3']

# Add edges based on the bitstring
for i, bit_a in enumerate(part_a):
    if bit_a == '1':  # Node in Part A is selected
        for j, bit_b in enumerate(part_b):
            if bit_b == '1':  # Node in Part B is selected
                B.add_edge(nodes_a[i], nodes_b[j])

# Visualize the bipartite graph
pos = nx.bipartite_layout(B, nodes_a)
nx.draw(B, pos, with_labels=True, node_color=['skyblue' if node in nodes_a else 'lightgreen' for node in B.nodes()])
plt.show()