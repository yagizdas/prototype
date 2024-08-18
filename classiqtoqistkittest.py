import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from scipy.optimize import minimize
from itertools import combinations
import random

# Initialize IBM Quantum service
service = QiskitRuntimeService()
backend = service.backend(name = 'ibm_kyiv')  # Use a real quantum device if needed

# Generate a random bipartite graph
donors = 5
recipients = 5
nodes = donors + recipients
G = nx.complete_bipartite_graph(donors, recipients)

# Assign random weights to the edges
for (u, v) in G.edges():
    G.edges[u, v]['weight'] = random.uniform(0, 1)

# QAOA parameters
depth = 8
rep = 1000
qubits = list(range(nodes))

# Initialize the quantum circuit with Hadamard gates
def initialization(qc, qubits):
    for q in qubits:
        qc.h(q)

# Define the cost unitary
def cost_unitary(qc, qubits, gamma):
    for u, v in G.edges():
        qc.cx(u, v)
        qc.rz(2 * gamma * G[u][v]['weight'], v)
        qc.cx(u, v)

# Define the mixer unitary
def mixer_unitary(qc, qubits, beta):
    for q in qubits:
        qc.rx(2 * beta, q)

# Create the QAOA circuit
def create_circuit(params):
    gammas = params[0::2]
    betas = params[1::2]

    qc = QuantumCircuit(nodes)
    initialization(qc, qubits)

    for d in range(depth):
        cost_unitary(qc, qubits, gammas[d])
        mixer_unitary(qc, qubits, betas[d])

    qc.measure_all()
    return qc

# Define the cost function
def cost_function(params):
    qc = create_circuit(params)
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc, shots=rep)
    result = job.result()
    counts = result.get_counts(qc)

    total_cost = 0
    for bitstring, count in counts.items():
        bit_list = [int(bit) for bit in bitstring]

        for i, u in enumerate(qubits[:donors]):
            for j, v in enumerate(qubits[donors:]):
                if bit_list[u] and bit_list[v]:  # If both donor and recipient are connected
                    total_cost += G[u][v]['weight'] * count
                else:
                    total_cost -= 1e6  # Penalize if a donor or recipient is connected to more than one

    total_cost = total_cost / rep
    return -total_cost  # Minimize the negative cost

# Perform optimization
optimal_params = None
optimal_val = np.inf

for i in range(8):
    init_params = np.random.uniform(-np.pi, np.pi, 2 * depth)
    res = minimize(cost_function, x0=init_params, method="COBYLA", options={'maxiter': 200})
    print(f"Iteration {i}: cost = {-res.fun}")

    if res.fun < optimal_val:
        optimal_params = res.x
        optimal_val = res.fun

# Run the final circuit with the optimal parameters
qc = create_circuit(optimal_params)
t_qc = transpile(qc, backend)
job = backend.run(t_qc, shots=rep)
result = job.result()
counts = result.get_counts(qc)

# Process the results
quantum_preds = []
for bitstring in counts:
    temp = []
    for pos, bit in enumerate(bitstring):
        if bit == '1':
            temp.append(pos)
    quantum_preds.append(temp)

print("Quantum predictions:", quantum_preds)
