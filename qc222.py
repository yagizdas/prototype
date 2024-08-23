import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import random
from itertools import combinations
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

#Connect to IBM Quantum service
service = QiskitRuntimeService()

#Specify the backend
backend = service.backend('ibm_kyiv')  # Use 'ibmq_qasm_simulator' or a real quantum device

#Generate a random graph
nodes = 10
G = nx.erdos_renyi_graph(n=nodes, p=0.6)

weighted = True

for (u, v) in G.edges():
    if weighted:
        w = random.uniform(0, 1)
    else:
        w = 1
    G.edges[u, v]['weight'] = w

#QAOA parameters
depth = 8
rep = 1000
qubits = list(range(nodes))

#Initialization of the circuit (Hadamard gates)
def initialization(qc, qubits):
    for q in qubits:
        qc.h(q)

#Define the cost unitary
def cost_unitary(qc, qubits, gamma):
    for u, v in G.edges():
        qc.cx(u, v)
        qc.rz(2 * gamma * G[u][v]['weight'], v)
        qc.cx(u, v)

#Define the mixer unitary
def mixer_unitary(qc, qubits, beta):
    for q in qubits:
        qc.rx(2 * beta, q)

#Create the QAOA circuit
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

#Define the cost function
def cost_function(params):
    qc = create_circuit(params)
    t_qc = transpile(qc, backend)
    new_circuit = transpile(t_qc, backend, shots=rep)
    job = backend.run(new_circuit)
    result = job.result()
    counts = result.get_counts(qc)

    total_cost = 0
    for bitstring, count in counts.items():
        bit_list = [int(bit) for bit in bitstring]
        for u, v in G.edges():
            total_cost += G[u][v]['weight'] * 0.5 * ((1 - 2 * bit_list[u]) * (1 - 2 * bit_list[v]) - 1) * count
    total_cost = total_cost / rep
    return total_cost

#Perform optimization
from scipy.optimize import minimize

optimal_params = None
optimal_val = np.inf

for i in range(8):
    init_params = np.random.uniform(-np.pi, np.pi, 2 * depth)
    res = minimize(cost_function, x0=init_params, method="COBYLA", options={'maxiter':200})
    print(f"Iteration {i}: cost = {res.fun}")

    if res.fun < optimal_val:
        optimal_params = res.x
        optimal_val = res.fun

#Run the final circuit with the optimal parameters
qc = create_circuit(optimal_params)
t_qc = transpile(qc, backend)
new_circuit = transpile(t_qc, backend, shots=rep)
job = backend.run(new_circuit)
result = job.result()
counts = result.get_counts(qc)

#Process the results
quantum_preds = []
for bitstring in counts:
    temp = []
    for pos, bit in enumerate(bitstring):
        if bit == '1':
            temp.append(pos)
    quantum_preds.append(temp)

#Calculate classical cuts for comparison
sub_lists = []
for i in range(nodes + 1):
    temp = [list(x) for x in combinations(G.nodes(), i)]
    sub_lists.extend(temp)

cut_classic = []
for sub_list in sub_lists:
    cut_classic.append(nx.algorithms.cuts.cut_size(G, sub_list, weight='weight'))

cut_quantum = []
for cut in quantum_preds:
    cut_quantum.append(nx.algorithms.cuts.cut_size(G, cut, weight='weight'))

print("Quantum mean cut:", np.mean(cut_quantum))
print("Max classical cut:", np.max(cut_classic))
print("Ratio:", np.mean(cut_quantum) / np.max(cut_classic))
