import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary, maximize
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

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
    (0, 5): True,
    (0, 6): True,
    (1, 4): True,
    (1, 5): True,
    (1, 6): True,
    (2, 4): True,
    (2, 5): True,
    (2, 6): True,
    (3, 4): True,
    (3, 5): True,
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





#service = QiskitRuntimeService(channel="ibm_quantum", token="4d675380adc3c0291ae0f7eebeb5011cf50f5ec7822256ec60f44a95eeae7a62ef3bed681f5b0ec3e8af0b30d7cb77445c71f13b9fbb90dbceb8d205dca56ad1")
#backend = service.backend(name = 'ibm_kyiv')  # Use a real quantum device if needed

#Local simulator parameters
aer_sim = AerSimulator(max_parallel_threads=10)
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ['u3'])

# sorunlu graph

nx.draw(B, with_labels = True)  
weighted = True

#PYOMO parametreleri
'''
model = ConcreteModel()

# Decision variables: x[i,j] = 1 if donor i is matched with recipient j
model.x = Var(B.edges(), within=Binary)

# Objective function: maximize the total weight of connections
def objective_rule(model):
    return sum(B.edges[i, j]['weight'] * model.x[i, j] for i, j in B.edges())
model.obj = Objective(rule=objective_rule, sense=maximize)  # Minimize negative, equivalent to maximize

#Constraints: Each donor and recipient is matched only once
def donor_constraint_rule(model, i):
    return sum(model.x[i, j] for j in G.neighbors(i) if (i, j) in model.x) <= 1
model.donor_constraint = Constraint(range(donors), rule=donor_constraint_rule)

def recipient_constraint_rule(model, j):
    return sum(model.x[i, j] for i in G.neighbors(j) if (i, j) in model.x) <= 1
model.recipient_constraint = Constraint(range(donors, donors + recipients), rule=recipient_constraint_rule)
'''

#QAOA KISMI 

# QAOA parameters
depth = 8
rep = 1000
qubits = list(range(num_donors + num_receivers))

# Her qubit için bir hadamard gate
def initialization(qc, qubits):
    for q in qubits:
        qc.h(q)

# Define the cost hamiltonian (CNOT,RZ,CNOT at bağlantılı nodes)
def cost_unitary(qc, qubits, gamma):
    for i, j in B.edges():
        qc.cx(qubits[i], qubits[j])
        qc.rz(2 * gamma * B.edges[i, j]['weight'], qubits[j])
        qc.cx(qubits[i], qubits[j])

# Define the mixer unitary
def mixer_unitary(qc, qubits, beta):
    for q in qubits:
        qc.rx(2 * beta, q)

# Create the QAOA circuit
def create_circuit(params):
    gammas = [j for i, j in enumerate(params) if i % 2 == 0]
    betas = [j for i, j in enumerate(params) if i % 2 == 1]

    qc = QuantumCircuit(num_donors + num_receivers)
    initialization(qc, qubits)

    for d in range(depth):
        cost_unitary(qc, qubits, gammas[d])
        mixer_unitary(qc, qubits, betas[d])

    qc.measure_all()
    return qc


# Define the cost function
def cost_function(params):
    qc = create_circuit(params)
    transpiled_qc = transpile(qc, backend=aer_sim)
    job = aer_sim.run(transpiled_qc, shots=rep, noise_model=noise_model)
    result = job.result()
    counts = result.get_counts()
    
    total_cost = 0
    for bitstring, count in counts.items():
        bit_list = [int(bit) for bit in bitstring]
        # Penalty for multiple connections
        penalty = 0
        for node in range(num_donors + num_receivers):
            connected_edges = [1 if (bit_list[i] == 1 and bit_list[j] == 1) else 0 for i, j in B.edges() if i == node or j == node]
            if sum(connected_edges) > 1:
                penalty += sum(connected_edges) - 1  # Penalize multiple connections
        
        for i, j in B.edges():
            total_cost += B.edges[i, j]['weight'] * 0.5 * ((1 - 2 * bit_list[i]) * (1 - 2 * bit_list[j]) - 1) * count
        total_cost += penalty * 10000  # Add the penalty to the total cost
    total_cost = total_cost / rep
    return total_cost


# Optimize using QAOA and Pyomo
optimal_params = None
optimal_val = np.inf

for _ in range(8):
    init = np.random.uniform(-np.pi, np.pi, 2 * depth)
    res = minimize(cost_function, x0=init, method="COBYLA", options={'maxiter':200})
    if res.fun < optimal_val:
        optimal_params = res.x
        optimal_val = res.fun

# Run the final circuit with the optimal parameters
qc = create_circuit(optimal_params)
transpiled_circuit = transpile(qc, backend=aer_sim)
job = aer_sim.run(transpiled_circuit, shots=rep, noise_model=noise_model)
result = job.result()
counts = result.get_counts()
print(counts)

plot_histogram(counts)

quantum_preds = []
for bitstring in counts:
    temp = []
    for pos, bit in enumerate(bitstring):
        if bit == '1':
            temp.append(pos)
    quantum_preds.append(temp)
print(quantum_preds)

# Calculate classical cuts for comparison
sub_lists = []
for i in range(num_donors + num_receivers + 1):
    temp = [list(x) for x in combinations(B.nodes(), i)]
    sub_lists.extend(temp)

cut_classic = []
for sub_list in sub_lists:
    cut_classic.append(nx.algorithms.cuts.cut_size(B, sub_list, weight='weight'))

cut_quantum = []
for cut in quantum_preds:
    cut_quantum.append(nx.algorithms.cuts.cut_size(B, cut, weight='weight'))
    print(cut_quantum)

print("Quantum mean cut:", np.max(cut_quantum))
print("Max classical cut:", np.max(cut_classic))
print("Ratio:", np.max(cut_quantum) / np.max(cut_classic))