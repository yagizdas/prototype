import networkx as nx
import numpy as np
import random
from itertools import combinations
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, SolverFactory, Binary, maximize
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2

#service = QiskitRuntimeService(channel="ibm_quantum", token="4d675380adc3c0291ae0f7eebeb5011cf50f5ec7822256ec60f44a95eeae7a62ef3bed681f5b0ec3e8af0b30d7cb77445c71f13b9fbb90dbceb8d205dca56ad1")
#backend = service.backend(name = 'ibm_kyiv')  # Use a real quantum device if needed

# Define a bipartite graph (donors and recipients)
donors = 4
recipients = 4
G = nx.complete_bipartite_graph(donors, recipients)

weighted = True

for (u, v) in G.edges():
    if weighted:
        w = random.uniform(0, 1)
    else:
        w = 1
    G.edges[u, v]['weight'] = w


model = ConcreteModel()

# Decision variables: x[i,j] = 1 if donor i is matched with recipient j
model.x = Var(G.edges(), within=Binary)

# Objective function: maximize the total weight of connections
def objective_rule(model):
    return sum(G.edges[i, j]['weight'] * model.x[i, j] for i, j in G.edges())
model.obj = Objective(rule=objective_rule, sense=maximize)  # Minimize negative, equivalent to maximize

#Constraints: Each donor and recipient is matched only once
def donor_constraint_rule(model, i):
    return sum(model.x[i, j] for j in G.neighbors(i) if (i, j) in model.x) <= 1
model.donor_constraint = Constraint(range(donors), rule=donor_constraint_rule)

def recipient_constraint_rule(model, j):
    return sum(model.x[i, j] for i in G.neighbors(j) if (i, j) in model.x) <= 1
model.recipient_constraint = Constraint(range(donors, donors + recipients), rule=recipient_constraint_rule)

# QAOA parameters
depth = 8
rep = 1000
qubits = list(range(donors + recipients))

# Initialization of the circuit (Hadamard gates)
def initialization(qc, qubits):
    for q in qubits:
        qc.h(q)

# Define the cost unitary
def cost_unitary(qc, qubits, gamma):
    for i, j in G.edges():
        qc.cx(qubits[i], qubits[j])
        qc.rz(2 * gamma * G.edges[i, j]['weight'], qubits[j])
        qc.cx(qubits[i], qubits[j])

# Define the mixer unitary
def mixer_unitary(qc, qubits, beta):
    for q in qubits:
        qc.rx(2 * beta, q)

# Create the QAOA circuit
def create_circuit(params):
    gammas = [j for i, j in enumerate(params) if i % 2 == 0]
    betas = [j for i, j in enumerate(params) if i % 2 == 1]

    qc = QuantumCircuit(donors + recipients)
    initialization(qc, qubits)

    for d in range(depth):
        cost_unitary(qc, qubits, gammas[d])
        mixer_unitary(qc, qubits, betas[d])

    qc.measure_all()
    return qc


# Define the cost function
def cost_function(params):

    qc = create_circuit(params)

    #yeni qiskit notasyonunda devre calistirmak icin boyle yapmak gerekiyor

    pass_manager_circuit = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pass_manager_circuit.run(qc)

    cost_service = Sampler(backend)
    job = cost_service.run([isa_circuit])
    result = job.result()
    counts = result.get_counts()
    print(counts)

    total_cost = 0
    for bitstring, count in counts.items():
        bit_list = [int(bit) for bit in bitstring]
        for i, j in G.edges():
            total_cost += G.edges[i, j]['weight'] * 0.5 * ((1 - 2 * bit_list[i]) * (1 - 2 * bit_list[j]) - 1) * count
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
transpiled_circuit = transpile(qc, backend)
job = backend.run(transpiled_circuit, shots=rep)
result = job.result()
counts = result.get_counts()

# Process the results
quantum_preds = []
for bitstring in counts:
    temp = []
    for pos, bit in enumerate(bitstring):
        if bit == '1':
            temp.append(pos)
    quantum_preds.append(temp)

# Calculate classical cuts for comparison
sub_lists = []
for i in range(donors + recipients + 1):
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
