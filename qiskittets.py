from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

simulator_aer = AerSimulator()

# Create a quantum circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all
# Transpile the circuit for the AerSimulator
simulator = AerSimulator()
transpiled_qc = transpile(qc, backend=simulator_aer)

# Execute the circuit on the simulator
result = simulator_aer.run(transpiled_qc).result()
print(result)
# Get the result counts
#counts = result.get_counts()
#print(counts)
