from qiskit import *
import qiskit.visualization
circuit = QuantumCircuit(1,1)
circuit.x(0)

simulator = aer.get_backend("aer_simulator")
result = execute(circuit, backend = simulator).result()
statevector = result.get_statevector()
print(statevector)
