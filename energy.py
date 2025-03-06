import numpy as np
from qiskit import Aer
from qiskit.algorithms import VQE, NumPyEigensolver
from qiskit.algorithms.optimizers import ADAM
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.circuit.library import HartreeFock
import matplotlib.pyplot as plt

# Define the BeH2 molecule
molecule = Molecule(
    geometry=[
        ["Be", [0.0, 0.0, 0.0]],
        ["H", [0.0, 0.0, -1.3]],
        ["H", [0.0, 0.0, 1.3]]
    ],
    charge=0,
    multiplicity=1
)

# Create driver and electronic structure problem
driver = ElectronicStructureMoleculeDriver(molecule, basis='sto-3g')
es_problem = ElectronicStructureProblem(driver)

# Define the active space (4 electrons in 6 orbitals as mentioned in the paper)
# We exclude the 1s orbital of Be atom as mentioned in the paper
active_space = [1, 2, 3, 4, 5, 6]  # Orbitals to include in active space
es_problem.active_space = (4, active_space)  # 4 electrons in selected orbitals

# Get the second quantized operators
second_q_ops = es_problem.second_q_ops()
main_op = second_q_ops[0]

# Define the mapper and converter
mapper = ParityMapper()
converter = QubitConverter(mapper, two_qubit_reduction=True)  # Using parity mapping with reduction

# Get the qubit operator
qubit_op = converter.convert(main_op)

# Calculate the exact ground state energy for reference
numpy_solver = NumPyEigensolver(k=1)
result = numpy_solver.compute_eigenvalues(qubit_op)
exact_energy = result.eigenvalues[0].real
print(f"Exact BeH2 ground state energy: {exact_energy:.6f} Hartrees")

# Create initial state (Hartree-Fock)
init_state = HartreeFock(
    es_problem.num_spatial_orbitals, 
    es_problem.num_particles,
    converter
)

# Implementation of Adaptive VQE
class AdaptiveVQE:
    def __init__(self, qubit_op, initial_state, threshold=0.05):
        self.qubit_op = qubit_op
        self.initial_state = initial_state
        self.threshold = threshold
        self.ansatz = None
        self.parameters = []
        self.parameter_values = []
        self.energies = []
        
    def _generate_excitation_pool(self):
        """Generate pool of single and double excitation operators"""
        excitation_pool = []
        n_qubits = self.qubit_op.num_qubits
        
        # Generate single excitations
        for i in range(n_qubits):
            param = Parameter(f"s_{i}")
            circuit = QuantumCircuit(n_qubits)
            circuit.rx(param, i)
            excitation_pool.append((circuit, param, f"Single_{i}", "single"))
        
        # Generate double excitations
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param = Parameter(f"d_{i}_{j}")
                circuit = QuantumCircuit(n_qubits)
                circuit.cx(i, j)
             
