# imports
import numpy as np
import matplotlib.pyplot as plt
from circleNotationClass import QubitSystem
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister


# Function to get statevector from Aer simulator
def statevector_from_aer(circ: QuantumCircuit) -> np.ndarray:
    backend = Aer.get_backend("aer_simulator_statevector")
    tqc = transpile(circ, backend)
    result = backend.run(tqc).result()
    return np.asarray(result.get_statevector(tqc), dtype=np.complex128)

DTYPE = np.complex128

## Declare class for Shor Algorithm circuit and subroutine functions ##

class ShorCircuit:

    # Initialisation of the quantum circuit
    def __init__(self, a: int, working_bits: int, precision_bits: int):
        """
        Creates an instance of the ShorCircuit class, which is a quantum circuit. The instance is initialised with the base a, 
        the number of working registers, and the number of auxiliary registers. 
        
        """
        # Parameters
        self.base = a
        self.working_bits = working_bits
        self.precision_bits = precision_bits

        ## Circuit initialisation ##
        # Working Register: Quantum + Classical
        self.working_register = QuantumCircuit(
            QuantumRegister(working_bits, name='W'), 
            ClassicalRegister(working_bits, name='Readout'), 
            name="Working Register")
        
        # Precision Register: Quantum only
        self.precision_register = QuantumCircuit(
            QuantumRegister(precision_bits, name='P'), 
            name="Precision Register")
        
        # Full Circuit: Working + Precision
        self.full_circuit = self.precision_register.tensor(self.working_register)
        self.full_circuit.barrier()
        pass
        
    ## SUBROUTINE FUNCTIONS ON QUANTUM CIRCUITS ##

    # Controlled Multiplication by 2
    def controlled_multiplication_by_2(self, control_bit: int):
        for idx in range(self.working_bits-1):
        
            self.full_circuit.cswap(
                control_qubit=self.working_bits+control_bit, 
                target_qubit1=self.working_bits - idx- 1, 
                target_qubit2=self.working_bits - idx - 2
                )
        return 


    # Modular Exponentiation
    def modular_exponentiation(self):
        
        # Step 1: Put precision register in full binary state
        [self.precision_register.h(idx) for idx in range(self.precision_bits)]

        # Step 2: Put LSB bit of working register in state |1>
        self.working_register.x(0)

        # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
        for idx in range(self.precision_bits):
            for exp in range(idx+1):
                self.controlled_multiplication_by_2(control_bit=idx)
        pass

    # QFT
    def qft(self):
        pass    

    # Inverse QFT
    def inverse_qft(self):
        pass


    ## PLOTTING FUNCTIONS ##
    # Draw the full circuit    
    def circuit_draw(self):
        self.full_circuit.draw("mpl", initial_state=True)
        plt.show()
        pass

f = ShorCircuit(2, 4, 4)
f.modular_exponentiation()
f.circuit_draw()



