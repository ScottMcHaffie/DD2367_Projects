# imports
import numpy as np
import matplotlib.pyplot as plt
from circleNotationClass import QubitSystem
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister


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

            # Controlled SWAP between qubits in working register, starting from MSB <--> MSB - 1 ... down to LSB <--> LSB + 1
            self.full_circuit.cswap(
                control_qubit=self.working_bits+control_bit, 
                target_qubit1=self.working_bits - idx - 1, 
                target_qubit2=self.working_bits - idx - 2
                )
        return 


    # Modular Exponentiation
    def modular_exponentiation(self):
    
        # Step 1: Put precision register in full binary state
        [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

        # Step 2: Put LSB bit of working register in state |1>
        self.full_circuit.x(0)
        # Step 2.1: Barrier
        self.full_circuit.barrier(label='ME Init')

        # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
        for idx in range(self.precision_bits):
            for exp in range(idx+1):
                self.controlled_multiplication_by_2(control_bit=idx)

            # Draw barrier after each modular exponentiation stage
            self.full_circuit.barrier(label='ME 2^' + str(idx + 1))
        pass

    # QFT
    def shor_qft(self):
        
        # Step 1: For loop over working bits in reverse order
        for idx in range(self.working_bits):
            # Step 2: Apply Hadamard gate to the current bit
            self.full_circuit.h(self.working_bits-idx - 1)

            # Step 3: Apply controlled phase rotations to all lesser significant bits
            if idx == self.working_bits - 1:
                self.full_circuit.barrier(label=f'QFT {idx+1}')
                break

            for jdx in range(self.working_bits - idx - 1):
                # Calculate phase angle
                c_phase_angle = np.pi / (2 ** (jdx + 1))

                # Apply controlled phase rotation
                self.full_circuit.cp(c_phase_angle, self.working_bits-idx - 1, self.working_bits-idx - 2 - jdx)
            
            # Draw barrier after each qft stage
            self.full_circuit.barrier(label=f'QFT {idx+1}')
        pass 

    # Inverse QFT
    def shor_inverse_qft(self):
        i=0
        # Step 1: For loop over working bits in reverse order
        for idx in reversed(range(self.working_bits)):
            i=i+1
            
            # Step 3: Apply controlled phase rotations to all lesser significant bits
            for jdx in reversed(range(self.working_bits - idx - 1)):
                # Calculate phase angle
                c_phase_angle = -np.pi / (2 ** (jdx + 1))

                # Apply controlled phase rotation
                self.full_circuit.cp(c_phase_angle, self.working_bits-idx - 1, self.working_bits-idx - 2 - jdx)
            
            # Step 2: Apply Hadamard gate to the current bit
            self.full_circuit.h(self.working_bits-idx - 1)
            # Draw barrier after each qft stage
            self.full_circuit.barrier(label=f'invQFT {i}')
        pass

    # Overall Circuit
    def shor_overall_circuit(self):
        self.modular_exponentiation()
        self.shor_qft()
        pass

    ## PLOTTING FUNCTIONS ##
    # Draw the full circuit    
    def shor_draw(self, scale):
        self.full_circuit.draw("mpl", initial_state=True, scale = 0.5)
        plt.show()
        pass

    # Circle Notation of final Statevector
    def shor_circle_viz(self):
        state_vec = Statevector(self.full_circuit)
        QubitSystem(statevector=state_vec, label="Final state").viz_circle()
        pass

## Testing ##
f = ShorCircuit(2, 4, 2)
f.shor_qft()
f.shor_inverse_qft()
#f.shor_overall_circuit()
f.shor_draw(scale=0.5)
# f.shor_circle_viz()

