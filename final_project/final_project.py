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
        self.flag_bit = self.precision_bits + self.working_bits

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
        
        # Flag Register: Quantum only
        self.flag_register = QuantumCircuit(
            QuantumRegister(1, name='F'), 
            name="Flag Register"
        )
        # Full Circuit: Working + Precision
        self.full_circuit = self.flag_register.tensor(self.precision_register)
        self.full_circuit = self.full_circuit.tensor(self.working_register)
        self.full_circuit.barrier()
        pass
        
    ## ---- SUBROUTINE FUNCTIONS ON QUANTUM CIRCUITS ---- ##

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
    
    ## Increment/Decrement Operators ##

    # Increment by 2^M :

    def inc_pow_2(self, power_of_two):
        for idx in range(self.working_bits - power_of_two):
            if idx == self.working_bits - power_of_two - 1:
                self.full_circuit.x(power_of_two)
                break

            control_bits = []
            for jdx in range(self.working_bits - power_of_two - idx - 1):
                control_bits.append(jdx + power_of_two)
            self.full_circuit.mcx(control_qubits=control_bits, target_qubit=self.working_bits - idx - 1)
        pass

    # Controlled Increment by 2^M    
    def cont_inc_pow_2(self, power_of_two): 
        for idx in range(self.working_bits - power_of_two):
            # For last bit
            if idx == self.working_bits - power_of_two - 1:
                self.full_circuit.cx(
                    control_qubit=self.flag_bit, 
                    target_qubit=power_of_two
                    )
                break
            
            # For other bits
            control_bits = []
            for jdx in range(self.working_bits - power_of_two - idx - 1):
                control_bits.append(jdx + power_of_two)
            control_bits.append(self.flag_bit)
            self.full_circuit.mcx(control_qubits=control_bits, target_qubit=self.working_bits - idx - 1)
        pass

    # Decrement by 2^M :
    def dec_pow_2(self, power_of_two):
        for idx in range(self.working_bits - power_of_two):
            if idx == 0:
                self.full_circuit.x(power_of_two)
            else:
                control_bits = []
                for jdx in range(idx):
                    control_bits.append(power_of_two + jdx)

                self.full_circuit.mcx(
                    control_qubits=control_bits, 
                    target_qubit=power_of_two + idx
                    )
        pass

    # Decrement 15 :
    def dec15(self):
        self.dec_pow_2(power_of_two=3)
        self.full_circuit.barrier(label=f'- 2^{3}')
        self.dec_pow_2(power_of_two=2)
        self.full_circuit.barrier(label=f'- 2^{2}')
        self.dec_pow_2(power_of_two=1)
        self.full_circuit.barrier(label=f'- 2^{1}')
        self.dec_pow_2(power_of_two=0)
        self.full_circuit.barrier(label=f'- 2^{0}')
        pass
    
    # Controlled Increment 15:
    def cont_inc_15(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.full_circuit.barrier(label=f'+ 2^{0}')
        self.cont_inc_pow_2(power_of_two=1)
        self.full_circuit.barrier(label=f'+ 2^{1}')
        self.cont_inc_pow_2(power_of_two=2)
        self.full_circuit.barrier(label=f'+ 2^{2}')
        self.cont_inc_pow_2(power_of_two=3)
        self.full_circuit.barrier(label=f'+ 2^{3}')
        pass

    # Decrement 21 : 
    def dec21(self):
        self.dec_pow_2(power_of_two=4)
        self.dec_pow_2(power_of_two=2)
        self.dec_pow_2(power_of_two=0)
        pass

    # Controlled Increment 21:
    def cont_inc_21(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=2)
        self.cont_inc_pow_2(power_of_two=4)
        pass
        
    # Decrement 33 :
    def dec33(self):
        self.dec_pow_2(power_of_two=5)
        self.dec_pow_2(power_of_two=0)
        pass

    # Controlled Increment 33:
    def cont_inc_33(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=5)
        pass

    # Decrement 35 : 
    def dec35(self):
        self.dec_pow_2(power_of_two=5)
        self.dec_pow_2(power_of_two=1)
        self.dec_pow_2(power_of_two=0)
        pass

    # Controlled Increment 35:
    def cont_inc_35(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=1)
        self.cont_inc_pow_2(power_of_two=5)
        pass

    # >= N check, k times
    def controlled_geq_check(self, decrement_value, k):
        for i in range(k):
            if decrement_value == 'dec15':
                self.dec15()
                self.full_circuit.cx(
                    control_qubit=self.working_bits- 1, 
                    target_qubit=self.flag_bit
                    )
                self.full_circuit.barrier(label="C-NOT Flag")
                self.cont_inc_15()

            elif decrement_value == 'dec21':
                self.dec21()
                self.full_circuit.cx(
                    control_qubit=self.working_bits-1, 
                    target_qubit=self.flag_bit
                    )
                self.cont_inc_21()

            elif decrement_value == 'dec33':
                self.dec33()
                self.full_circuit.cx(
                    control_qubit=self.working_bits-1, 
                    target_qubit=self.flag_bit
                    )
                self.cont_inc_33()
            
            elif decrement_value == 'dec35':
                self.dec35()
                self.full_circuit.cx(
                    control_qubit=self.working_bits-1, 
                    target_qubit=self.flag_bit
                    )
                self.cont_inc_35()

        # Reverse flag qubit back to initial state
        self.full_circuit.x(self.working_bits - 1)
        self.full_circuit.cx(
            control_qubit=self.working_bits - 1,
            target_qubit=self.flag_bit
        )        
        self.full_circuit.x(self.working_bits - 1)

        # Barrier
        self.full_circuit.barrier(label='Reverse flag qubit')
        pass


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
f = ShorCircuit(2, 7, 0)
f.inc_pow_2(6)
f.full_circuit.barrier(label='init')

f.controlled_geq_check('dec35', 1)

#f.shor_draw(scale=0.7)
#f.shor_circle_viz()
print(np.real(np.round(Statevector(f.full_circuit), 0)))


