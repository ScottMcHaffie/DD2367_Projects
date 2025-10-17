# imports

#%%
import matplotlib
# matplotlib.use('module://matplotlib_inline.backend_inline')
import numpy as np
import matplotlib.pyplot as plt
import circleNotationClass
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister
import importlib
importlib.reload(circleNotationClass)
from circleNotationClass import QubitSystem
from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator
from qiskit.circuit.library import FullAdderGate
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from math import gcd

def statevector_from_aer(circ: QuantumCircuit) -> np.ndarray:
    backend = Aer.get_backend("aer_simulator_statevector")
    tqc = transpile(circ, backend)
    result = backend.run(tqc).result()
    return np.asarray(result.get_statevector(tqc), dtype=np.complex128)

class ShorCircuit:

    # Initialisation of the quantum circuit
    def __init__(self, a: int, N: int, working_bits: int, precision_bits: int):
        """
        Creates an instance of the ShorCircuit class, which is a quantum circuit. The instance is initialised with the base a, 
        the number of working registers, and the number of auxiliary registers. 
        
        """
        # Parameters
        self.base = a
        self.N = N
        self.working_bits = working_bits
        self.precision_bits = precision_bits
        self.dummy_A_bits =self.working_bits
        self.B_bits = self.working_bits
        self.carry_bits = 2
        self.N_bits = self.working_bits
        # Position of Top Bit = Total number of bits - 1
        self.flag_bit = self.N_bits + self.carry_bits + self.B_bits + self.dummy_A_bits + self.precision_bits + self.working_bits


        ## Circuit initialisation ##
        # Working Register: Quantum + Classical
        self.working_register = QuantumCircuit(
            QuantumRegister(self.working_bits, name='W'), 
            name="Working Register")
        
        # Precision Register: Quantum only
        self.precision_register = QuantumCircuit(
            QuantumRegister(self.precision_bits, name='P'),
            ClassicalRegister(self.precision_bits, name='Readout'), 
            name="Precision Register")

        # Dummy register for multiplication: Quantum Only
        self.dummy_A_register = QuantumCircuit(
            QuantumRegister(self.working_bits, name='DA'),
            name="Dummy A Register"
            )

        # C_in register
        self.c_in_register = QuantumCircuit(
            QuantumRegister(1, name='C_in'), 
            name="Carry-In Register"
        )

        # B Register: Quantum Only
        self.B_register = QuantumCircuit(
            QuantumRegister(self.B_bits, name='B'), 
            name="B Register"
        )
        
        # C_out register
        self.c_out_register = QuantumCircuit(
            QuantumRegister(1, name='C_out'), 
            name="Carry-Out Register"
        )

        # N Register: Quantum only
        self.N_register = QuantumCircuit(
            QuantumRegister(self.N_bits, name='N'), 
            name="N Register"
        )



        # Flag Register: Quantum Only
        self.flag_register = QuantumCircuit(
            QuantumRegister(1, name='T'), 
            name="Flag Register"
        )


        # Full Circuit: Working + Precision
        self.full_circuit = QuantumCircuit(name='Full Circuit')
        self.full_circuit = self.flag_register.tensor(self.N_register)
        self.full_circuit = self.full_circuit.tensor(self.c_out_register)
        self.full_circuit = self.full_circuit.tensor(self.B_register)
        self.full_circuit = self.full_circuit.tensor(self.c_in_register)
        self.full_circuit = self.full_circuit.tensor(self.dummy_A_register)
        self.full_circuit = self.full_circuit.tensor(self.precision_register)
        self.full_circuit = self.full_circuit.tensor(self.working_register)
        self.full_circuit.barrier()
        pass

        # Set register indices for passing to full adder function
        self.working_indices = list(np.arange(self.working_bits))

        self.dummy_A_indices = list(self.working_bits +
                                     self.precision_bits + 
                                     np.array(self.working_indices)
                                     )
        
        self.c_in_index = self.working_bits + self.precision_bits + self.dummy_A_bits
        self.b_indices = list(self.c_in_index + 1 +  np.arange(self.B_bits))
        self.c_out_index = + self.c_in_index + 1 + self.B_bits

    ## ------ ADDER SUBROUTINES --------- ##

    # N_register set to |N>
    def set_N_register(self, controlled_reset = False):
        if self.N == 15 and controlled_reset == False:
            indices = self.c_out_index + 1 + np.arange(4)
            self.full_circuit.x(list(indices))
        
        if self.N == 15 and controlled_reset == True:
            indices = self.dummy_A_indices[0] + np.arange(4)
            self.full_circuit.cx(
                control_qubit=self.flag_bit,
                target_qubit=list(indices)
                )

        if self.N == 21 and controlled_reset == False:
            indices = self.c_out_index + 1 + np.array([0, 2, 4])
            self.full_circuit.x(list(indices))

        if self.N == 21 and controlled_reset == True:
            indices = self.dummy_A_indices[0] + np.array([0, 2, 4])
            self.full_circuit.cx(
                control_qubit=self.flag_bit,
                target_qubit=list(indices)
                )

        if self.N == 33 and controlled_reset == False:
            indices = self.c_out_index + 1 + np.array([0, 5])
            self.full_circuit.x(list(indices))

        if self.N == 33 and controlled_reset == True:
            indices = self.dummy_A_indices[0] + np.array([0, 5])
            self.full_circuit.cx(
                control_qubit=self.flag_bit,
                target_qubit=list(indices)
                )

        if self.N == 35 and controlled_reset == False:
            indices = self.c_out_index + 1 + np.array([0, 1, 5])
            self.full_circuit.x(list(indices))
        
        if self.N == 35 and controlled_reset == True:
            indices = self.dummy_A_indices[0] + np.array([0, 1, 5])
            self.full_circuit.cx(
                control_qubit=self.flag_bit,
                target_qubit=list(indices)
                )
        pass

    # Adder modulo N

    def add_mod_N(self):
        '''
        a = working 
        b = 0 
        N = number to be factorized
        '''

        # Step 1: ADDER : a, b --> a, a + b : a = dummy, b = B
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits),
            [self.c_in_index] + 
            self.dummy_A_indices + 
            self.b_indices + 
            [self.c_out_index]
            )
        
        # Step 2: Bitwise SWAP a <--> N
        for idx in range(self.dummy_A_bits):
            self.full_circuit.swap(
                qubit1 = self.dummy_A_indices[idx],
                qubit2 = idx + self.c_out_index + 1
            )
    
        # Step 3: SUBTRACTER : a, b --> a, b - a: a = N, b = a + b
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits).inverse(),
            [self.c_in_index] + 
            self.dummy_A_indices + 
            self.b_indices + 
            [self.c_out_index]
            )
        
        # Step 4: Conditional Flag : Control: MSB of b, Target: Flag
        self.full_circuit.x(self.b_indices[-1])
        self.full_circuit.cx(
            control_qubit=self.b_indices[-1],
            target_qubit=self.flag_bit
        )
        self.full_circuit.x(self.b_indices[-1])

        # Step 5: Controlled RESET : Control: Flag, Target : Register a
        self.set_N_register(controlled_reset=True)

        # Step 6: ADDER : a, b --> a, a + b : a = 0 (or) N, b = a + b - N
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits),
            [self.c_in_index] + 
            self.dummy_A_indices + 
            self.b_indices + 
            [self.c_out_index]
            )

        # Step 7: Controlled RESET : Control: Flag, Target : Register a
        self.set_N_register(controlled_reset=True)

        # Step 8: Bitwise SWAP a <--> N
        for idx in range(self.dummy_A_bits):
            self.full_circuit.swap(
                qubit1 = self.dummy_A_indices[idx],
                qubit2 = idx + self.c_out_index + 1
            )

        # Step 9: SUBTRACTER : a, b --> a, a + b : a = 0 (or) N, b = a + b - N (or) a + b 
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits).inverse(),
            [self.c_in_index] + 
            self.dummy_A_indices + 
            self.b_indices + 
            [self.c_out_index]
            )
        
        # Step 10: Controlled X: control bit = MSB of b, target bit = flag 
        self.full_circuit.cx(
            control_qubit=self.b_indices[-1],
            target_qubit=self.flag_bit
        )

        # Step 11: ADDER : a, b --> a, a + b : a = a, b = a + b mod N (a + b or a + b - N)
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits),
            [self.c_in_index] + 
            self.dummy_A_indices + 
            self.b_indices + 
            [self.c_out_index]
            )

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
    
    # Controlled Working --> Dummy RESET
    def cont_X_reset(self, pow_2_idx):
        self.full_circuit.cx(
            control_qubit=pow_2_idx,
            target_qubit=self.dummy_A_indices[0] + pow_2_idx
        )
        pass

    # Multiplication mod N
    def multiplication_mod_N(self):
        # Step 1: RESET Dummy to 2^0
        self.cont_X_reset(pow_2_idx=0)

        # Step 2: ADDER 
        self.add_mod_N()
        self.full_circuit.barrier()

        # Step 3: Loop over remaining x-bit values
        for idx in range(self.working_bits-1):
            self.cont_X_reset(pow_2_idx = idx)
            self.cont_X_reset(pow_2_idx = idx+1)    
            self.add_mod_N()
            self.full_circuit.barrier()
                
        # Step 4: final RESET MSB of dummy register
        self.cont_X_reset(pow_2_idx=self.working_bits-1)

    ## ---------------- Helper Methods to Create Reusable Gates ------------- ##

    
    # QFT module gate

    def _create_qft_gate(self,inv: bool = False) -> 'Gate':

        n = self.precision_bits
        qft_circ = QuantumCircuit(n, name="QFT")

        # --- QFT Logic (Hadamards and Controlled Rotations) ---
        # Based on your original implementation
        for idx in range(n):
            target_h = n - 1 - idx
            qft_circ.h(target_h)
            
            for jdx in range(n - 1 - idx):
                if inv:
                    c_phase_angle = np.pi / (2 ** (jdx + 1))
                else:
                    c_phase_angle = -np.pi / (2 ** (jdx + 1))
                control_cp = n - 1 - idx
                target_cp = n - 2 - idx - jdx
                qft_circ.cp(c_phase_angle, control_cp, target_cp)
        
        # --- SWAP Gates ---
        for idx in range(n // 2):
            qft_circ.swap(idx, n - 1 - idx)
        return qft_circ.to_gate(label="QFT")

    ## -------------- Subroutines Applied to the Main Circuit --------------- ##
    
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

    # Flag Controlled Increment by 2^M
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

    # Modulo Controlled Decrement by 2^M
    def cont_dec_pow_2(self, power_of_two):
        for idx in range(self.working_bits - power_of_two):
            if idx == 0:
                self.full_circuit.cx(
                    control_qubit=self.modulo_bit,
                    target_qubit=power_of_two
                    )
            else:
                control_bits = []
                for jdx in range(idx):
                    control_bits.append(power_of_two + jdx)
                control_bits.append(self.modulo_bit)

                self.full_circuit.mcx(
                    control_qubits=control_bits, 
                    target_qubit=power_of_two + idx
                    )
        pass

    # Modulo controlled Decrement 15 : 
    def cont_dec15(self):
        self.cont_dec_pow_2(power_of_two=3)
        self.cont_dec_pow_2(power_of_two=2)
        self.cont_dec_pow_2(power_of_two=1)
        self.cont_dec_pow_2(power_of_two=0)
        pass
    
    # Flag controlled Increment 15:
    def cont_inc15(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=1)
        self.cont_inc_pow_2(power_of_two=2)
        self.cont_inc_pow_2(power_of_two=3)
        pass

    # Modulo controlled Decrement 21 : 
    def cont_dec21(self):
        self.cont_dec_pow_2(power_of_two=4)
        self.cont_dec_pow_2(power_of_two=2)
        self.cont_dec_pow_2(power_of_two=0)
        pass

    # Flag controlled Increment 21:
    def cont_inc21(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=2)
        self.cont_inc_pow_2(power_of_two=4)
        pass
        
    # Modulo controlled Decrement 33 : 
    def cont_dec33(self):
        self.cont_dec_pow_2(power_of_two=5)
        self.cont_dec_pow_2(power_of_two=0)
        pass

    # Flag controlled Increment 33:
    def cont_inc33(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=5)
        pass

    # Modulo controlled Deccrement 35 : 
    def cont_dec35(self):
        self.cont_dec_pow_2(power_of_two=5)
        self.cont_dec_pow_2(power_of_two=1)
        self.cont_dec_pow_2(power_of_two=0)
        pass

    # Flag controlled Increment 35:
    def cont_inc35(self):
        self.cont_inc_pow_2(power_of_two=0)
        self.cont_inc_pow_2(power_of_two=1)
        self.cont_inc_pow_2(power_of_two=5)
        pass

    # >= N check, k times
    def controlled_geq_check(self, k):
        # Step 1: X-gate to Modulo Bit
        self.full_circuit.x(self.modulo_bit)

        # Step 2: Repeated controlled decrement and increments
        for i in range(k):
            # Decrement
            if self.N == 15: self.cont_dec15()
            if self.N == 21: self.cont_dec21()
            if self.N == 33: self.cont_dec33()
            if self.N == 35: self.cont_dec35()
            self.full_circuit.x(self.modulo_bit)

            # -ve check
            self.full_circuit.cx(
                control_qubit=self.working_bits- 1, 
                target_qubit=self.flag_bit
                )
            self.full_circuit.cx(
                control_qubit=self.working_bits- 1, 
                target_qubit=self.modulo_bit
                )
            
            # increment
            if self.N == 15: self.cont_inc15()
            if self.N == 21: self.cont_inc21()
            if self.N == 33: self.cont_inc33()
            if self.N == 35: self.cont_inc35()
            
            # Reverse flag and modulo qubits back to initial state
            self.full_circuit.cx(
                control_qubit=self.modulo_bit,
                target_qubit=self.flag_bit
            )        
            self.full_circuit.x(self.modulo_bit)

        # Barrier
        self.full_circuit.barrier(label='Mod Exp')
        pass
    
    # Modular Exponentiation Block
    def modular_exponentiation_block(self):

        if self.precision_bits == 0 and self.working_bits == 0:
            return
            
        mod_exp_gate = self._create_mod_exp_gate()
        
        # Apply the gate to all qubits in the circuit
        all_qubits = range(self.working_bits + self.precision_bits)
        self.full_circuit.append(mod_exp_gate, all_qubits)
        self.full_circuit.barrier(label='ModExp') # This barrier is fine
        pass

    # Modular Exponentiation
    def exponentiation(self):
    
        # Step 1: Put precision register in full binary state
        [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

        # Step 2: Put LSB bit of working register in state |1>
        self.full_circuit.x(0)
        # Step 2.1: Barrier
        self.full_circuit.barrier(label='ME Init')

        # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
        for idx in range(self.precision_bits):
            for exp in range(2 ** idx):
                self.controlled_multiplication_by_2(control_bit=idx)

            # Draw barrier after each modular exponentiation stage
            self.full_circuit.barrier(label='ME 2^' + str(2 ** idx))
        pass

    def modular_exponentiation(self):
        if self.N == 15:  
            # Step 1: Put precision register in full binary state
            [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

            # Step 2: Put LSB bit of working register in state |1>
            self.full_circuit.x(0)

            # Step 2.1: Barrier
            self.full_circuit.barrier(label='ME Init')

            # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
            for idx in range(self.precision_bits):
                for exp in range(2 ** idx):
                    self.controlled_multiplication_by_2(control_bit=idx)
                    
                # Draw barrier after each modular exponentiation stage
                self.full_circuit.barrier(label='ME 2^' + str(2 ** idx))


                

        if self.N == 21:
            # Step 1: Put precision register in full binary state
            [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

            # Step 2: Put LSB bit of working register in state |1>
            self.full_circuit.x(0)

            # Step 2.1: Barrier
            self.full_circuit.barrier(label='ME Init')

            # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
            for idx in range(self.precision_bits):
                for exp in range(2 ** idx):
                    self.controlled_multiplication_by_2(control_bit=idx)

                # Draw barrier after each modular exponentiation stage
                self.full_circuit.barrier(label='ME 2^' + str(2 ** idx))

                # Implement modulo computation
                if idx == 2: self.controlled_geq_check(k = 7)
                if idx == 3: self.controlled_geq_check(k = 12)
                    
    
        if self.N == 33:
            # Step 1: Put precision register in full binary state
            [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

            # Step 2: Put LSB bit of working register in state |1>
            self.full_circuit.x(0)

            # Step 2.1: Barrier
            self.full_circuit.barrier(label='ME Init')

            # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
            for idx in range(self.precision_bits):
                for exp in range(2 ** idx):
                    self.controlled_multiplication_by_2(control_bit=idx)
                # Draw barrier after each modular exponentiation stage
                self.full_circuit.barrier(label='ME 2^' + str(2 ** idx))

                # Implement modulo computation
                if idx == 2: self.controlled_geq_check(k = 4)
                if idx == 3: self.controlled_geq_check(k = 8)

        if self.N == 35:
            # Step 1: Put precision register in full binary state
            [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

            # Step 2: Put LSB bit of working register in state |1>
            self.full_circuit.x(0)

            # Step 2.1: Barrier
            self.full_circuit.barrier(label='ME Init')

            # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
            for idx in range(self.precision_bits):
                for exp in range(2 ** idx):
                    self.controlled_multiplication_by_2(control_bit=idx)

                # Draw barrier after each modular exponentiation stage
                self.full_circuit.barrier(label='ME 2^' + str(2 ** idx))

                # Implement modulo computation
                if idx == 2: self.controlled_geq_check(k = 4)
                if idx == 3: self.controlled_geq_check(k = 8)
        pass

    # QFT
    def shor_qft(self):

        if self.precision_bits == 0:
            return
            
        # 1. Create the QFT gate from the helper method
        qft_gate = self._create_qft_gate()

        # 2. Define the qubits for the gate (the precision register)
        precision_qubits = range(self.working_bits, self.working_bits + self.precision_bits)
        
        # 3. Append the gate to the circuit
        self.full_circuit.append(qft_gate, precision_qubits)
        self.full_circuit.barrier(label='QFT')
        pass

    # Inverse QFT
    def inverse_shor_qft(self):
        """
        Appends the custom Inverse QFT (IQFT) gate to the precision register.
        The IQFT gate is generated by taking the inverse of the QFT gate.
        """
        if self.precision_bits == 0:
            return
            
        # 1. Create the forward QFT gate first
        qft_gate = self._create_qft_gate()

        # 2. Create the inverse gate from it
        iqft_gate = qft_gate.inverse()
        iqft_gate.label = "IQFT" # Rename the label for clarity
        
        # 3. Define the qubits for the gate
        precision_qubits = range(self.working_bits, self.working_bits + self.precision_bits)

        # 4. Append the gate to the circuit
        self.full_circuit.append(iqft_gate, precision_qubits)
        self.full_circuit.barrier(label='IQFT')
        pass

    # Overall Circuit
    def shor_overall_circuit(self):
        self.modular_exponentiation()
        self.shor_qft()
        pass

    ## PLOTTING FUNCTIONS ##
    # Draw the full circuit    
    def shor_draw(self, scale = 0.5):
        self.full_circuit.draw("mpl", initial_state=True, scale = scale)
        plt.show()
        pass

    # Circle Notation of final Statevector
    def shor_circle_viz(self, output = 'binary', cols=16):
        state_vec = Statevector(self.full_circuit)
        # QubitSystem(statevector=state_vec, label="Final state").viz_circle(max_cols=16)
        
        if output == 'binary':
            QubitSystem(statevector=state_vec, label="Final state").viz_circle_with_mag(
                max_cols=cols
            )

        if output == 'tp':
            QubitSystem(statevector=state_vec, label="Final state").viz_circle_with_mag(
                max_cols=cols,
                working_bits=self.working_bits,
                precision_bits=self.precision_bits,
                Dummy_A_bits=self.dummy_A_bits,
                Cin_bits=1,
                B_bits = self.B_bits,
                Cout_bits = 1,
                N_bits =self.N_bits,
                Flag_bits = 1
                )      
        pass
    
    # measure the precision register
    def shor_precision_measure(self):
        self.full_circuit.barrier(label='Meas')
        self.full_circuit.measure(
            range(self.working_bits, self.working_bits + self.precision_bits),
            range(self.precision_bits)
        )
        pass

    # run the quantum circuit (simulate)
    def shor_run_qc(self):
        # decompose the full circuit
        qc = self.full_circuit.decompose()

        # Transpile for simulator
        simulator = AerSimulator()
        circ = transpile(qc, simulator)

        # Run and get counts
        result = simulator.run(circ).result()
        counts = result.get_counts(circ)
        counts_int_states = {int(k, 2): v for k, v in counts.items()}

        # store in class object and return the list of counds based on state
        self.counts_int_states = counts_int_states
        return counts_int_states

    def shor_estimate_num_spikes(self, spike):
        range_ = self.base**self.precision_bits
        # Mirror the JS behavior for spike < range/2
        if spike < range_ / 2:
            spike = range_ - spike

        best_error = 1.0
        e0, e1, e2 = 0, 0, 0
        actual = spike / range_
        candidates = []

        denom = 1.0
        while denom < spike:
            numerator = round(denom * actual)
            estimated = numerator / denom
            error = abs(estimated - actual)

            e0 = e1
            e1 = e2
            e2 = error

            # Local minimum check (same logic as the JS version)
            if e1 <= best_error and e1 < e0 and e1 < e2:
                repeat_period = denom - 1
                candidates.append(denom - 1)
                best_error = e1

            denom += 1.0

        return candidates
    
    def shor_logic(self, repeat_period):
        """
        Given the repeat period, find the actual factors of N.
        """

        ar2 = self.base ** (repeat_period / 2.0)
        factor1 = gcd(self.N, int(ar2 - 1))
        factor2 = gcd(self.N, int(ar2 + 1))
        return factor1, factor2
    
    def shor_factors(self):
        for state, num_counts in sorted(self.counts_int_states.items(), key=lambda item: item[1], reverse=True):
            possible_periods = self.shor_estimate_num_spikes(state)
            for possible_period in possible_periods:
                factors = self.shor_logic(possible_period)
                if factors[0] != self.N and factors[1] != self.N and factors[0] * factors[1] == self.N: # self.N
                    print (factors)



## Testing ##
a = 2; N = 15
f = ShorCircuit(a, N, 4, 2)
#f.full_circuit.x([4])
#f.full_circuit.x(f.b_indices[4])
# print (f.working_bits)
f.exponentiation()
f.multiplication_mod_N()
#f.set_N_register()
#f.shor_qft()

#f.shor_draw(scale=0.7)
f.shor_circle_viz(cols=1, output='binary')



#%%

#f.shor_precision_measure()
#f.shor_run_qc()
#f.shor_factors()