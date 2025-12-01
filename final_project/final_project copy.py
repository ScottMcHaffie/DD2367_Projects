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
        self.B_bits = self.working_bits
        self.carry_bits = 2
        self.N_bits = self.working_bits
        
        self.total_bits = self.working_bits*3 + self.precision_bits + self.carry_bits + 1
        
        ## Circuit initialisation ##
        # Working Register: Quantum + Classical
        self.working_register = QuantumCircuit(
            QuantumRegister(self.working_bits, name='W'), 
            name="Working Register")
        
        # Precision Register: Quantum only
        self.precision_register = QuantumCircuit(
            QuantumRegister(self.precision_bits, name='P'),
            # ClassicalRegister(self.precision_bits, name='Readout'), 
            name="Precision Register")

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

        self.full_circuit = QuantumCircuit(name='Full Circuit')
        self.full_circuit = self.flag_register                     # 0
        # self.full_circuit = self.full_circuit.tensor(self.flag_register)
        self.full_circuit = self.full_circuit.tensor(self.N_register)
        self.full_circuit = self.full_circuit.tensor(self.c_out_register)
        self.full_circuit = self.full_circuit.tensor(self.B_register)
        self.full_circuit = self.full_circuit.tensor(self.precision_register)
        self.full_circuit = self.full_circuit.tensor(self.working_register)
        self.full_circuit = self.full_circuit.tensor(self.c_in_register)
        # self.full_circuit.barrier()

        # Now update the index mapping:
        # Set register indices for passing to gates
        self.c_in_index = 0
        self.working_indices = list(1 + np.arange(self.working_bits)) 
        self.precision_indices = list(1 + self.working_bits + np.arange(self.precision_bits)) 
        # self.c_in_index = self.working_bits + self.precision_bits 
        self.b_indices = list(self.precision_indices[-1] + 1 + np.arange(self.B_bits)) 
        self.c_out_index = self.precision_indices[-1] + 1 + self.B_bits 
        self.n_indices = list(self.c_out_index + 1 + np.arange(self.N_bits)) 
        self.flag_bit = self.c_out_index + 1 + self.N_bits 
        pass


    ## ------ ADDER SUBROUTINES --------- ##

    # N_register set to |N>
    def set_N_register(self, controlled_reset=False):

        binary_str = bin(self.N)[2:]
        powers = [len(binary_str) - 1 - i for i, bit in enumerate(binary_str) if bit == '1']
        powers_array = np.array(powers)

        if controlled_reset:
            base_index = self.working_indices[0]
            target_indices = list(base_index + powers_array)
            self.full_circuit.cx(control_qubit=self.flag_bit, target_qubit=target_indices)
        else:
            base_index = self.c_out_index + 1
            target_indices = list(base_index + powers_array)
            self.full_circuit.x(target_indices)
        pass

    # Adder modulo N
    def add_mod_N(self):
        '''
        Computes (a + b) mod N.
        Input: |a> in working_register, |b> in B_register
        Output: |a> in working_register, |(a+b) mod N> in B_register
        '''
        self.set_N_register()
        # Step 1: ADDER : a, b --> a, a + b
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits),
            [self.c_in_index] + 
            self.working_indices + 
            self.b_indices + 
            [self.c_out_index]
            )
        
        # Step 2: Bitwise SWAP a <--> N
        for idx in range(self.working_bits):
            self.full_circuit.swap(
                qubit1 = self.working_indices[idx],
                qubit2 = idx + self.c_out_index + 1
            )
    
        # Step 3: SUBTRACTER : N, a+b --> N, a+b-N
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits).inverse(),
            [self.c_in_index] + 
            self.working_indices + 
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

        # Step 6: ADDER : N, a+b-N --> N, a+b
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits),
            [self.c_in_index] + 
            self.working_indices + 
            self.b_indices + 
            [self.c_out_index]
            )

        # Step 7: Controlled RESET : Control: Flag, Target : Register a
        self.set_N_register(controlled_reset=True)
        
        # self.full_circuit.barrier()

        # Step 8: Bitwise SWAP a <--> N
        for idx in range(self.working_bits):
            self.full_circuit.swap(
                qubit1 = self.working_indices[idx],
                qubit2 = idx + self.c_out_index + 1
            )
        # self.full_circuit.barrier()

        # Step 9: SUBTRACTER : a, b --> a, b-a
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits).inverse(),
            [self.c_in_index] + 
            self.working_indices + 
            self.b_indices + 
            [self.c_out_index]
            )
        
        # Step 10: Controlled X: control bit = MSB of b, target bit = flag 
        self.full_circuit.cx(
            control_qubit=self.b_indices[-1],
            target_qubit=self.flag_bit
        )

        # Step 11: ADDER : a, b-a --> a, b
        self.full_circuit.append(
            FullAdderGate(num_state_qubits=self.N_bits),
            [self.c_in_index] + 
            self.working_indices + 
            self.b_indices + 
            [self.c_out_index]
            )
        

        

    # Controlled Multiplication by 2
    def controlled_multiplication_by_2(self, control_bit: int):
        for idx in range(self.working_bits-1):

            # Controlled SWAP between qubits in working register, starting from MSB <--> MSB - 1 ... down to LSB <--> LSB + 1
            self.full_circuit.cswap(
                control_qubit=self.working_bits+control_bit, 
                target_qubit1=self.working_bits - idx - 0, 
                target_qubit2=self.working_bits - idx - 1
                )
        return 
    
    ## ---------------- Helper Methods to Create Reusable Gates ------------- ##

    # QFT module gate
    def _create_qft_gate(self,inv: bool = False) -> 'Gate': #type: ignore
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

    # General Controlled Operation (Increment/Decrement) based on binary representation
    def controlled_operation(self, number, operation_type):

        if not isinstance(number, int) or number < 0:
            raise ValueError("Input 'number' must be a non-negative integer.")

        # Determine the correct method and order of operations
        if operation_type == 'increment':
            target_method = self.cont_inc_pow_2
            reverse_order = True
        elif operation_type == 'decrement':
            target_method = self.cont_dec_pow_2
            reverse_order = False
        else:
            raise ValueError("operation_type must be 'increment' or 'decrement'")

        # Get the binary representation of the number
        binary_str = bin(number)[2:]

        # Find the powers of two corresponding to the '1's in the binary string
        powers = [
            len(binary_str) - 1 - i
            for i, bit in enumerate(binary_str)
            if bit == '1'
        ]
        
        if reverse_order:
            powers.reverse()

        for p in powers:
            target_method(power_of_two=p)

    # >= N check, k times
    def controlled_geq_check(self, k):
        # Step 1: X-gate to Modulo Bit
        self.full_circuit.x(self.modulo_bit)

        # Step 2: Repeated controlled decrement and increments
        for i in range(k):
            # Decrement
            self.controlled_operation(self, self.N, 'decrement')
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
            self.controlled_operation(self, self.N, 'increment')
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
        [self.full_circuit.h(self.working_bits + idx + 1) for idx in range(self.precision_bits)]

        # Step 2: Put LSB bit of working register in state |1>
        self.full_circuit.x(1)
        # Step 2.1: Barrier
        self.full_circuit.barrier(label='ME Init')

        # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
        for idx in range(self.precision_bits):
            for exp in range(2 ** idx):
                self.controlled_multiplication_by_2(control_bit=idx+1)

            # Draw barrier after each modular exponentiation stage
            self.full_circuit.barrier(label='ME 2^' + str(2 ** idx))
        pass


    # Modular Exponentiation with modulo checks
    def modular_exponentiation(self):
        # Dictionary to store the specific modulo parameters that differ for each N
        modulo_params = {
            21: {2: 7, 3: 12},
            33: {2: 4, 3: 8},
            35: {2: 4, 3: 8}
            # N=15 has no special checks, so it is not included
        }
        # Step 1: Put precision register in superposition
        [self.full_circuit.h(self.working_bits + idx) for idx in range(self.precision_bits)]

        # Step 2: Set working register to state |1>
        self.full_circuit.x(0)
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

            # Barrier for visualization after each stage
            self.full_circuit.barrier(label=f'ME Stage idx={idx}')
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
    def shor_draw(self, scale = 0.5, s_fold=25):
        self.full_circuit.draw("mpl", initial_state=True, scale = scale, fold=s_fold)
        plt.show()
        pass

    # Circle Notation of final Statevector
    def shor_circle_viz(self, output = 'binary', cols=16):
        state_vec = Statevector(self.full_circuit)
        
        if output == 'binary':
            QubitSystem(statevector=state_vec, label="Final state").viz_circle_with_mag(
                max_cols=cols
            )

        if output == 'tp':
            QubitSystem(statevector=state_vec, label="Final state").viz_circle_with_mag(
                max_cols=cols,
                working_bits=self.working_bits,
                precision_bits=self.precision_bits,
                Cin_bits=1,
                B_bits = self.B_bits,
                Cout_bits = 1,
                N_bits =self.N_bits,
                Flag_bits = 1
                )           
        pass
    
    ## Measurement and Classical Part ##
    
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
        result = simulator.run(circ, shots=2_048).result()
        counts = result.get_counts(circ)
        counts_int_states = {int(k, 2): v for k, v in counts.items()}

        # store in class object and return the list of counds based on state
        self.counts_int_states = counts_int_states
        return counts_int_states
    
    # Estimate repeat period from spike
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
                candidates.append(repeat_period)
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


    # # Controlled Multiplication by 2
    # def controlled_multiplication_by_2(self, control_bit: int):
    #     for idx in range(self.working_bits-1):

    #         # Controlled SWAP between qubits in working register, starting from MSB <--> MSB - 1 ... down to LSB <--> LSB + 1
    #         self.mod_exp_circ.cswap(
    #             control_qubit=self.working_bits+control_bit, 
    #             target_qubit1=self.working_bits - idx - 1, 
    #             target_qubit2=self.working_bits - idx - 2
    #             )
    #     all_qubits = range(self.working_bits + self.precision_bits)
    #     return self.full_circuit.append(self.mod_exp_circ.to_gate(),all_qubits)
    
    # # Modular Exponentiation
    # def exponentiation(self):
    #     n_p = self.precision_bits
    #     n_w = self.working_bits
        
    #     self.mod_exp_circ = QuantumCircuit(n_p + n_w, name="ModExp")
    #     # Step 1: Put precision register in full binary state
    #     [self.mod_exp_circ.h(self.working_bits + idx) for idx in range(self.precision_bits)]

    #     # Step 2: Put LSB bit of working register in state |1>
    #     self.mod_exp_circ.x(0)
    #     # Step 2.1: Barrier
    #     # self.mod_exp_circ.barrier(label='ME Init')

    #     # Step 3: Multiply by 2^x conditioned on precision bit state |x>: 
    #     for idx in range(self.precision_bits):
    #         for exp in range(2 ** idx):
    #             self.controlled_multiplication_by_2(control_bit=idx)

    #         # Draw barrier after each modular exponentiation stage
    #         # self.mod_exp_circ.barrier(label='ME 2^' + str(2 ** idx))
    #     pass


## Testing ##
a = 2; N = 15
f = ShorCircuit(a, N, 7, 3)
# f.exponentiation()
f_new = ShorCircuit(a, N, 7, 3)
# # f_new.exponentiation()

f_new_new = QuantumCircuit(3)
# # f_new_new.exponentiation
# # f.shor_qft()

# # f.exponentiation()

# # exponentiation_gate = f.create_exponentiation_gate()
# # f.full_circuit.append()
# # f.set_N_register()
# f.add_mod_N()
f_new.full_circuit.append(f_new_new.to_gate(label="Exp"), range(8, 11))
f_new.full_circuit.barrier(label="$a^x$")
f_new.full_circuit.append(f.full_circuit.to_gate(label="Add_modN"), range(0, f.total_bits))
f_new.full_circuit.barrier(label="(W + B)Mod(N)")

# f.shor_qft()
# f.shor_precision_measure()
# f.shor_run_qc()
# f.shor_factors()

# f.set_N_register()
# f.add_mod_N()
#f.set_N_register()
#f.shor_qft()

# f.shor_draw(scale=0.7, s_fold=100)
f_new.full_circuit.draw("mpl", initial_state=True, scale = 1, fold=100, filename=r"C:\Users\T480s\OneDrive\Documents\aKTH\KTH Second Year\DD2367\final_circuit_7W.jpg")

# f.shor_circle_viz(cols=1, output='tp')



#%%

#f.shor_precision_measure()
#f.shor_run_qc()
#f.shor_factors()