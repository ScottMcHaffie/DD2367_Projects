# imports

#%%
import matplotlib
# matplotlib.use('module://matplotlib_inline.backend_inline')
import numpy as np
import math
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

        ## Circuit initialisation ##
        # Working Register: Quantum + Classical
        self.working_register = QuantumCircuit(
            QuantumRegister(working_bits, name='W'), 
            name="Working Register")
        
        # Precision Register: Quantum only
        self.precision_register = QuantumCircuit(
            QuantumRegister(precision_bits, name='P'),
            ClassicalRegister(precision_bits, name='Readout'), 
            name="Precision Register")
        

        # Full Circuit: Working + Precision
        self.full_circuit = self.precision_register.tensor(self.working_register)
        self.full_circuit.barrier()
        pass
    
    ## ---------- Modular Exponentiation -------- ##
    def mod_exp_classical(self):
        signal = []
        for idx in range(2 ** self.precision_bits):
            signal.append((self.base ** idx) % self.N)
        return signal
    
    def precision_init(self):
        self.full_circuit.h(list(self.working_bits + np.arange(self.precision_bits)))
        self.full_circuit.barrier(label='Init')

    def mod_exp_init(self):
        '''
        Initialise the working register to |a^x mod N>_w (.X.) |x>_p
        '''
        # Step 1: Get binary representation of a^x mod N
        signal_size = 2 ** self.precision_bits

        # Step 1.1: Create signal array of a^x % N
        signal_array = self.mod_exp_classical()
        
        # Step 1.2: Get binary representation of signal array
        signal_binary = [np.binary_repr(signal_array[idx], width = self.working_bits) for idx in range(signal_size)]
        
        # Step 1.3: Invert each element of array 
        signal_binary = [signal_binary[idx][::-1] for idx in range(signal_size)]
        # Step 2: controlled RESET working register to a^x mod N based on binary representation 
        # || Control: Precision Register
        # || Target: Working register
        
        # Step 2.1: For each x value (cycle over signal array)
        for precision_idx in range(signal_size):

            # Step 2.1.1: Get binary representation of x
            idx_bin = np.binary_repr(num=precision_idx, width=self.precision_bits)
            idx_bin = idx_bin[::-1]

            # Step 2.1.2: Add X gates to precision qubits as per binary representation of x
            for jdx in range(self.precision_bits):
                if idx_bin[jdx] == '0': self.full_circuit.x(self.working_bits + jdx)

            # Step 2.1.3: Add C..CNOT gates from R_p --> R_w based on binary representation of a^x % N
            for jdx in range(self.working_bits):
                if signal_binary[precision_idx][jdx] == '1':
                    self.full_circuit.mcx(
                        control_qubits=list(
                            self.working_bits + np.arange(self.precision_bits)
                            ),
                            target_qubit=jdx
                    )

            # Step 2.1.4: Complete X CX X sandwich
            for jdx in range(self.precision_bits):
                if idx_bin[jdx] == '0': self.full_circuit.x(self.working_bits + jdx)

            # Step 2.2: Add barrier after each RESET
            self.full_circuit.barrier(label=f'${self.base}^{precision_idx}$ mod {self.N}')

    ## ---------- Quantum Fourier Transform -------- ##
    
    # Circuit block for QFT stage
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

    ## ---------- Overall Circuit -------- ##
    def shor_overall_circuit(self):
        self.precision_init()
        self.mod_exp_init()
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
                flag_bit=1
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
        result = simulator.run(circ, shots=2048).result()
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
                candidates.append(repeat_period)
                best_error = e1

            denom += 1.0

        return candidates
    
    def shor_logic(self, repeat_period):
        """
        Given the repeat period, find the actual factors of N.
        """
        ar2 = math.pow(self.base, (repeat_period / 2.0))
        factor1 = gcd(self.N, int(ar2 - 1))
        factor2 = gcd(self.N, int(ar2 + 1))
        return factor1, factor2
    
    def shor_factors(self):
        for state, num_counts in sorted(self.counts_int_states.items(), key=lambda item: item[1], reverse=True):
            possible_periods = self.shor_estimate_num_spikes(state)
            for possible_period in possible_periods:
                factors = self.shor_logic(possible_period)

                if factors[0] != self.N and factors[1] != self.N and factors[0] * factors[1] == self.N: # self.N
                    print('Prime factors of ', self.N, ": ", factors)
                    return 
                

                


## Testing ##
a = 2; N = 81
f = ShorCircuit(a, N, 7, 6)

# Circuit
f.shor_overall_circuit()

# Measurement
f.shor_precision_measure()

# Spike analysis
f.shor_run_qc()

# Get factors
f.shor_factors()

#f.shor_draw(scale=0.7)


#%%