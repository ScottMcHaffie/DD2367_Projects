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

def statevector_from_aer(circ: QuantumCircuit) -> np.ndarray:
    backend = Aer.get_backend("aer_simulator_statevector")
    tqc = transpile(circ, backend)
    result = backend.run(tqc).result()
    return np.asarray(result.get_statevector(tqc), dtype=np.complex128)

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
            name="Working Register")
        
        # Precision Register: Quantum only
        self.precision_register = QuantumCircuit(
            QuantumRegister(precision_bits, name='P'),
            ClassicalRegister(precision_bits, name='Readout'), 
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
    
    ## Helper Methods to Create Reusable Gates ##
    def _create_mod_exp_gate(self) -> 'Gate':
        n_p = self.precision_bits
        n_w = self.working_bits
        
        mod_exp_circ = QuantumCircuit(n_p + n_w, name="ModExp")

        [mod_exp_circ.h(n_w + idx) for idx in range(n_p)]

        mod_exp_circ.x(0)
        
        for idx in range(n_p ):
            for _ in range(idx + 1):
                control_qubit_local = n_w + idx
                for swap_idx in range(n_w - 1):
                    target1 = n_w - swap_idx - 1
                    target2 = n_w - swap_idx - 2
                    mod_exp_circ.cswap(control_qubit_local, target1, target2)
            
        return mod_exp_circ.to_gate(label="ModExp")


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

    ## Subroutines Applied to the Main Circuit ##
    
    # Modular Exponentiation
    def modular_exponentiation(self):

        if self.precision_bits == 0 and self.working_bits == 0:
            return
            
        mod_exp_gate = self._create_mod_exp_gate()
        
        # Apply the gate to all qubits in the circuit
        all_qubits = range(self.working_bits + self.precision_bits)
        self.full_circuit.append(mod_exp_gate, all_qubits)
        self.full_circuit.barrier(label='ModExp') # This barrier is fine
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
    def shor_draw(self, scale):
        self.full_circuit.draw("mpl", initial_state=True, scale = 0.5)
        plt.show()
        pass

    # Circle Notation of final Statevector
    def shor_circle_viz(self):
        state_vec = Statevector(self.full_circuit)
        # QubitSystem(statevector=state_vec, label="Final state").viz_circle(max_cols=16)
        QubitSystem(statevector=state_vec, label="Final state").viz_circle_with_mag(max_cols=8)   
        pass
    
    # measure the precision register
    def shor_precision_measure(self):
        self.full_circuit.barrier(label='Meas')
        self.full_circuit.measure(
            range(self.working_bits, self.working_bits + self.precision_bits),
            range(self.precision_bits)
        )
        pass

## Testing ##
f = ShorCircuit(2, 7, 0)
f.inc_pow_2(6)
f.full_circuit.barrier(label='init')

f.controlled_geq_check('dec35', 1)

#f.shor_draw(scale=0.7)
#f.shor_circle_viz()
print(np.real(np.round(Statevector(f.full_circuit), 0)))


