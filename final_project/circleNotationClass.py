import matplotlib.pyplot as plt
import math
import numpy as np

class QubitSystem:
    def __init__(self, statevector: np.ndarray, label: str = "Qubit System"):
        self.label = label
        self.set_statevector(statevector)

    def set_statevector(self, statevector: np.ndarray):
        sv = np.asarray(statevector, dtype=np.complex128).flatten()
        if sv.ndim != 1:
            raise ValueError("Statevector must be 1D.")
        n_states = sv.size
        n_qubits = int(round(math.log2(n_states)))
        if 2**n_qubits != n_states:
            raise ValueError("Length must be a power of 2.")
        
        norm = np.linalg.norm(sv)
        if norm != 0 and not np.isclose(norm, 1.0):
            sv = sv / norm

        self.n_qubits = n_qubits
        self.n_states = n_states
        self.amps   = sv
        self.prob   = np.abs(sv)**2
        self.phase  = np.angle(sv)

    def viz_circle(self, max_cols: int = 8, figsize_scale: float = 2.3):
        cols = max(1, min(max_cols, self.n_states))
        rows = int(math.ceil(self.n_states / cols))

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols*figsize_scale, rows*(figsize_scale+0.2))
        )
        axes = np.atleast_2d(axes)

        def bitstr(i: int, n: int) -> str:
            return format(i, f"0{n}b")

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.set_aspect("equal")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")

            if idx >= self.n_states:
                ax.set_visible(False)
                continue

            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))
            radius = 0.48 * np.sqrt(self.prob[idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, alpha=0.25))
            angle = self.phase[idx]
            L = 0.45
            x2 = 0.5 + L * np.cos(angle)
            y2 = 0.5 + L * np.sin(angle)
            ax.arrow(0.5, 0.5, x2 - 0.5, y2 - 0.5,
                     head_width=0.03, head_length=0.05, length_includes_head=True)

            ax.set_title(f"|{bitstr(idx, self.n_qubits)}⟩", fontsize=10)

        fig.suptitle(self.label, fontsize=12)
        plt.tight_layout()
        plt.show()

    # --- MODIFIED FUNCTION ---
    def viz_circle_with_mag(
        self,
        max_cols: int = 8,
        figsize_scale: float = 2.3,
        tol: float = 1e-9,
        max_plots: int = 64, # <<< NEW: Limit the number of plots
        working_bits: int = 0,
        precision_bits: int = 0,
        Cin_bits: int = 1,
        B_bits: int = 0,
        Cout_bits: int = 0,
        N_bits: int = 0,
        Flag_bits: int = 0,
        reverse_bits: bool = False
    ):
        """
        Visualizes the qubit states with significant magnitude, limiting the output
        to the `max_plots` most probable states.
        """
        use_new_labels = any(b > 0 for b in [working_bits, precision_bits, B_bits, Cout_bits, N_bits, Flag_bits])
        
        # This logic for deriving P/W bits seems complex, simplifying for clarity
        # but retaining original functionality if only one is given.
        if use_new_labels:
            total_labeled_bits = (working_bits + precision_bits  + Cin_bits +
                                  B_bits + Cout_bits + N_bits + Flag_bits)
            if total_labeled_bits != self.n_qubits:
                 print(f"Warning: The sum of bit parts ({total_labeled_bits}) does not equal the total number of qubits ({self.n_qubits}). Labeling may be incorrect.")


        def get_title(state_idx):
            bitstring = format(state_idx, f'0{self.n_qubits}b')

            if not use_new_labels:
                if reverse_bits:
                    bitstring = bitstring[::-1]
                return f"|{bitstring}⟩"
            else:
                parts = []
                start = 0
                
                if Flag_bits > 0:
                    val = int(bitstring[start:start+Flag_bits], 2)
                    parts.append(f"F|{val}⟩")
                    start += Flag_bits
                if N_bits > 0:
                    val = int(bitstring[start:start+N_bits], 2)
                    parts.append(f"N|{val}⟩")
                    start += N_bits
                if Cout_bits > 0:
                    val = int(bitstring[start:start+Cout_bits], 2)
                    parts.append(f"Cout|{val}⟩")
                    start += Cout_bits
                if B_bits > 0:
                    val = int(bitstring[start:start+B_bits], 2)
                    parts.append(f"B|{val}⟩")
                    start += B_bits
                if Cin_bits > 0:
                    val = int(bitstring[start:start+Cin_bits], 2)
                    parts.append(f"Cin|{val}⟩")
                    start += Cin_bits
                # Handle precision and working bits at the end, as in original logic
                precision_str = bitstring[start:start+precision_bits]
                working_str = bitstring[start+precision_bits:]

                if reverse_bits: # Assuming reverse only applies to precision
                    precision_str = precision_str[::-1]
                
                if precision_bits > 0:
                    val = int(precision_str, 2)
                    parts.append(f"P|{val}⟩")
                if working_bits > 0:
                    val = int(working_str, 2)
                    parts.append(f"W|{val}⟩")
                
                return " ".join(parts)


        # 1. Find all indices with probability greater than the tolerance
        mag_indices = np.where(self.prob > tol)[0]

        if mag_indices.size == 0:
            print(f"No states with probability > {tol} to plot for '{self.label}'.")
            return

        # 2. If there are more states than max_plots, sort by probability and take the top ones
        if mag_indices.size > max_plots:
            # Get probabilities for these specific indices
            probs_for_indices = self.prob[mag_indices]
            # Get the indices that would sort these probabilities in descending order
            sorted_by_prob_indices = np.argsort(probs_for_indices)[::-1]
            # Keep only the top `max_plots` indices from the original `mag_indices`
            mag_indices = mag_indices[sorted_by_prob_indices[:max_plots]]
        
        # --- Your original sorting logic by P/W bits can still apply to the filtered list ---
        # This part is complex and assumes a specific bit ordering. It will now sort
        # the *subset* of most probable states, which is more efficient.
        if use_new_labels and len(mag_indices) > 0 and (precision_bits > 0 or working_bits > 0):
             states_to_sort = []
             # Define bit sections based on total number of qubits and specified parts
             start_p = Flag_bits + N_bits + Cout_bits + B_bits + Cin_bits 
             start_w = start_p + precision_bits

             for state_idx in mag_indices:
                 bitstring = format(state_idx, f'0{self.n_qubits}b')
                 precision_str = bitstring[start_p : start_w]
                 working_str = bitstring[start_w : start_w + working_bits]

                 if reverse_bits:
                     precision_str = precision_str[::-1]

                 precision_val = int(precision_str, 2) if precision_str else 0
                 working_val = int(working_str, 2) if working_str else 0
                 states_to_sort.append((precision_val, working_val, state_idx))
            
             states_to_sort.sort() # Sorts by P, then W
             mag_indices = [item[2] for item in states_to_sort]


        n_to_plot = len(mag_indices)
        cols = max(1, min(max_cols, n_to_plot))
        rows = int(math.ceil(n_to_plot / cols))

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * figsize_scale, rows * (figsize_scale + 0.3)),
            squeeze=False
        )
        axes_flat = axes.flatten()

        for i, state_idx in enumerate(mag_indices):
            ax = axes_flat[i]
            ax.set_aspect("equal")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            
            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))
            radius = 0.48 * np.sqrt(self.prob[state_idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, color='blue', alpha=0.25))
            angle = self.phase[state_idx]
            L = 0.45
            x2 = 0.5 + L * np.cos(angle)
            y2 = 0.5 + L * np.sin(angle)
            ax.arrow(0.5, 0.5, x2 - 0.5, y2 - 0.5,
                     head_width=0.03, head_length=0.05, length_includes_head=True, color='red')
            
            ax.set_title(get_title(state_idx), fontsize=9)

        # Hide any unused subplots
        for i in range(n_to_plot, len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        fig.suptitle(f"{self.label} (Top {n_to_plot} most probable states)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()