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
        # Defensive normalization (harmless if already normalized)
        norm = np.linalg.norm(sv)
        if norm != 0 and not np.isclose(norm, 1.0):
            sv = sv / norm

        self.n_qubits = n_qubits
        self.n_states = n_states
        self.amps  = sv
        self.prob  = np.abs(sv)**2
        self.phase = np.angle(sv)

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

            # Outer reference circle
            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))

            # Filled disk: radius ∝ sqrt(probability) so area ∝ probability
            radius = 0.48 * np.sqrt(self.prob[idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, alpha=0.25))

            # Phase arrow
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
        
    def viz_circle_with_mag(self, max_cols: int = 8, figsize_scale: float = 2.3, tol: float = 1e-9, working_bits: int = 0, precision_bits: int = 0, flag_bit:int = 0, reverse_bits: bool = False):
        use_new_labels = working_bits > 0 or precision_bits > 0
        if use_new_labels:
            if working_bits > 0 and precision_bits == 0:
                precision_bits = self.n_qubits - working_bits
            elif precision_bits > 0 and working_bits == 0:
                working_bits = self.n_qubits - precision_bits

            if working_bits + precision_bits + flag_bit != self.n_qubits or working_bits < 0 or precision_bits < 0:
                raise ValueError(
                    f"Sum of working ({working_bits}) and precision ({precision_bits}) bits and flag bit ({flag_bit}) "
                    f"must equal total qubits ({self.n_qubits}) and be non-negative."
                )

        def get_title(state_idx):
            bitstring = format(state_idx, f'0{self.n_qubits}b')

            if not use_new_labels:
                if reverse_bits:
                    bitstring = bitstring[::-1]
                return f"|{bitstring}⟩"
            else:
                # When using P/W labels, reverse_bits only applies to the precision part
                precision_str = bitstring[:precision_bits-flag_bit]
                working_str = bitstring[precision_bits:]
                flag_str = bitstring[precision_bits-flag_bit:precision_bits]

                if reverse_bits:
                    precision_str = precision_str[::-1]

                precision_val = int(precision_str, 2) if precision_str else 0
                working_val = int(working_str, 2) if working_str else 0
                flag_val = int(flag_str, 2) if flag_str else 0
                return f"F|{flag_val}⟩ P|{precision_val}⟩ W|{working_val}⟩"

        mag_indices = np.where(self.prob > tol)[0]

        # --- Sort the indices by Precision then Working bits if applicable ---
        if use_new_labels and len(mag_indices) > 0:
            states_to_sort = []
            for state_idx in mag_indices:
                bitstring = format(state_idx, f'0{self.n_qubits}b')
                precision_str = bitstring[:precision_bits]
                working_str = bitstring[precision_bits:]

                if reverse_bits:
                    precision_str = precision_str[::-1]

                precision_val = int(precision_str, 2) if precision_str else 0
                working_val = int(working_str, 2) if working_str else 0
                states_to_sort.append((precision_val, working_val, state_idx))

            # Sort by P, then W, then original index as a tie-breaker
            states_to_sort.sort()

            # Overwrite mag_indices with the new sorted order
            mag_indices = [item[2] for item in states_to_sort]
        # --- End Sorting ---

        n_to_plot = len(mag_indices)

        if n_to_plot == 0:
            print(f"No states with probability > {tol} to plot for '{self.label}'.")
            return

        cols = max(1, min(max_cols, n_to_plot))
        rows = int(math.ceil(n_to_plot / cols))

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * figsize_scale, rows * (figsize_scale + 0.2)),
            squeeze=False
        )
        axes_flat = axes.flatten()

        for i in range(len(axes_flat)):
            ax = axes_flat[i]
            if i >= n_to_plot:
                ax.set_visible(False)
                continue

            ax.set_aspect("equal")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis("off")
            state_idx = mag_indices[i]
            ax.add_patch(plt.Circle((0.5, 0.5), 0.48, fill=False, lw=1.0, alpha=0.5))
            radius = 0.48 * np.sqrt(self.prob[state_idx])
            ax.add_patch(plt.Circle((0.5, 0.5), radius, alpha=0.25))
            angle = self.phase[state_idx]
            L = 0.45
            x2 = 0.5 + L * np.cos(angle)
            y2 = 0.5 + L * np.sin(angle)
            ax.arrow(0.5, 0.5, x2 - 0.5, y2 - 0.5,
                     head_width=0.03, head_length=0.05, length_includes_head=True)

            ax.set_title(get_title(state_idx), fontsize=10)

        fig.suptitle(f"{self.label} (states with magnitude)", fontsize=12)
        plt.tight_layout()
        plt.show()