#!/usr/bin/env python3
"""
mini_qsim_tk.py — Single-file minimal quantum state-vector simulator with Tkinter GUI.

Requirements:
- Python 3.8+
- numpy

Run:
    python mini_qsim_tk.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, scrolledtext
import numpy as np
import math
from typing import List, Tuple, Dict

# ---------------------------
# Quantum core
# ---------------------------

def kron_n(mats: List[np.ndarray]) -> np.ndarray:
    res = mats[0]
    for m in mats[1:]:
        res = np.kron(res, m)
    return res

def idx_bits(idx: int, n: int) -> str:
    return format(idx, '0{}b'.format(n))

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)

def RX(theta):
    return np.array([[math.cos(theta/2), -1j*math.sin(theta/2)],
                     [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=complex)

def RY(theta):
    return np.array([[math.cos(theta/2), -math.sin(theta/2)],
                     [math.sin(theta/2), math.cos(theta/2)]], dtype=complex)

def RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    ops = []
    for i in range(n):
        ops.append(gate if i == target else I)
    full = kron_n(ops)
    return full.dot(state)

def apply_controlled_gate(state: np.ndarray, control: int, target: int, gate: np.ndarray, n: int) -> np.ndarray:
    dim = 2 ** n
    new = np.zeros_like(state, dtype=complex)
    for idx in range(dim):
        amp = state[idx]
        if abs(amp) < 1e-18:
            continue
        bits = idx_bits(idx, n)
        if bits[control] == '1':
            # compute partner indices for target=0 and target=1 under the same other bits
            bs = list(bits)
            bs[target] = '0'
            idx0 = int(''.join(bs), 2)
            bs[target] = '1'
            idx1 = int(''.join(bs), 2)
            a0 = state[idx0]
            a1 = state[idx1]
            r0 = gate[0,0]*a0 + gate[0,1]*a1
            r1 = gate[1,0]*a0 + gate[1,1]*a1
            new[idx0] += r0
            new[idx1] += r1
        else:
            new[idx] += amp
    return new

class QuantumState:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.state = np.zeros((self.dim,), dtype=complex)
        self.state[0] = 1.0

    def reset(self):
        self.state[:] = 0
        self.state[0] = 1.0

    def apply_gate(self, name: str, targets: List[int], params: Tuple = ()):
        name = name.upper()
        if name in ['H', 'X', 'Y', 'Z', 'S', 'T']:
            gate_map = {'H': H, 'X': X, 'Y': Y, 'Z': Z, 'S': S, 'T': T}
            gate = gate_map[name]
            t = targets[0]
            self.state = apply_single_qubit_gate(self.state, gate, t, self.n)
        elif name in ['RX', 'RY', 'RZ']:
            theta = params[0] if params else 0.0
            gate = {'RX': RX(theta), 'RY': RY(theta), 'RZ': RZ(theta)}[name]
            t = targets[0]
            self.state = apply_single_qubit_gate(self.state, gate, t, self.n)
        elif name in ['CX', 'CNOT']:
            control, target = targets
            self.state = apply_controlled_gate(self.state, control, target, X, self.n)
        elif name == 'CZ':
            control, target = targets
            self.state = apply_controlled_gate(self.state, control, target, Z, self.n)
        else:
            raise ValueError("Unknown gate: " + name)

    def amplitudes(self) -> List[Tuple[str, complex]]:
        return [(idx_bits(i, self.n), amp) for i, amp in enumerate(self.state)]

    def probabilities(self) -> List[Tuple[str, float]]:
        probs = np.abs(self.state) ** 2
        return [(idx_bits(i, self.n), float(probs[i])) for i in range(self.dim)]

    def measure(self, shots: int = 1) -> Dict[str, int]:
        probs = np.abs(self.state) ** 2
        outcomes = np.random.choice(self.dim, size=shots, p=probs)
        counts: Dict[str, int] = {}
        for o in outcomes:
            k = idx_bits(o, self.n)
            counts[k] = counts.get(k, 0) + 1
        if shots == 1:
            sampled = outcomes[0]
            newstate = np.zeros_like(self.state)
            newstate[sampled] = 1.0
            self.state = newstate
        return counts

# ---------------------------
# Examples
# ---------------------------

def example_bell(n=2):
    qs = QuantumState(n)
    qs.apply_gate('H', [0])
    qs.apply_gate('CX', [0, 1])
    return qs

def example_ghz(n=3):
    qs = QuantumState(n)
    qs.apply_gate('H', [0])
    for i in range(n-1):
        qs.apply_gate('CX', [i, i+1])
    return qs

EXAMPLES = {'bell': example_bell, 'ghz': example_ghz}

# ---------------------------
# GUI
# ---------------------------

class MiniQSimGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("mini_qsim — Tkinter GUI")
        self.geometry("950x640")
        self.resizable(False, False)

        # Top controls
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(top, text="Qubits:").pack(side=tk.LEFT)
        self.n_spin = tk.Spinbox(top, from_=1, to=6, width=4, command=self.change_n)
        self.n_spin.delete(0, 'end'); self.n_spin.insert(0, "3")
        self.n_spin.pack(side=tk.LEFT, padx=(4,12))

        ttk.Button(top, text="Reset State", command=self.reset_state).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load Example", command=self.load_example_dialog).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Clear Program", command=self.clear_program).pack(side=tk.LEFT, padx=4)

        # Left: program builder
        left = ttk.LabelFrame(self, text="Program / Builder", width=360, height=560)
        left.place(x=10, y=60)

        # Gate selection
        gate_row = ttk.Frame(left)
        gate_row.pack(padx=8, pady=6, fill=tk.X)
        ttk.Label(gate_row, text="Gate:").pack(side=tk.LEFT)
        self.gate_cb = ttk.Combobox(gate_row, values=['H','X','Y','Z','S','T','RX','RY','RZ','CX','CZ'], width=6)
        self.gate_cb.set('H'); self.gate_cb.pack(side=tk.LEFT, padx=6)

        ttk.Label(gate_row, text="Targets (comma sep)").pack(side=tk.LEFT, padx=(8,2))
        self.targets_entry = ttk.Entry(gate_row, width=12)
        self.targets_entry.pack(side=tk.LEFT)

        ttk.Label(gate_row, text="Angle (rad)").pack(side=tk.LEFT, padx=(8,2))
        self.angle_entry = ttk.Entry(gate_row, width=8)
        self.angle_entry.pack(side=tk.LEFT)

        btn_row = ttk.Frame(left)
        btn_row.pack(padx=8, pady=(0,6), fill=tk.X)
        ttk.Button(btn_row, text="Add Gate", command=self.add_gate).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Insert Gate", command=self.insert_gate).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Run Program", command=self.run_program).pack(side=tk.LEFT, padx=6)

        # Program listbox
        self.prog_list = tk.Listbox(left, width=52, height=24)
        self.prog_list.pack(padx=8, pady=6)

        # Middle: state / probs
        mid = ttk.LabelFrame(self, text="State & Probabilities", width=330, height=560)
        mid.place(x=380, y=150)

        self.state_text = scrolledtext.ScrolledText(mid, width=40, height=16, wrap=tk.WORD)
        self.state_text.pack(padx=8, pady=6)

        probs_row = ttk.Frame(mid)
        probs_row.pack(fill=tk.X, padx=8)
        ttk.Button(probs_row, text="Show Amplitudes", command=self.show_amplitudes).pack(side=tk.LEFT)
        ttk.Button(probs_row, text="Show Probabilities", command=self.show_probabilities).pack(side=tk.LEFT, padx=6)

        meas_row = ttk.Frame(mid)
        meas_row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(meas_row, text="Shots:").pack(side=tk.LEFT)
        self.shots_entry = ttk.Entry(meas_row, width=6); self.shots_entry.insert(0, "1")
        self.shots_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(meas_row, text="Measure", command=self.measure).pack(side=tk.LEFT, padx=6)

        # Right: histogram & info
        right = ttk.LabelFrame(self, text="Measurement Histogram", width=240, height=560)
        right.place(x=725, y=60)

        self.canvas = tk.Canvas(right, width=220, height=300, bg="white")
        self.canvas.pack(padx=8, pady=8)

        info = ttk.LabelFrame(self, text="Notes", width=240, height=200)
        info.place(x=725, y=380)
        info_lbl = tk.Label(info, text="Qubit indexing: 0 = MSB (leftmost)\nExample: |01> -> qubit0=0, qubit1=1\nRecommended n ≤ 5 for GUI", justify=tk.LEFT)
        info_lbl.pack(padx=6, pady=6)

        # Initialize quantum state
        self.n = int(self.n_spin.get())
        self.qs = QuantumState(self.n)
        self.program: List[Tuple[str, List[int], Tuple]] = []

        # Fill with a small default program (optional)
        self.update_ui_state()

    # ---------------------------
    # Program builder helpers
    # ---------------------------
    def parse_targets(self, txt: str) -> List[int]:
        if not txt.strip():
            raise ValueError("Provide target qubit index(es).")
        parts = [p.strip() for p in txt.split(',') if p.strip()!='']
        vals = []
        for p in parts:
            v = int(p)
            if v < 0 or v >= self.n:
                raise ValueError(f"Qubit index must be 0..{self.n-1}")
            vals.append(v)
        return vals

    def add_gate(self):
        try:
            name = self.gate_cb.get().strip().upper()
            targets = self.parse_targets(self.targets_entry.get())
            params = ()
            if name in ['RX','RY','RZ']:
                theta = float(self.angle_entry.get().strip() or 0.0)
                params = (theta,)
            if name in ['H','X','Y','Z','S','T'] and len(targets) != 1:
                raise ValueError(f"{name} expects 1 target.")
            if name in ['RX','RY','RZ'] and len(targets) != 1:
                raise ValueError(f"{name} expects 1 target.")
            if name in ['CX','CNOT','CZ'] and len(targets) != 2:
                raise ValueError(f"{name} expects 2 targets (control,target).")
            self.program.append((name, targets, params))
            self.prog_list.insert(tk.END, self.program[-1])
            self.targets_entry.delete(0, 'end')
            self.angle_entry.delete(0, 'end')
        except Exception as e:
            messagebox.showerror("Error adding gate", str(e))

    def insert_gate(self):
        sel = self.prog_list.curselection()
        if not sel:
            messagebox.showinfo("Insert", "Select a position in the program list to insert.")
            return
        idx = sel[0]
        try:
            name = self.gate_cb.get().strip().upper()
            targets = self.parse_targets(self.targets_entry.get())
            params = ()
            if name in ['RX','RY','RZ']:
                theta = float(self.angle_entry.get().strip() or 0.0)
                params = (theta,)
            self.program.insert(idx, (name, targets, params))
            self.prog_list.insert(idx, self.program[idx])
        except Exception as e:
            messagebox.showerror("Error inserting gate", str(e))

    def remove_selected(self):
        sel = self.prog_list.curselection()
        if not sel:
            messagebox.showinfo("Remove", "Select item(s) to remove.")
            return
        for i in reversed(sel):
            self.prog_list.delete(i)
            del self.program[i]

    def clear_program(self):
        self.program.clear()
        self.prog_list.delete(0, tk.END)

    # ---------------------------
    # State / run
    # ---------------------------
    def change_n(self):
        try:
            newn = int(self.n_spin.get())
            if newn < 1 or newn > 12:
                messagebox.showerror("Invalid n", "Choose n between 1 and 12 (recommended ≤ 6).")
                return
            self.n = newn
            self.qs = QuantumState(self.n)
            self.program.clear()
            self.prog_list.delete(0, tk.END)
            self.update_ui_state()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_state(self):
        self.qs = QuantumState(self.n)
        self.update_ui_state()
        messagebox.showinfo("Reset", "State reset to |0...0>")

    def load_example_dialog(self):
        name = simpledialog.askstring("Example", "Enter example (bell or ghz):", parent=self)
        if not name:
            return
        name = name.strip().lower()
        if name not in EXAMPLES:
            messagebox.showerror("Unknown example", "Supported: bell, ghz")
            return
        # If example n doesn't match chosen n, we try to adapt: bell uses n=2 but can run on larger register applying gates on MSBs
        # We'll just create fresh state and apply gates on first n qubits as examples do.
        ex_qs = EXAMPLES[name](self.n if self.n >= (2 if name=='bell' else 3) else (2 if name=='bell' else 3))
        # If ex_qs.n doesn't equal current n, reinitialize and apply same gates manually
        # Simpler: replace current state with ex_qs truncated/expanded appropriately:
        if ex_qs.n == self.n:
            self.qs = ex_qs
        elif ex_qs.n < self.n:
            # expand by tensoring extra zeros on right (least significant)
            # build state vector of size 2^n with ex_qs.state occupying MSB qubits
            new_state = np.zeros(2**self.n, dtype=complex)
            for i, amp in enumerate(ex_qs.state):
                # shift left by (n - ex_n) qubits
                new_index = i << (self.n - ex_qs.n)
                new_state[new_index] = amp
            self.qs.state = new_state
        else:
            # ex_qs.n > self.n: compress by tracing out lower qubits (not exact). We'll just reset and run example on first self.n qubits
            self.qs = QuantumState(self.n)
            if name == 'bell' and self.n >= 2:
                self.qs.apply_gate('H', [0]); self.qs.apply_gate('CX', [0,1])
            elif name == 'ghz' and self.n >= 2:
                self.qs.apply_gate('H', [0])
                for i in range(min(self.n-1, 5)):
                    self.qs.apply_gate('CX', [i, i+1])
        self.update_ui_state()
        messagebox.showinfo("Example loaded", f"Loaded example '{name}' into state (n={self.n}).")

    def run_program(self):
        try:
            if not self.program:
                messagebox.showinfo("Run", "Program is empty.")
                return
            for g, ts, p in self.program:
                self.qs.apply_gate(g, ts, p)
            self.update_ui_state()
            messagebox.showinfo("Run", "Program executed.")
        except Exception as e:
            messagebox.showerror("Runtime error", str(e))

    def show_amplitudes(self):
        lines = []
        for bits, amp in self.qs.amplitudes():
            if abs(amp) > 1e-9:
                amp_s = f"{amp.real:.5f}" + (f" + {amp.imag:.5f}j" if abs(amp.imag) > 1e-9 else "")
                lines.append(f"|{bits}> : amp={amp_s}  prob={abs(amp)**2:.6f}")
        self.state_text.delete(1.0, tk.END)
        self.state_text.insert(tk.END, "\n".join(lines))

    def show_probabilities(self):
        probs = sorted(self.qs.probabilities(), key=lambda x:-x[1])
        lines = [f"|{bits}> : {p:.6f}" for bits,p in probs if p>1e-9]
        self.state_text.delete(1.0, tk.END)
        self.state_text.insert(tk.END, "\n".join(lines))

    def measure(self):
        try:
            shots = int(self.shots_entry.get().strip() or "1")
            if shots < 1:
                raise ValueError("shots must be ≥ 1")
            counts = self.qs.measure(shots=shots)
            # show counts and draw histogram
            sorted_counts = sorted(counts.items(), key=lambda x:-x[1])
            txt = "\n".join([f"{k} : {v}" for k,v in sorted_counts])
            self.state_text.delete(1.0, tk.END)
            self.state_text.insert(tk.END, txt)
            self.draw_histogram(counts, shots)
        except Exception as e:
            messagebox.showerror("Measure error", str(e))

    def draw_histogram(self, counts: Dict[str,int], shots: int):
        self.canvas.delete("all")
        if not counts:
            return
        items = sorted(counts.items(), key=lambda x: x[0])  # lexicographic
        maxv = max(counts.values())
        w = 200
        h = 260
        margin_x = 10
        margin_y = 10
        bar_w = (w - 2*margin_x) / len(items)
        for i, (k, v) in enumerate(items):
            x0 = margin_x + i*bar_w
            x1 = x0 + bar_w*0.8
            height = (v / maxv) * (h - 2*margin_y) if maxv>0 else 0
            y0 = margin_y + (h - margin_y) - height
            y1 = margin_y + (h - margin_y)
            # rectangle
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="#6fa8dc", outline="black")
            # label
            self.canvas.create_text((x0+x1)/2, y1+8, text=k, anchor=tk.N, font=("TkDefaultFont", 8))
            self.canvas.create_text((x0+x1)/2, y0-6, text=str(v), anchor=tk.S, font=("TkDefaultFont", 8))

    def update_ui_state(self):
        # update program listbox text in case program exists
        self.prog_list.delete(0, tk.END)
        for item in self.program:
            self.prog_list.insert(tk.END, item)
        # refresh state display summary
        self.state_text.delete(1.0, tk.END)
        self.state_text.insert(tk.END, f"State: |{'0'*self.n}>\n")
        # show a short prob summary
        probs = sorted(self.qs.probabilities(), key=lambda x:-x[1])
        top = probs[:6]
        lines = ["\nProbabilities (top):"]
        for bits,p in top:
            if p > 1e-9:
                lines.append(f"{bits} : {p:.6f}")
        self.state_text.insert(tk.END, "\n".join(lines))

# ---------------------------
# Run app
# ---------------------------

if __name__ == '__main__':
    try:
        import numpy as _np_check  # ensure numpy available
    except Exception:
        messagebox.showerror("Missing dependency", "This app requires numpy. Install with: pip install numpy")
        raise
    app = MiniQSimGUI()
    app.mainloop()
