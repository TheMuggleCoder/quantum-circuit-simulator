# A basic Quantum State Simulator with Tkinter GUI

quantum-circuit-simulator is a lightweight and easy-to-use quantum state vector simulator implemented in Python with a Tkinter graphical user interface. It supports basic quantum gates, controlled operations, and measurements for up to 6 qubits.

## Features

- Single-file Python implementation, minimal dependencies (only `numpy`)
- Supports standard quantum gates: H, X, Y, Z, S, T, RX, RY, RZ, CX, CZ
- Visualize quantum state amplitudes and probabilities
- Measure quantum states with configurable shots and histogram display
- Load example circuits like Bell and GHZ states
- Simple and intuitive GUI for building and running quantum programs

## Limitations

- Supports up to 6 qubits due to exponential memory requirements
- No support for noise models or error correction
- Basic visualization only (no circuit diagrams)
- Single-threaded execution, so large circuits might be slow
- Focused on educational and experimental use, not optimized for production
  
## Requirements

- Python 3.8+
- numpy (`pip install numpy`)

## Usage

Run the simulator with:

```bash
python mini_qsim_tk.py
```

Use the GUI to add quantum gates, build circuits, and visualize results.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to experiment and expand the simulator! Contributions and feedback are welcome.
