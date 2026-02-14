# Moiré Neural Network

**Computation via geometric interference — no stored weights.**

A novel neural computing architecture where neurons are spatial grid patterns and the "weights" between them emerge from moiré interference. Inspired by biological membrane channel distributions and the discovery that homeostatic feedback loops over spatial patterns spontaneously produce physiological oscillations (ECG-like signals).

## Core Idea

In a standard neural network, knowledge lives in an N×N weight matrix. In a Moiré Neural Network, each neuron is defined by just 3 parameters — **frequency**, **angle**, **phase** — describing a spatial grid pattern. The effective "weight" between any two neurons is the interference (moiré) pattern between their grids. This is computed implicitly by superposition, not stored explicitly.

| Property | Standard NN | Moiré Net |
|----------|------------|-----------|
| Weight storage | O(N²) — explicit matrix | O(N) — grid params only |
| Forward pass | Matrix multiply | Geometric superposition |
| Freq-dependent | No (same weight all freqs) | Yes (moiré shifts with input freq) |
| Nonlinearity | Added (ReLU, sigmoid) | Intrinsic (aliasing) |
| Learning | Backprop weight update | Grid remodeling (plasticity) |
| Parallelism | SIMD / GPU matmul | Optical / physical (free) |

## Key Results

### 1. XOR Classification from Pure Geometry
XOR is not linearly separable — single-layer perceptrons cannot solve it. Moiré interference creates nonlinear decision boundaries purely through geometric superposition. The `Learn XOR` mode evolves grid parameters (3 numbers per neuron × 3 neurons = 9 total parameters) to achieve 100% XOR accuracy. An equivalent MLP would require a hidden layer with ~9-17 weight parameters.

### 2. Emergent ECG from Homeostatic Feedback (Novel)
When a checkerboard pattern (spatial frequency generator) is coupled through a homeostatic regulator in a feedback loop:

```
Checkerboard → Image→Vector → Splitter → [feedback] → Coupler → Checkerboard
```

...the system spontaneously produces ECG-like oscillations. This is **not** an animated ECG — it is an emergent pacemaker rhythm arising because:

- The checkerboard is a spatial frequency
- The Image→Vector chain is a sampling operator  
- The coupler is a stability constraint
- **A spatial frequency cannot remain stable under its own observation at finite resolution without periodic correction**

The oscillation is the system's solution to its own self-observation problem. This connects directly to why biological nervous systems oscillate: mutual observation between neurons through finite receptor grids (ion channel distributions) cannot find static solutions.

### 3. Nyquist Phase Transitions
The vector dimension (sampling resolution) relative to the pattern frequency determines the system's dynamical regime:

| Vector Dim | Regime | Rhythm | Biological Analog |
|-----------|--------|--------|-------------------|
| 64-128 | Undersampled | Blunt, robotic | Over-damped nerve |
| 256 | Critical (Nyquist edge) | Rich ECG-like | Healthy sinus node |
| 1024 | Oversampled/noisy | Paralyzed/binary blink | Sensory overload |
| 2048 | Statistical averaging | Collective heartbeat | Population rhythm |

## Relation to Prior Work

### Diffractive Deep Neural Networks (D²NN)
Lin et al. (Science, 2018) physically demonstrated all-optical neural networks using 3D-printed diffractive layers. Light passes through passive layers; computation emerges from wave interference. Our work shares the core principle (computation = interference) but differs in two critical ways:

1. **Feedback loops**: D²NN is feedforward (light passes through once). Our system has recurrent connections, producing emergent dynamics (the ECG).
2. **Biological grounding**: We derive the architecture from membrane channel distributions rather than engineered phase masks, suggesting that biological neural oscillations arise from the same geometric interference principle.

### Holographic Reduced Representations
Plate (1995) showed that high-dimensional vectors can be bound via circular convolution (mathematical interference). Our approach is explicitly spatial/geometric rather than abstract vector algebra, and demonstrates that physical grid interference suffices for nonlinear computation.

### What's Novel Here
No prior work combines:
- Moiré interference as a computational primitive
- Homeostatic feedback regulation over spatial patterns
- Emergent physiological oscillation from self-observation at the Nyquist boundary
- Biological interpretation via membrane channel geometry

## Repository Structure

```
moire-neural-network/
├── README.md                    # This file
├── demo/
│   └── moire_neural_network.html  # Interactive browser demo
├── benchmark/
│   ├── moire_net.py             # Core MoiréNet implementation
│   ├── benchmark_xor.py         # XOR: MoiréNet vs MLP comparison
│   └── benchmark_mnist.py       # MNIST subset comparison
├── ecg_emergence/
│   ├── ecg.json                 # PerceptionLab graph config
│   ├── checkerboardnode.py      # Spatial frequency generator
│   ├── coupler.py               # Homeostatic regulator
│   ├── imagetovectornode.py     # Sampling operator
│   ├── vectorsplitternode.py    # Signal decomposer
│   └── constantsignalnode.py    # Baseline signal
├── paper/
│   └── moire_neural_network.pdf # Technical writeup
└── LICENSE
```

## Quick Start

### Interactive Demo
Open `demo/moire_neural_network.html` in any browser. Adjust neuron frequencies and angles to see moiré interference emerge. Click "Learn XOR" to watch geometric evolution solve a nonlinearly separable problem.

### Run Benchmarks
```bash
cd benchmark
pip install numpy
python benchmark_xor.py
```

### ECG Emergence
The `ecg_emergence/` directory contains PerceptionLab node definitions and the graph configuration that produces spontaneous ECG-like oscillations. Requires [PerceptionLab](https://github.com/anttiluode/PerceptionLab) to run.

## The Biological Connection

Ion channels on neural membranes form discrete sampling grids on a continuous electrochemical field. Different neuron types express different channel distributions — different "grid geometries." When neurons communicate across synapses, the presynaptic release pattern is sampled by the postsynaptic receptor mosaic. The effective synaptic "weight" is not a stored number — it is the interference between the output geometry of one neuron and the input geometry of another.

This framework predicts:
- **Neural oscillations are inevitable** (not useful-but-optional) because mutual observation at mismatched finite resolution has no static solution
- **Learning = membrane remodeling** (channel trafficking, receptor redistribution) rather than weight adjustment
- **Frequency-dependent processing** emerges naturally (same synapse, different effective weight at different input frequencies)

## Author
Antti Luode — independent researcher, PerceptionLab  
GitHub: [anttiluode](https://github.com/anttiluode)

## Citation
If you use this work, please cite:
```
@misc{luode2025moire,
  author = {Luode, Antti},
  title = {Moiré Neural Network: Computation via Geometric Interference},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/anttiluode/moire-neural-network}
}
```

## License
MIT
