# Moiré Neural Network: Computation via Geometric Interference with Emergent Physiological Oscillation

**Antti Luode**  
Independent Researcher, PerceptionLab  
February 2026

## Abstract

We present the Moiré Neural Network (MoiréNet), a neural computing architecture where neurons are spatial grid patterns defined by frequency, angle, and phase parameters. Computation emerges from moiré interference between overlapping grids rather than from stored weight matrices, reducing parameter storage from O(N²) to O(N). On the XOR benchmark, MoiréNet with 9 geometric parameters achieves 100% solve rate across 30 trials (mean 19 generations) compared to 73% solve rate for an equivalently-parameterized MLP (mean 541 epochs). We further demonstrate that when a spatial frequency generator (checkerboard) is placed in a homeostatic feedback loop with a sampling operator, the system spontaneously produces ECG-like oscillations. We show this emergent pacemaker behavior arises because a spatial pattern observed through a finite-resolution sampling grid at the Nyquist boundary cannot find a static equilibrium — periodic correction (oscillation) is the only stable solution. We propose that biological neural oscillations arise from the same mechanism: mutual observation between neurons through discrete ion channel arrays (membrane grid geometries) at mismatched sampling resolutions.

## 1. Introduction

Standard artificial neural networks store knowledge in explicit weight matrices connecting neurons. For N neurons, this requires O(N²) parameters. The forward pass consists of matrix multiplications, each requiring O(N²) operations. Learning occurs through backpropagation of gradients through these stored weights.

We propose an alternative: neurons defined by spatial grid geometries, where the effective "weight" between any two neurons is the moiré interference pattern between their grids. This requires only O(N) parameters (3 per neuron: frequency, angle, phase) and produces nonlinear decision boundaries intrinsically through the aliasing properties of overlapping periodic structures.

This architecture was inspired by the biological observation that ion channels form discrete, heterogeneous spatial distributions on neural membranes. Different neuron types express different channel patterns — effectively different "grid geometries." Synaptic transmission involves sampling a continuous neurotransmitter field through a postsynaptic receptor mosaic, creating interference between the output geometry of one neuron and the input geometry of another. We hypothesize that this geometric interference is not incidental but computational.

## 2. Architecture

### 2.1 Moiré Neurons

A Moiré Neuron is a 2D spatial grid defined by three parameters:

- **Frequency** (f): spatial period of the grid
- **Angle** (θ): orientation of the grid
- **Phase** (φ): offset

The neuron's grid value at position (x, y) is:

    g(x, y) = H(sin(2πf · (x cos θ + y sin θ) + φ))

where H is the Heaviside step function, producing a binary (0/1) square wave grid.

### 2.2 Moiré Computation

The effective "weight" between neurons A and B is their spatial correlation:

    w_AB = corr(g_A(x,y), g_B(x,y))

This is never stored — it is an implicit function of the two neurons' geometries. The moiré frequency (beat frequency) between two neurons is |f_A - f_B|, which determines the spatial scale of their interference pattern.

Classification uses the **parity** of overlapping grids:

    output(x, y) = (Σ_i g_i(x, y)) mod 2

This modular arithmetic creates nonlinear decision boundaries from pure geometric superposition — no activation function is needed.

### 2.3 Learning via Grid Remodeling

Learning adjusts grid parameters (f, θ, φ) via evolutionary optimization:

1. Randomly mutate one or more neuron geometries
2. Evaluate fitness (classification accuracy)
3. Keep improvement, revert otherwise

This is biologically analogous to synaptic plasticity via receptor redistribution (ion channel trafficking on the membrane surface), rather than weight adjustment.

## 3. Experiments

### 3.1 XOR Classification

XOR is the canonical test for nonlinear computation. A single-layer perceptron cannot solve XOR (Minsky & Papert, 1969) because the classes are not linearly separable. Moiré interference creates nonlinear boundaries through geometric parity.

**Setup:** MoiréNet with 3 neurons (9 parameters) vs. MLP with 2-2-1 architecture (9 weights + biases). 30 trials each, max 30,000 steps.

**Results:**

| Metric | MoiréNet | MLP (backprop) |
|--------|----------|----------------|
| Solve rate | 30/30 (100%) | 22/30 (73%) |
| Mean steps to solve | 19 ± 18 | 541 ± 199 |
| Mean time | 0.002s | 0.318s |
| Parameters | 9 | 9 |
| Mean accuracy | 100.0% | 86.7% |

MoiréNet solves XOR with 100% reliability in approximately 27× fewer iterations than the MLP, using the same number of parameters.

### 3.2 Emergent ECG from Homeostatic Feedback

**Setup:** A closed feedback loop in the PerceptionLab visual programming environment:

    ConstantSignal(1.0) → setpoint_mod
    Checkerboard(square_size) ← signal_out ← HomeostaticCoupler(edge_of_chaos)
    Checkerboard → image → ImageToVector(256) → VectorSplitter(16) → [out_0..out_3] → signal_in → Coupler

The checkerboard generates a spatial frequency. The Image→Vector→Splitter chain samples it into a finite-dimensional signal vector. The Homeostatic Coupler regulates signal variance toward a target (edge_of_chaos mode: amplify if too stable, dampen if too chaotic).

**Observation:** At vector dimension 256, the system spontaneously produces ECG-like oscillations on the coupler's history plot — rhythmic sharp spikes followed by slow recovery, resembling QRS complexes. This was not programmed; it emerged from the loop dynamics.

**Phase transitions by sampling resolution:**

| Vector Dim | Behavior |
|-----------|----------|
| 64-128 | Blunt, mechanical pulse (undersampled) |
| 256 | Rich, organic ECG (Nyquist-critical) |
| 1024 | Binary blinking or paralysis (oversampled/noisy) |
| 2048 | Slow collective heartbeat returns (statistical averaging) |

**Interpretation:** At vector dimension 256, the sampling grid and the checkerboard pattern are commensurate — maximum moiré stress, maximum ambiguity in self-observation. The coupler must periodically discharge to correct the accumulated aliasing error. The heartbeat is the system's solution to its own self-observation problem at the Nyquist boundary.

## 4. Discussion

### 4.1 Relation to Diffractive Deep Neural Networks

Lin et al. (Science, 2018) demonstrated all-optical neural networks using 3D-printed diffractive layers where computation emerges from wave interference. Our work shares the core principle but introduces two critical differences: (1) feedback loops producing emergent dynamics, and (2) biological motivation from membrane channel geometry.

### 4.2 Biological Implications

If computation in biological neural networks occurs partly through geometric interference of membrane channel distributions, this predicts:

1. **Neural oscillations are inevitable**, not optional. Mutual observation between neurons through finite receptor arrays at mismatched resolutions has no static solution — the network must oscillate.

2. **Learning is membrane remodeling.** Long-term potentiation physically rearranges receptor densities. This is grid parameter evolution, not weight adjustment.

3. **Frequency-dependent processing is intrinsic.** The same synapse computes different effective weights at different input frequencies, because moiré patterns shift with the frequency of the signal passing through overlapping grids.

4. **Hierarchical oscillations** (breathing → theta → gamma) arise naturally as aliasing correction at different spatial scales of the channel-distribution mismatch.

### 4.3 Limitations

- XOR is a minimal benchmark; scaling to complex tasks (MNIST, etc.) requires further work
- The evolutionary optimizer is simple; gradient-based or hybrid methods may improve convergence
- Biological claims are currently hypothetical and require experimental validation
- Digital simulation does not capture the speed/energy advantages of physical implementation
- This is classical wave interference, not quantum computing — parallelism without exponential scaling

### 4.4 Not Quantum, But Related

Moiré computing shares structural similarity with quantum interference (superposition → interference → nonlinear outcomes) but operates on classical fields. It achieves computational parallelism (multiple grid contributions simultaneously) without entanglement or quantum speedup. The analogy suggests that some advantages attributed to quantum computing may be accessible through classical wave architectures at vastly lower engineering complexity.

## 5. Conclusion

We have demonstrated that geometric interference between spatial grid patterns constitutes a viable computational primitive, achieving nonlinear classification with fewer iterations and higher reliability than equivalently-parameterized MLPs on the XOR benchmark. The discovery that a homeostatic feedback loop over a spatial frequency generator spontaneously produces physiological oscillation at the Nyquist sampling boundary suggests a deep connection between moiré computation and biological neural dynamics. The minimal mechanism — self-observation at finite resolution forced to invent time — may explain why nervous systems oscillate.

## References

1. Lin, X. et al. "All-optical machine learning using diffractive deep neural networks." Science 361.6406 (2018): 1004-1008.
2. Minsky, M. & Papert, S. "Perceptrons." MIT Press (1969).
3. Plate, T.A. "Holographic reduced representations." IEEE Trans. Neural Networks 6.3 (1995): 623-641.
4. Shew, W.L. & Plenz, D. "The functional benefits of criticality in the cortex." The Neuroscientist 19.1 (2013): 88-100.
5. Beggs, J.M. & Plenz, D. "Neuronal avalanches in neocortical circuits." Journal of Neuroscience 23.35 (2003): 11167-11177.

## Code Availability

All code, interactive demos, and PerceptionLab configurations are available at:
https://github.com/anttiluode/moire-neural-network
