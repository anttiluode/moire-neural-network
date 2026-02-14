"""
Moiré Neural Network — Core Implementation
============================================
Computation via geometric interference. No stored weight matrices.

Each neuron is a spatial grid defined by (frequency, angle, phase).
The effective "weight" between neurons emerges from moiré interference.
Learning = evolving grid parameters (biological: membrane remodeling).

Author: Antti Luode
"""

import numpy as np
from typing import List, Tuple, Optional


class MoireNeuron:
    """
    A single neuron defined by its grid geometry.
    
    Instead of storing N weights to N other neurons,
    stores only 3 parameters: frequency, angle, phase.
    The effective connection to any other neuron is computed
    implicitly via moiré interference.
    """
    
    def __init__(self, freq: float = 10.0, angle: float = 0.0, phase: float = 0.0):
        self.freq = freq
        self.angle = angle
        self.phase = phase
    
    def grid_value(self, x: float, y: float) -> float:
        """
        Evaluate this neuron's grid at position (x, y).
        Returns 0 or 1 (square wave grid).
        """
        rx = x * np.cos(self.angle) + y * np.sin(self.angle)
        return 1.0 if np.sin(rx * self.freq * 2 * np.pi + self.phase) > 0 else 0.0
    
    def grid_value_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized grid evaluation."""
        rx = x * np.cos(self.angle) + y * np.sin(self.angle)
        return (np.sin(rx * self.freq * 2 * np.pi + self.phase) > 0).astype(np.float32)
    
    def params(self) -> np.ndarray:
        return np.array([self.freq, self.angle, self.phase])
    
    def set_params(self, params: np.ndarray):
        self.freq = float(params[0])
        self.angle = float(params[1])
        self.phase = float(params[2])


class MoireLayer:
    """
    A layer of moiré neurons.
    
    Computation: input signal is sampled through the interference
    pattern of all neurons in the layer. The moiré between neuron
    grids creates nonlinear transformations without explicit weights.
    """
    
    def __init__(self, n_neurons: int, freq_range: Tuple[float, float] = (3, 30)):
        self.neurons = []
        for _ in range(n_neurons):
            freq = np.random.uniform(*freq_range)
            angle = np.random.uniform(0, np.pi)
            phase = np.random.uniform(0, 2 * np.pi)
            self.neurons.append(MoireNeuron(freq, angle, phase))
    
    def forward(self, x: float, y: float) -> np.ndarray:
        """
        Evaluate all neurons at position (x, y).
        Returns array of grid values.
        """
        return np.array([n.grid_value(x, y) for n in self.neurons])
    
    def interference(self, x: float, y: float) -> float:
        """
        Compute the moiré interference (parity) at position (x, y).
        This is the key nonlinear operation — modular arithmetic
        on overlapping grids creates XOR-like decision boundaries.
        """
        values = self.forward(x, y)
        return float(int(np.sum(values)) % 2)
    
    def param_count(self) -> int:
        return len(self.neurons) * 3
    
    def get_params(self) -> np.ndarray:
        return np.concatenate([n.params() for n in self.neurons])
    
    def set_params(self, params: np.ndarray):
        for i, neuron in enumerate(self.neurons):
            neuron.set_params(params[i*3:(i+1)*3])


class MoireNet:
    """
    Complete Moiré Neural Network.
    
    Architecture: One or more layers of moiré neurons.
    Forward pass: Input coordinates → grid interference → classification.
    Learning: Evolutionary optimization of grid parameters.
    
    Total parameters: N_neurons × 3 (freq, angle, phase each)
    Compare: MLP with same neurons would need N² weights.
    """
    
    def __init__(self, layer_sizes: List[int] = [3],
                 freq_range: Tuple[float, float] = (3, 30)):
        self.layers = [MoireLayer(n, freq_range) for n in layer_sizes]
    
    def predict(self, x: float, y: float) -> int:
        """
        Classify point (x, y) using moiré interference.
        
        For single-layer: direct parity of neuron grids.
        For multi-layer: cascade interference patterns.
        """
        if len(self.layers) == 1:
            return int(self.layers[0].interference(x, y))
        
        # Multi-layer: each layer's interference feeds the next
        # as a frequency modulation
        val = 0.0
        for layer in self.layers:
            val = layer.interference(x + val * 0.1, y + val * 0.1)
        return int(val)
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict for array of (x, y) points."""
        return np.array([self.predict(x, y) for x, y in X])
    
    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers)
    
    def get_params(self) -> np.ndarray:
        return np.concatenate([layer.get_params() for layer in self.layers])
    
    def set_params(self, params: np.ndarray):
        offset = 0
        for layer in self.layers:
            n = layer.param_count()
            layer.set_params(params[offset:offset+n])
            offset += n


class MoireEvolver:
    """
    Evolutionary optimizer for MoiréNet.
    
    This is the "grid remodeling" — analogous to biological
    synaptic plasticity where receptor distributions change
    on the membrane surface.
    
    Uses simple (μ+λ) evolution strategy:
    - Mutate grid parameters (freq, angle, phase)
    - Keep mutations that improve fitness
    - No gradient needed — works on discrete/non-differentiable objectives
    """
    
    def __init__(self, net: MoireNet, 
                 mutation_rate: float = 0.5,
                 freq_mutation: float = 2.0,
                 angle_mutation: float = 0.3,
                 phase_mutation: float = 0.5):
        self.net = net
        self.mutation_rate = mutation_rate
        self.freq_mutation = freq_mutation
        self.angle_mutation = angle_mutation
        self.phase_mutation = phase_mutation
        self.best_params = net.get_params().copy()
        self.best_fitness = -np.inf
    
    def mutate(self) -> np.ndarray:
        """Create a mutated copy of current best parameters."""
        params = self.best_params.copy()
        n_params = len(params)
        
        for i in range(0, n_params, 3):
            if np.random.random() < self.mutation_rate:
                # Mutate freq
                params[i] += np.random.randn() * self.freq_mutation
                params[i] = np.clip(params[i], 2.0, 40.0)
            if np.random.random() < self.mutation_rate:
                # Mutate angle
                params[i+1] += np.random.randn() * self.angle_mutation
            if np.random.random() < self.mutation_rate:
                # Mutate phase
                params[i+2] += np.random.randn() * self.phase_mutation
        
        return params
    
    def step(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
        """
        One evolution step.
        
        Args:
            X: Input points, shape (N, 2)
            y: Target labels, shape (N,)
        
        Returns:
            (accuracy, improved): Current accuracy and whether we improved
        """
        # Try mutation
        candidate = self.mutate()
        self.net.set_params(candidate)
        predictions = self.net.predict_batch(X)
        fitness = np.mean(predictions == y)
        
        improved = False
        if fitness >= self.best_fitness:
            self.best_fitness = fitness
            self.best_params = candidate.copy()
            improved = True
        else:
            # Revert
            self.net.set_params(self.best_params)
        
        return float(fitness), improved
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              max_generations: int = 10000,
              target_accuracy: float = 1.0,
              verbose: bool = True) -> dict:
        """
        Train the MoiréNet via evolutionary grid remodeling.
        
        Returns dict with training history.
        """
        history = {
            'generation': [],
            'accuracy': [],
            'best_accuracy': [],
            'params': []
        }
        
        for gen in range(max_generations):
            acc, improved = self.step(X, y)
            
            if gen % 500 == 0 or improved and self.best_fitness > history.get('prev_best', 0):
                history['generation'].append(gen)
                history['accuracy'].append(acc)
                history['best_accuracy'].append(self.best_fitness)
                
                if verbose and gen % 1000 == 0:
                    print(f"  Gen {gen:6d}: accuracy={self.best_fitness:.2%} "
                          f"params={self.best_params[:6].round(2)}")
                
                history['prev_best'] = self.best_fitness
            
            if self.best_fitness >= target_accuracy:
                if verbose:
                    print(f"  SOLVED at gen {gen}: accuracy={self.best_fitness:.2%}")
                break
        
        # Ensure best params are loaded
        self.net.set_params(self.best_params)
        
        history['final_accuracy'] = self.best_fitness
        history['total_generations'] = gen + 1
        history['total_params'] = self.net.param_count()
        
        return history


# ============================================================
# Utility: Effective weight computation
# ============================================================

def compute_effective_weight(neuron_a: MoireNeuron, neuron_b: MoireNeuron,
                              resolution: int = 64) -> float:
    """
    Compute the effective "weight" between two neurons.
    
    This is the spatial correlation of their grid patterns —
    the moiré interference strength. Not stored, computed
    from geometry alone.
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    
    grid_a = neuron_a.grid_value_batch(xx_flat, yy_flat)
    grid_b = neuron_b.grid_value_batch(xx_flat, yy_flat)
    
    # Pearson correlation = effective weight
    mean_a = np.mean(grid_a)
    mean_b = np.mean(grid_b)
    cov = np.mean((grid_a - mean_a) * (grid_b - mean_b))
    std_a = np.std(grid_a)
    std_b = np.std(grid_b)
    
    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0
    
    return float(cov / (std_a * std_b))


def moire_frequency(neuron_a: MoireNeuron, neuron_b: MoireNeuron) -> float:
    """
    The moiré frequency between two neurons.
    This is the beat frequency — the low-frequency envelope
    that emerges from superposing two different grid frequencies.
    Analogous to the "difference tone" in acoustics.
    """
    return abs(neuron_a.freq - neuron_b.freq)
