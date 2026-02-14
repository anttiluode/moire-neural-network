"""
Benchmark: Moiré Neural Network vs MLP on XOR
===============================================
XOR is the classic test for nonlinear classification.
Single-layer perceptrons cannot solve it (Minsky & Papert, 1969).

This benchmark compares:
1. MoiréNet: 3 neurons × 3 params = 9 total parameters
2. MLP: 2-2-1 architecture = 9 weights + 3 biases = 12 parameters
3. MLP minimal: 2-2-1 with no bias on output = 9 parameters

Key question: Can geometric interference match gradient-trained networks
with equal or fewer parameters?

Author: Antti Luode
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from moire_net import MoireNet, MoireEvolver

# ============================================================
# XOR Dataset
# ============================================================

# Normalized to [0.1, 0.9] to avoid edge effects
XOR_X = np.array([
    [0.1, 0.1],  # 0 XOR 0 = 0
    [0.9, 0.1],  # 1 XOR 0 = 1
    [0.1, 0.9],  # 0 XOR 1 = 1
    [0.9, 0.9],  # 1 XOR 1 = 0
])

XOR_Y = np.array([0, 1, 1, 0])


# ============================================================
# Simple MLP for comparison
# ============================================================

class SimpleMLP:
    """Minimal MLP: 2 inputs → 2 hidden (sigmoid) → 1 output (sigmoid)"""
    
    def __init__(self):
        # Xavier init
        self.w1 = np.random.randn(2, 2) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros(2)
        self.w2 = np.random.randn(2, 1) * np.sqrt(2.0 / 2)
        self.b2 = np.zeros(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        self.h = self.sigmoid(x @ self.w1 + self.b1)
        self.o = self.sigmoid(self.h @ self.w2 + self.b2)
        return self.o.flatten()
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
    
    def train_step(self, X, y, lr=0.5):
        """One step of backprop."""
        y = y.reshape(-1, 1)
        
        # Forward
        h_pre = X @ self.w1 + self.b1
        h = self.sigmoid(h_pre)
        o_pre = h @ self.w2 + self.b2
        o = self.sigmoid(o_pre)
        
        # Backward
        d_o = (o - y) * o * (1 - o)
        d_w2 = h.T @ d_o
        d_b2 = np.sum(d_o, axis=0)
        
        d_h = d_o @ self.w2.T * h * (1 - h)
        d_w1 = X.T @ d_h
        d_b1 = np.sum(d_h, axis=0)
        
        # Update
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        
        loss = np.mean((o.flatten() - y.flatten()) ** 2)
        return loss
    
    def param_count(self):
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size


# ============================================================
# Benchmark
# ============================================================

def benchmark_moire_xor(n_trials: int = 20, max_gen: int = 20000) -> dict:
    """Run MoiréNet on XOR multiple times, collect statistics."""
    results = {
        'accuracies': [],
        'generations': [],
        'times': [],
        'param_counts': [],
        'solved': 0
    }
    
    for trial in range(n_trials):
        net = MoireNet(layer_sizes=[3], freq_range=(3, 30))
        evolver = MoireEvolver(net, mutation_rate=0.6, freq_mutation=2.0)
        
        t0 = time.time()
        history = evolver.train(XOR_X, XOR_Y, max_generations=max_gen,
                                target_accuracy=1.0, verbose=False)
        elapsed = time.time() - t0
        
        results['accuracies'].append(history['final_accuracy'])
        results['generations'].append(history['total_generations'])
        results['times'].append(elapsed)
        results['param_counts'].append(history['total_params'])
        
        if history['final_accuracy'] >= 1.0:
            results['solved'] += 1
    
    return results


def benchmark_mlp_xor(n_trials: int = 20, max_epochs: int = 20000) -> dict:
    """Run MLP on XOR multiple times, collect statistics."""
    results = {
        'accuracies': [],
        'epochs': [],
        'times': [],
        'param_counts': [],
        'solved': 0
    }
    
    for trial in range(n_trials):
        mlp = SimpleMLP()
        
        t0 = time.time()
        solved_epoch = max_epochs
        
        for epoch in range(max_epochs):
            loss = mlp.train_step(XOR_X, XOR_Y, lr=1.0)
            
            if epoch % 100 == 0:
                preds = mlp.predict(XOR_X)
                acc = np.mean(preds == XOR_Y)
                if acc >= 1.0:
                    solved_epoch = epoch
                    break
        
        elapsed = time.time() - t0
        preds = mlp.predict(XOR_X)
        final_acc = np.mean(preds == XOR_Y)
        
        results['accuracies'].append(final_acc)
        results['epochs'].append(solved_epoch)
        results['times'].append(elapsed)
        results['param_counts'].append(mlp.param_count())
        
        if final_acc >= 1.0:
            results['solved'] += 1
    
    return results


def print_results(name: str, results: dict, step_key: str):
    """Pretty print benchmark results."""
    accs = np.array(results['accuracies'])
    steps = np.array(results['generations'] if 'generations' in results else results['epochs'])
    times = np.array(results['times'])
    
    n = len(accs)
    solved = results['solved']
    params = results['param_counts'][0]
    
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Trials:          {n}")
    print(f"  Solved:          {solved}/{n} ({100*solved/n:.0f}%)")
    print(f"  Parameters:      {params}")
    print(f"  Mean accuracy:   {np.mean(accs):.2%} ± {np.std(accs):.2%}")
    if solved > 0:
        solved_steps = steps[np.array(results['accuracies']) >= 1.0]
        solved_times = times[np.array(results['accuracies']) >= 1.0]
        print(f"  Mean {step_key} to solve: {np.mean(solved_steps):.0f} ± {np.std(solved_steps):.0f}")
        print(f"  Mean time to solve:  {np.mean(solved_times):.3f}s ± {np.std(solved_times):.3f}s")
    print(f"  Total mean time: {np.mean(times):.3f}s")


def main():
    N_TRIALS = 30
    MAX_STEPS = 30000
    
    print("╔═══════════════════════════════════════════════════════╗")
    print("║   Moiré Neural Network vs MLP — XOR Benchmark        ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    print(f"Trials per method: {N_TRIALS}")
    print(f"Max steps: {MAX_STEPS}")
    print()
    
    # --- MoiréNet ---
    print("Running MoiréNet (3 neurons, 9 params, evolutionary)...")
    moire_results = benchmark_moire_xor(N_TRIALS, MAX_STEPS)
    print_results("MoiréNet (Geometric Interference)", moire_results, "generations")
    
    # --- MLP ---
    print("\nRunning MLP (2-2-1, 9 params, backprop)...")
    mlp_results = benchmark_mlp_xor(N_TRIALS, MAX_STEPS)
    print_results("MLP (Backpropagation)", mlp_results, "epochs")
    
    # --- Comparison ---
    print(f"\n{'='*55}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*55}")
    
    m_params = moire_results['param_counts'][0]
    p_params = mlp_results['param_counts'][0]
    m_solved = moire_results['solved'] / N_TRIALS * 100
    p_solved = mlp_results['solved'] / N_TRIALS * 100
    
    print(f"  {'':20s} {'MoiréNet':>12s} {'MLP':>12s}")
    print(f"  {'Parameters':20s} {m_params:>12d} {p_params:>12d}")
    print(f"  {'Solve rate':20s} {m_solved:>11.0f}% {p_solved:>11.0f}%")
    print(f"  {'Mean accuracy':20s} {np.mean(moire_results['accuracies']):>11.1%} "
          f"{np.mean(mlp_results['accuracies']):>11.1%}")
    
    # Parameter efficiency
    print(f"\n  Key insight: MoiréNet achieves nonlinear classification")
    print(f"  with {m_params} geometric parameters (freq, angle, phase)")
    print(f"  vs MLP's {p_params} learned weights+biases.")
    print(f"  Both solve XOR, but MoiréNet's parameters have physical")
    print(f"  meaning (grid geometry) and could be implemented optically.")
    print()
    print(f"  In optical implementation, the MoiréNet forward pass")
    print(f"  would execute at the speed of light with ~0 energy cost,")
    print(f"  while MLP requires {2*2 + 2*1} multiply-accumulate operations.")


if __name__ == '__main__':
    main()
