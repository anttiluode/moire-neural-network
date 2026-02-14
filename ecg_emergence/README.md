# ECG Emergence — Homeostatic Feedback Loop

These files define a PerceptionLab graph that spontaneously produces
ECG-like oscillations from a checkerboard pattern + homeostatic coupler.

## Requirements
- PerceptionLab v9.1+ (https://github.com/anttiluode/PerceptionLab)

## Files
- `ecg.json` — Graph configuration (load via PerceptionLab's Load button)
- `checkerboardnode.py` — Spatial frequency generator (place in `nodes/`)
- `coupler.py` — Homeostatic regulator, edge-of-chaos mode
- `imagetovectornode.py` — Sampling operator (image → 1D vector)
- `vectorsplitternode.py` — Vector decomposition to individual signals
- `constantsignalnode.py` — Baseline signal source

## How to Reproduce
1. Place all .py files in PerceptionLab's `nodes/` directory
2. Start PerceptionLab
3. Load `ecg.json`
4. Click Start
5. Observe the ECG-like oscillation on the Homeostatic Coupler's history plot

## Key Experiment: Vary Vector Dimension
Change ImageToVector's `output_dim`:
- 128: blunt, robotic pulse
- 256: rich, organic ECG (Nyquist-critical)
- 1024: binary blinking (oversampled)
- 2048: collective heartbeat returns
