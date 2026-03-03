# SIS/SIR Stochastic Epidemic Model

A neural network pipeline for estimating epidemic parameters (β, γ, R₀) from
stochastic SIR simulation data, with uncertainty quantification via bootstrap
and jackknife resampling methods.

---

## Project Structure

```
sis-stochastic-model/
├── src/
│   ├── simple.py            # Core stochastic simulators (SIS + SIR)
│   ├── runX.py              # Generate synthetic SIR training data
│   ├── train.py             # Define SIRNet and train it
│   ├── predict.py           # Predict β, γ, R₀ from a single observation
│   └── verify_predictions.py# (WIP) Visual verification of predictions
│
├── experiments/
│   ├── bootstrap_parametric.py          # Parametric bootstrap (SIS model)
│   ├── traditional_bootstrap.py         # Non-parametric bootstrap (SIS model)
│   ├── jackknife.py                     # Delete-d jackknife (SIS model)
│   ├── visualize_bootstrap.py           # Visualize parametric bootstrap results
│   ├── visualize_traditional_bootstrap.py
│   └── visualize_jackknife.py
│
├── data/
│   ├── training_data_with_time.csv      # Generated SIR training data
│   └── training_data_with_time_real.csv # Real-world or curated SIS data
│
├── models/
│   ├── sir_model_best.pth   # Best SIRNet checkpoint
│   └── sis_model_best.pth   # Best SISNet checkpoint (used by experiments)
│
├── results/
│   ├── bootstrap/
│   ├── jackknife/
│   └── traditional_bootstrap/
│
├── archive/                 # Legacy SIS prototype scripts (not part of pipeline)
│
└── pipeline.txt             # Detailed technical pipeline notes
```

---

## Core Pipeline (SIR Model)

All commands should be run from the project root (`sis-stochastic-model/`).

### Step 1 — Generate Synthetic Training Data

```bash
python src/runX.py <number_of_simulations>
```

Example:
```bash
python src/runX.py 5000
```

Runs `<number_of_simulations>` parallel stochastic SIR simulations with randomly
sampled parameters. Records infected and recovered counts at 6 fractional
timepoints per simulation.

**Output:** `data/training_data_with_time.csv`

---

### Step 2 — Train the Model

```bash
python src/train.py
```

Trains `SIRNet` (a 14→2 feedforward network) on the generated CSV. Uses an
80/20 train/validation split and saves the checkpoint with the lowest
validation loss.

**Input:** `data/training_data_with_time.csv`  
**Output:** `models/sir_model_best.pth`

---

### Step 3 — Predict Parameters

```bash
python src/predict.py <gens> <population> <I25> <I75> <R25> <R75>
```

Example:
```bash
python src/predict.py 200 5000 120 340 80 210
```

Loads the trained checkpoint and predicts β, γ, and R₀ from a single
epidemic observation. Only the t25 and t75 timepoints are required —
the remaining four are approximated internally.

**Input:** `models/sir_model_best.pth`  
**Output:** Prints predicted β, γ, and R₀ to stdout.

---

## Experiments (SIS Model — Uncertainty Quantification)

These scripts estimate uncertainty around SISNet predictions using three
resampling methods. They operate on the SIS model and checkpoint
(`models/sis_model_best.pth`), independently of the SIR pipeline above.

### Parametric Bootstrap

Generates B independent SIS replicates under fixed true parameters and
collects the distribution of predictions to form a 95% CI.

```bash
python experiments/bootstrap_parametric.py \
    --beta 0.4 --gamma 0.2 --pop 5000 --gens 200 --B 500
```

**Output:** `results/bootstrap/bootstrap_results.csv`

```bash
python experiments/visualize_bootstrap.py results/bootstrap/bootstrap_results.csv
```

**Output:** `results/bootstrap/bootstrap_results_plot.png`

---

### Traditional (Non-Parametric) Bootstrap

Holds out a fixed test set, trains B models on bootstrap resamples of the
training data, and computes per-sample 95% CIs.

```bash
python experiments/traditional_bootstrap.py --B 20 --epochs 200
```

**Output:** `results/traditional_bootstrap/traditional_bootstrap_results.csv`

```bash
python experiments/visualize_traditional_bootstrap.py \
    results/traditional_bootstrap/traditional_bootstrap_results.csv
```

**Output:** `results/traditional_bootstrap/traditional_bootstrap_results_plot.png`

---

### Delete-d Jackknife

Iteratively holds out d samples, retrains on the remainder, and records
prediction errors to estimate bias and variance.

```bash
python experiments/jackknife.py --delete 10 --epochs 500
```

**Output:** `results/jackknife/jackknife_results.csv`

```bash
python experiments/visualize_jackknife.py results/jackknife/jackknife_results.csv
```

**Output:** `results/jackknife/jackknife_results_plot.png`

---

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- pandas
- scikit-learn
- matplotlib
- scipy
