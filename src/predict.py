"""
predict.py
----------
Load a trained SIRNet checkpoint and predict beta, gamma, and R0 from
a single epidemic observation.

Usage (from project root):
    python src/predict.py <gens> <population> <I25> <I75> <R25> <R75>

Arguments:
    gens        — number of generations observed
    population  — total population size
    I25         — infected count at 25% of generations
    I75         — infected count at 75% of generations
    R25         — recovered count at 25% of generations
    R75         — recovered count at 75% of generations

Missing timepoints (t10, t50, t60, t100) are approximated from the
provided t25 / t75 values (see predict() for details).

Reads:
    models/sir_model_best.pth

Prints:
    Predicted beta, gamma, and R0 to stdout.
"""

import sys
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH = "models/sir_model_best.pth"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

checkpoint = torch.load(MODEL_PATH, weights_only=False)

# Import SIRNet from train.py (both live in src/; Python adds src/ to sys.path
# automatically when running `python src/predict.py` from the project root).
from train import SIRNet

model = SIRNet()
model.load_state_dict(checkpoint["model_state"])
model.eval()

beta_min  = checkpoint["beta_min"]
beta_max  = checkpoint["beta_max"]
gamma_min = checkpoint["gamma_min"]
gamma_max = checkpoint["gamma_max"]


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------

def predict(gens, population,
            i25, i75,
            r25, r75,
            i10=None, i60=None, i50=None, i100=None,
            r10=None, r60=None, r50=None, r100=None):
    """
    Predict beta, gamma, and R0 from an epidemic observation.

    Required parameters correspond to infected (i) and recovered (r) counts
    at the 25th and 75th percentile timepoints. All other timepoints are
    approximated if not provided:
        t10  ← t25  (assume early trajectory similar to t25)
        t50  ← mean(t25, t75)
        t60  ← t75  (assume similar to t75)
        t100 ← t75  (assume plateau at t75)

    Parameters
    ----------
    gens : float
        Total number of generations simulated.
    population : float
        Total population size.
    i25, i75 : float
        Infected counts at 25% and 75% of generations (required).
    r25, r75 : float
        Recovered counts at 25% and 75% of generations (required).
    i10, i60, i50, i100 : float, optional
        Infected counts at 10%, 60%, 50%, 100% of generations.
    r10, r60, r50, r100 : float, optional
        Recovered counts at 10%, 60%, 50%, 100% of generations.

    Prints
    ------
    Predicted beta, gamma, and R0 to stdout.
    """
    # --- Fill missing timepoints via simple approximation ---
    if i10  is None: i10  = i25
    if i50  is None: i50  = (i25 + i75) / 2
    if i60  is None: i60  = i75
    if i100 is None: i100 = i75

    if r10  is None: r10  = r25
    if r50  is None: r50  = (r25 + r75) / 2
    if r60  is None: r60  = r75
    if r100 is None: r100 = r75

    # --- Normalize to match training convention ---
    gen_norm = gens / 500.0
    pop_norm = population / 20_000.0

    def prop(x):
        """Convert raw count to proportion of population."""
        return x / (population + 1e-6)

    # Feature order must match training column order:
    # gen, pop, infected_t10/25/50/60/75/100, recovered_t10/25/50/60/75/100
    X = np.array([[
        gen_norm, pop_norm,
        prop(i10), prop(i25), prop(i50), prop(i60), prop(i75), prop(i100),
        prop(r10), prop(r25), prop(r50), prop(r60), prop(r75), prop(r100),
    ]], dtype=np.float32)

    X_tensor = torch.tensor(X)
    scaled_output = model(X_tensor).detach().numpy()[0]
    beta_scaled, gamma_scaled = scaled_output

    # --- Unscale predictions ---
    beta  = beta_scaled  * (beta_max  - beta_min)  + beta_min
    gamma = gamma_scaled * (gamma_max - gamma_min) + gamma_min
    R0    = beta / gamma if gamma > 0 else float("inf")

    print(f"Pred β  = {beta:.6f}")
    print(f"Pred γ  = {gamma:.6f}")
    print(f"Pred R₀ = {R0:.3f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python src/predict.py "
            "<gens> <population> <I25> <I75> <R25> <R75>"
        )
        print("       (t10, t50, t60, t100 are approximated from t25/t75)")
        sys.exit(1)

    gens = float(sys.argv[1])
    pop  = float(sys.argv[2])
    i25  = float(sys.argv[3])
    i75  = float(sys.argv[4])
    r25  = float(sys.argv[5])
    r75  = float(sys.argv[6])

    predict(gens, pop, i25, i75, r25, r75)