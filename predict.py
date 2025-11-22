import sys
import torch
import numpy as np

# ---- Load model + scaling data ----
checkpoint = torch.load("sis_model_with_time_fixed.pth", weights_only=False)

from train import SISNet
model = SISNet()
model.load_state_dict(checkpoint["model_state"])
model.eval()

beta_min  = checkpoint["beta_min"]
beta_max  = checkpoint["beta_max"]
gamma_min = checkpoint["gamma_min"]
gamma_max = checkpoint["gamma_max"]


def predict(gens, population, i25, i75, i10=None, i60=None, i50=None, i100=None):

    # ------------------------------
    # If only two points provided, reconstruct others (not ideal, but works)
    # ------------------------------
    if i10 is None:   i10 = i25
    if i50 is None:   i50 = (i25 + i75) / 2
    if i60 is None:   i60 = i75
    if i100 is None:  i100 = i75

    # ------------------------------
    # MATCH TRAINING NORMALIZATION
    # ------------------------------

    gen_norm = gens / 500.0
    pop_norm = population / 20000.0

    # proportions
    i25_norm  = i25  / (population + 1e-6)
    i75_norm  = i75  / (population + 1e-6)
    i10_norm  = i10  / (population + 1e-6)
    i60_norm  = i60  / (population + 1e-6)
    i50_norm  = i50  / (population + 1e-6)
    i100_norm = i100 / (population + 1e-6)

    X = np.array([[
        gen_norm,
        pop_norm,
        i25_norm,
        i75_norm,
        i10_norm,
        i60_norm,
        i50_norm,
        i100_norm
    ]], dtype=np.float32)

    X_tensor = torch.tensor(X)
    scaled_output = model(X_tensor).detach().numpy()[0]

    beta_scaled, gamma_scaled = scaled_output

    # unscale back to real β, γ
    beta  = beta_scaled  * (beta_max  - beta_min)  + beta_min
    gamma = gamma_scaled * (gamma_max - gamma_min) + gamma_min

    R0 = beta / gamma if gamma > 0 else float('inf')

    print(f"Pred β = {beta:.6f}")
    print(f"Pred γ = {gamma:.6f}")
    print(f"Pred R₀ = {R0:.3f}")


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 predict.py <gens> <population> <I(t1)> <I(t2)>")
        sys.exit(1)

    gens = float(sys.argv[1])
    pop = float(sys.argv[2])
    i25 = float(sys.argv[3])
    i75 = float(sys.argv[4])

    predict(gens, pop, i25, i75)
