import sys
import torch
import numpy as np

# ---- Load model + scaling data ----
checkpoint = torch.load("sir_model_best.pth", weights_only=False)

from train import SIRNet
model = SIRNet()
model.load_state_dict(checkpoint["model_state"])
model.eval()

beta_min  = checkpoint["beta_min"]
beta_max  = checkpoint["beta_max"]
gamma_min = checkpoint["gamma_min"]
gamma_max = checkpoint["gamma_max"]

def predict(gens, population,
            i25, i75,
            r25, r75,
            i10=None, i60=None, i50=None, i100=None,
            r10=None, r60=None, r50=None, r100=None):

    # ------------------------------
    # Fill missing infected points (rough reconstruction)
    # ------------------------------
    if i10 is None:   i10 = i25
    if i50 is None:   i50 = (i25 + i75) / 2
    if i60 is None:   i60 = i75
    if i100 is None:  i100 = i75

    # ------------------------------
    # Fill missing recovered points (rough reconstruction)
    # ------------------------------
    if r10 is None:   r10 = r25
    if r50 is None:   r50 = (r25 + r75) / 2
    if r60 is None:   r60 = r75
    if r100 is None:  r100 = r75

    # ------------------------------
    # MATCH TRAINING NORMALIZATION
    # ------------------------------
    gen_norm = gens / 500.0
    pop_norm = population / 20000.0

    # proportions
    def norm(x): return x / (population + 1e-6)

    i25n, i75n, i10n, i60n, i50n, i100n = map(norm, [i25, i75, i10, i60, i50, i100])
    r25n, r75n, r10n, r60n, r50n, r100n = map(norm, [r25, r75, r10, r60, r50, r100])

    # [gen, pop, infected_cols..., recovered_cols...]
    X = np.array([[
        gen_norm,
        pop_norm,
        i10n, i25n, i50n, i60n, i75n, i100n,
        r10n, r25n, r50n, r60n, r75n, r100n
    ]], dtype=np.float32)

    print("X shape:", X.shape)

    X_tensor = torch.tensor(X)
    scaled_output = model(X_tensor).detach().numpy()[0]

    beta_scaled, gamma_scaled = scaled_output

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
    if len(sys.argv) != 7:
        print("Usage: python3 predict.py <gens> <population> <I25> <I75> <R25> <R75>  (others approximated)")
        sys.exit(1)

    gens = float(sys.argv[1])
    pop  = float(sys.argv[2])
    i25  = float(sys.argv[3])
    i75  = float(sys.argv[4])
    r25  = float(sys.argv[5])
    r75  = float(sys.argv[6])

    predict(gens, pop, i25, i75, r25, r75)
