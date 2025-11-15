import sys
import torch
import numpy as np
from train import SISNet

# ---- Load model ----
model = SISNet()
model.load_state_dict(torch.load("sis_model_with_time_fixed.pth"))
model.eval()

def predict(gens, sample, inf1, inf2):
    max_sample = 1000000  # TEMP — match training
    gen_scaled = gens / 12.0
    sample_scaled = sample / max_sample

    inf1_frac = inf1 / sample if sample > 0 else 0
    inf2_frac = inf2 / sample if sample > 0 else 0

    inf1_raw_scaled = inf1 / max_sample
    inf2_raw_scaled = inf2 / max_sample

    X = np.array([
        [gen_scaled, sample_scaled,
         inf1_frac, inf2_frac,
         inf1_raw_scaled, inf2_raw_scaled]
    ], dtype=np.float32)

    X_tensor = torch.tensor(X)
    output = model(X_tensor).detach().numpy()[0]
    beta, gamma = output

    print(f"Pred beta={beta:.4f}, gamma={gamma:.4f}, R0={beta/gamma:.3f}")

# ---- MAIN ----
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 predict.py <gens> <sample> <inf1> <inf2>")
        sys.exit(1)

    gens = float(sys.argv[1])
    sample = float(sys.argv[2])
    inf1 = float(sys.argv[3])
    inf2 = float(sys.argv[4])

    predict(gens, sample, inf1, inf2)
