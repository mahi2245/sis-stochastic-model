import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F

class SISNet(nn.Module):
    def __init__(self):
        super(SISNet, self).__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softplus(x)


if __name__ == "__main__":

    print("Training model...")

    data = pd.read_csv("training_data_with_time.csv").values

    gens = data[:, 0].astype(np.float32)
    sample = data[:, 1].astype(np.float32)
    inf1 = data[:, 2].astype(np.float32)
    inf2 = data[:, 3].astype(np.float32)
    y = data[:, 4:6].astype(np.float32)

    max_sample = sample.max()

    gen_scaled = gens / 12.0
    sample_scaled = sample / max_sample

    inf1_frac = inf1 / (sample + 1e-6)
    inf2_frac = inf2 / (sample + 1e-6)

    max_inf = max(inf1.max(), inf2.max())

    inf1_raw_scaled = inf1 / max_inf
    inf2_raw_scaled = inf2 / max_inf

    X = np.stack([
        gen_scaled, sample_scaled, inf1_frac, inf2_frac,
        inf1_raw_scaled, inf2_raw_scaled
    ], axis=1).astype(np.float32)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    model = SISNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2000):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "sis_model_with_time_fixed.pth")
    print("✅ Training complete.")
