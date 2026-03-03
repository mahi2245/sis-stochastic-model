import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from train import SISNet

# Load trained model
model = SISNet()
model.load_state_dict(torch.load("sis_model_with_time.pth"))
model.eval()

def predict_beta_gamma(generations, sample, infected_t1, infected_t2):
    """Runs the SIS model to predict beta and gamma values."""
    input_tensor = torch.tensor([[generations, sample, infected_t1, infected_t2]], dtype=torch.float32)
    prediction = model(input_tensor).detach().numpy()
    return prediction[0]

# Define input parameters
generations = 500  
sample_size = 50  
infected_t1 = 10  
infected_t2 = 20  

# Get prediction
beta, gamma = predict_beta_gamma(generations, sample_size, infected_t1, infected_t2)

def create_gauge(ax, value, title, color):
    """Creates a gauge-style chart using Matplotlib."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Draw semicircle background
    ax.add_patch(Wedge((0, 0), 1, 0, 180, facecolor="lightgray", edgecolor="black"))

    # Draw indicator needle
    angle = 180 * value  # Scale value (0-1) to degrees (0-180)
    x = np.cos(np.radians(180 - angle))
    y = np.sin(np.radians(180 - angle))
    ax.plot([0, x], [0, y], color=color, linewidth=3)

    # Draw center circle
    ax.add_patch(Circle((0, 0), 0.1, color="black"))

    # Labels
    ax.text(-1, -0.2, "0", fontsize=12, ha='center', va='center')
    ax.text(1, -0.2, "1", fontsize=12, ha='center', va='center')
    ax.text(0, 1, f"{value:.2f}", fontsize=14, ha='center', va='center', fontweight="bold")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Create gauges for Beta and Gamma
create_gauge(axes[0], beta, "Beta (Infection Rate)", "blue")
create_gauge(axes[1], gamma, "Gamma (Recovery Rate)", "green")

# Display the gauges
plt.tight_layout()
plt.show()
