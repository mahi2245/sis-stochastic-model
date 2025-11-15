import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sis_ode(t, y, beta, gamma, N):
    """Defines the SIS model differential equations."""
    S, I = y
    dSdt = -beta * S * I + gamma * I
    dIdt = beta * S * I - gamma * I
    return [dSdt, dIdt]

def solve_sis_ode(S0, I0, beta, gamma, generations, N):
    """Solves the SIS model ODE."""
    t_span = [0, generations]  # Time span
    y0 = [S0, I0]  # Initial conditions
    sol = solve_ivp(sis_ode, t_span, y0, args=(beta, gamma, N), t_eval=np.linspace(0, generations, generations))
    return sol.t, sol.y[1]  # Returns time points and infected population

# Example parameters
N = 100        # Total population
I0 = 10        # Initial infected individuals
S0 = N - I0    # Initial susceptible individuals
beta = 0.5     # Infection rate
gamma = 0.2    # Recovery rate
generations = 200  # Simulation length

# Solve the ODE model
t, I_ode = solve_sis_ode(S0, I0, beta, gamma, generations, N)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(t, I_ode, label="ODE-based SIS Model", color='r', linestyle='dashed')
plt.xlabel("Generations (Time)")
plt.ylabel("Infected Individuals")
plt.legend()
plt.title("SIS Model Verification using ODEs")
plt.grid()
plt.show()