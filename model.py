import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.integrate import solve_ivp

def simulate_sis(n, m, beta, gamma, generations):
    """Simulates an SIS model and tracks infection counts over generations."""
    population = np.zeros(n, dtype=int)
    population[:m] = 1  # Initial infection
    infection_over_time = [np.sum(population)]  # Track infections over time

    for _ in range(generations):
        new_population = np.zeros(n, dtype=int)
        for i in range(n):
            if population[i] == 1:
                new_population[i] = 0 if np.random.random() < gamma else 1  # Recovery or stays infected
            else:
                if np.random.random() < beta * (np.sum(population) / n):  # Infection probability
                    new_population[i] = 1
        population = new_population
        infection_over_time.append(np.sum(population))  # Store infection count

    return infection_over_time

def sis_ode(t, y, beta, gamma):
    """Defines the SIS model ODE system."""
    S, I = y
    dSdt = -beta * S * I + gamma * I
    dIdt = beta * S * I - gamma * I
    return [dSdt, dIdt]

def solve_sis_ode(S0, I0, beta, gamma, generations, N):
    """Solves the SIS model using an ODE solver."""
    S0 = S0 / total_population
    I0 = I0 / total_population
    t_span = [0, generations]
    y0 = [S0, I0]
    sol = solve_ivp(sis_ode, t_span, y0, args=(beta, gamma), t_eval=np.linspace(0, generations, generations))
    return sol.t, sol.y[1]  # Returns time points and infected population

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an SIS model with ODE validation.")
    parser.add_argument("--S", type=int, default=90, help="Population size (S + I)")
    parser.add_argument("--I", type=int, default=10, help="Initial number of infected individuals")
    parser.add_argument("--beta", type=float, default=0.6, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=0.2, help="Recovery rate")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")

    args = parser.parse_args()

    total_population = args.S + args.I  # Total population (S + I)
    
    # Run stochastic SIS simulation
    infection_counts = simulate_sis(total_population, args.I, args.beta, args.gamma, args.generations)

    # Solve ODE-based SIS model
    t_ode, I_ode = solve_sis_ode(args.S, args.I, args.beta, args.gamma, args.generations, total_population)

    # Plot both models
    plt.figure(figsize=(8, 5))
    plt.plot(range(args.generations + 1), infection_counts, label="Simulation (Stochastic)", color='b', alpha=0.7)
    plt.plot(t_ode, I_ode * total_population, label="ODE-based SIS Model", color='r', linestyle='dashed')

    
    plt.xlabel("Generations (Time)")
    plt.ylabel("Infected Individuals")
    plt.title("SIS Model: Stochastic vs ODE")
    plt.legend()
    plt.grid()
    plt.show()
