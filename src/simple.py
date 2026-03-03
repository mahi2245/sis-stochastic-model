"""
simple.py
---------
Core stochastic simulators for SIS and SIR epidemic models.

Both models use a Wright-Fisher-style reproduction process:
  - Each generation, N offspring are produced by sampling parents
    with replacement from the current population.
  - Disease state is inherited from the parent, then modified by
    infection or recovery probabilities.

Used by:
  - src/runX.py  (data generation)
  - experiments/ (bootstrap / jackknife resampling)
"""

import numpy as np


def simulate_sis(n, m, beta, gamma, generations, x):
    """
    Simulate a stochastic SIS (Susceptible-Infected-Susceptible) model.

    Individuals are either Susceptible (0) or Infected (1). Recovered
    individuals return to the susceptible pool (no lasting immunity).

    Parameters
    ----------
    n : int
        Total population size.
    m : int
        Initial number of infected individuals.
    beta : float
        Per-contact infection probability (force of infection scales
        with prevalence: beta * I/N).
    gamma : float
        Per-generation recovery probability for infected individuals.
    generations : int
        Number of generations to simulate.
    x : int
        Minimum number of infected individuals required in the final
        generation. The simulation is retried (up to 1000 times) until
        this threshold is met.

    Returns
    -------
    history : list of np.ndarray
        Population state arrays (length n, values 0 or 1) for each
        generation, including the initial state.
    relationships : list of np.ndarray
        Parentage arrays for each generation transition; entry i gives
        the index of individual i's parent in the previous generation.

    Raises
    ------
    ValueError
        If the condition (>= x infected at end) is not met within
        1000 attempts.
    """
    max_attempts = 1000

    for _ in range(max_attempts):
        population = np.zeros(n, dtype=int)
        population[:m] = 1  # seed initial infections
        history = [population.copy()]
        relationships = []

        for _ in range(generations):
            new_population = np.zeros(n, dtype=int)
            parentage = np.random.choice(n, size=n, replace=True)
            parentage.sort()

            infection_prob = beta * (np.sum(population) / n)

            for i in range(n):
                if population[parentage[i]] == 1:  # parent infected
                    # recover with probability gamma, else stay infected
                    new_population[i] = 0 if np.random.random() < gamma else 1
                else:  # parent susceptible
                    # become infected via force of infection
                    new_population[i] = 1 if np.random.random() < infection_prob else 0

            history.append(new_population)
            relationships.append(parentage)
            population = new_population

        if np.sum(history[-1]) >= x:
            return history, relationships

    raise ValueError(
        f"SIS simulation: condition (>= {x} infected at end) not met "
        f"after {max_attempts} attempts."
    )


def simulate_sir(n, m, beta, gamma, generations, x=0):
    """
    Simulate a stochastic SIR (Susceptible-Infected-Recovered) model.

    Individuals move through three states:
      0 = Susceptible (S)
      1 = Infected    (I)
      2 = Recovered   (R)  ← permanent; recovered individuals stay recovered.

    Parameters
    ----------
    n : int
        Total population size.
    m : int
        Initial number of infected individuals.
    beta : float
        Per-contact infection probability (force of infection scales
        with prevalence: beta * I/N).
    gamma : float
        Per-generation recovery probability for infected individuals.
    generations : int
        Number of generations to simulate.
    x : int, optional
        Minimum number of infected individuals required in the final
        generation (default 0 = no filter). The simulation is retried
        (up to 1000 times) until this threshold is met.

    Returns
    -------
    history : list of np.ndarray
        Population state arrays (length n, values 0/1/2) for each
        generation, including the initial state.
    relationships : list of np.ndarray
        Parentage arrays for each generation transition.

    Raises
    ------
    ValueError
        If the condition (>= x infected at end) is not met within
        1000 attempts.
    """
    max_attempts = 1000

    for _ in range(max_attempts):
        population = np.zeros(n, dtype=int)
        population[:m] = 1  # seed initial infections
        history = [population.copy()]
        relationships = []

        for _ in range(generations):
            new_population = np.zeros(n, dtype=int)
            parentage = np.random.choice(n, size=n, replace=True)
            parentage.sort()

            I_count = np.sum(population == 1)
            infection_prob = beta * (I_count / n)

            for i in range(n):
                parent_state = population[parentage[i]]

                if parent_state == 1:    # parent infected
                    new_population[i] = 2 if np.random.random() < gamma else 1
                elif parent_state == 0:  # parent susceptible
                    new_population[i] = 1 if np.random.random() < infection_prob else 0
                else:                    # parent recovered → stays recovered
                    new_population[i] = 2

            history.append(new_population)
            relationships.append(parentage)
            population = new_population

        if np.sum(history[-1] == 1) >= x:
            return history, relationships

    raise ValueError(
        f"SIR simulation: condition (>= {x} infected at end) not met "
        f"after {max_attempts} attempts."
    )


# ---------------------------------------------------------------------------
# CLI entry point — quick manual sanity-check of the SIS simulator
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single SIS simulation and print a summary."
    )
    parser.add_argument("--S",           type=int,   default=16,  help="Susceptible count")
    parser.add_argument("--I",           type=int,   default=4,   help="Initial infected count")
    parser.add_argument("--beta",        type=float, default=0.6, help="Infection rate")
    parser.add_argument("--gamma",       type=float, default=0.2, help="Recovery rate")
    parser.add_argument("--generations", type=int,   default=10,  help="Number of generations")
    parser.add_argument("--minI",        type=int,   default=6,   help="Min infected in last generation")
    args = parser.parse_args()

    total = args.S + args.I
    try:
        history, _ = simulate_sis(
            n=total,
            m=args.I,
            beta=args.beta,
            gamma=args.gamma,
            generations=args.generations,
            x=args.minI,
        )
        final_infected = int(np.sum(history[-1]))
        print(
            f"beta={args.beta}  gamma={args.gamma}  "
            f"pop={total}  infected_final={final_infected}"
        )
    except ValueError as e:
        print(e)