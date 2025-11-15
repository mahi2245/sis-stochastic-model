import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def simulate_sis(n, m, beta, gamma, generations, x):
    """
    Simulates an SIS model with given parameters and conditions.
    
    Parameters:
        n (int): Total population size.
        m (int): Initial number of infected individuals.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        generations (int): Number of generations to simulate.
        x (int): Minimum number of infected individuals in the final generation.

    Returns:b
        history (list of np.array): List of population states over generations.
        relationships (list of np.array): Parentage relationships between generations.
    """
    attempts = 0
    max_attempts = 1000
    while attempts < max_attempts:
        population = np.zeros(n, dtype=int)
        population[:m] = 1  # Initial infection
        history = [population.copy()]
        relationships = []

        for _ in range(generations):
            new_population = np.zeros(n, dtype=int)
            parentage = np.random.choice(range(n), size=n, replace=True)
            parentage.sort()  # Sort parents for clarity
            for i in range(n):
                if population[parentage[i]] == 1:  # Parent is infected
                    if np.random.random() < gamma:  # Recovery
                        new_population[i] = 0  # Recovered
                    else:
                        new_population[i] = 1  # Remains infected
                else:  # Parent is susceptible
                    infection_probability = beta * (np.sum(population) / n)
                    if np.random.random() < infection_probability:  # Infection
                        new_population[i] = 1  # Becomes infected
                    else:
                        new_population[i] = 0  # Remains susceptible

            history.append(new_population)
            relationships.append(parentage)
            population = new_population

        # Check if the condition is met
        if np.sum(history[-1]) >= x:
            return history, relationships

        attempts += 1

    raise ValueError(f"Condition not met after {max_attempts} attempts.")


def new_equivalence_class(tsi,tsi2):
        s1 = list(set(tsi))
        s2 = list(set(tsi2))
        return len(s1) == len(s2)

def indices(mylist,value, forbidden):
    indices = [i for i, x in enumerate(mylist) if x == value]
    return [i for i in indices if i not in forbidden]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate an SIS model with conditions.")
    parser.add_argument("--S", type=int, default=16, help="Population size (S + I)")
    parser.add_argument("--I", type=int, default=4, help="Initial number of infected individuals (I)")
    parser.add_argument("--beta", type=float, default=0.6, help="Infection rate")
    parser.add_argument("--gamma", type=float, default=0.2, help="Recovery rate")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--minI", type=int, default=6, help="Minimum number of infected in the last generation")
    parser.add_argument("--sample", type=int, default=3, help="Sample size for coalescent tree")

    args = parser.parse_args()

    try:
        total = args.S + args.I
        history, relationships = simulate_sis(
            n=total,
            m=args.I,
            beta=args.beta,
            gamma=args.gamma,
            generations=args.generations,
            x=args.minI
        )
        sample = np.random.choice(range(total), size=args.sample, replace=False)
        print(f'{args.beta} {args.gamma} {len(list(history[-1]))} {np.sum(history[-1])}')
        
    except ValueError as e:
        print(e)
