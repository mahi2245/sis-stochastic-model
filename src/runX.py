"""
runX.py
-------
Generate synthetic SIR training data via parallel simulation.

Usage (from project root):
    python src/runX.py <number_of_simulations>

Example:
    python src/runX.py 5000

For each simulation, random epidemic parameters are sampled, a full SIR
run is executed, and then 6 sub-simulations are run at fractional
timepoints to record infected/recovered counts. The result is written to:
    data/training_data_with_time.csv

Output CSV columns:
    generations, population,
    infected_t25, infected_t75, infected_t10, infected_t60, infected_t50, infected_t100,
    recovered_t25, recovered_t75, recovered_t10, recovered_t60, recovered_t50, recovered_t100,
    beta, gamma
"""

import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from simple import simulate_sir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fractional timepoints at which infected/recovered counts are recorded.
# Each entry t means we simulate up to floor(t * generations) generations.
TIME_POINTS = [0.25, 0.75, 0.10, 0.60, 0.50, 1.0]

# Parameter sampling ranges
POP_MIN,   POP_MAX   = 500,  20_000
GEN_MIN,   GEN_MAX   = 50,   500
GAMMA_MIN, GAMMA_MAX = 0.05, 0.5
R0_MIN,    R0_MAX    = 0.2,  6.0

OUTPUT_PATH = "data/training_data_with_time.csv"


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess via multiprocessing.Pool)
# ---------------------------------------------------------------------------

def run_single_sim(i):
    """
    Run one full SIR simulation and record snapshot counts at each timepoint.

    Parameters
    ----------
    i : int
        Simulation index (used only for progress logging).

    Returns
    -------
    list or None
        A row of features + labels:
            [generations, population,
             infected_t25, ..., infected_t100,
             recovered_t25, ..., recovered_t100,
             beta, gamma]
        Returns None if any sub-simulation raises an exception (row is
        dropped from the dataset).
    """
    print(f"[worker] starting simulation {i}")

    # --- Sample random epidemic parameters ---
    gamma       = np.random.uniform(GAMMA_MIN, GAMMA_MAX)
    R0          = np.random.uniform(R0_MIN, R0_MAX)
    beta        = min(R0 * gamma, 0.95)          # cap beta below 1
    N           = np.random.randint(POP_MIN, POP_MAX + 1)
    I0          = max(1, int(np.random.uniform(0.01, 0.10) * N))
    generations = np.random.randint(GEN_MIN, GEN_MAX)

    # --- Full simulation (validates parameters produce a valid outbreak) ---
    try:
        simulate_sir(n=N, m=I0, beta=beta, gamma=gamma, generations=generations, x=0)
    except Exception:
        return None

    # --- Sub-simulations at each fractional timepoint ---
    infected_vals  = []
    recovered_vals = []

    for tp in TIME_POINTS:
        t = max(1, int(tp * generations))
        try:
            hist_tp, _ = simulate_sir(n=N, m=I0, beta=beta, gamma=gamma, generations=t, x=0)
        except Exception:
            return None
        last = hist_tp[-1]
        infected_vals.append(int(np.sum(last == 1)))
        recovered_vals.append(int(np.sum(last == 2)))

    if len(infected_vals) != len(TIME_POINTS) or len(recovered_vals) != len(TIME_POINTS):
        return None

    return [generations, N] + infected_vals + recovered_vals + [beta, gamma]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/runX.py <number_of_simulations>")
        sys.exit(1)

    X = int(sys.argv[1])
    workers = cpu_count()
    print(f"Generating {X} simulations using {workers} cores...")

    with Pool(workers) as pool:
        results = pool.map(run_single_sim, range(X))

    results = [r for r in results if r is not None]
    print(f"Successful simulations: {len(results)} / {X}")

    # Build CSV header to match TIME_POINTS order
    infected_cols  = ",".join(f"infected_t{int(tp*100)}"  for tp in TIME_POINTS)
    recovered_cols = ",".join(f"recovered_t{int(tp*100)}" for tp in TIME_POINTS)
    header = f"generations,population,{infected_cols},{recovered_cols},beta,gamma"

    np.savetxt(
        OUTPUT_PATH,
        results,
        delimiter=",",
        header=header,
        comments="",
    )
    print(f"Saved training data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()