import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from simple import simulate_sis   # <-- import directly!

X = int(sys.argv[1])  # number of simulations

time_points = [0.25, 0.75, 0.10, 0.60, 0.50, 1.0]

POP_MIN, POP_MAX = 500, 20000
GEN_MIN, GEN_MAX = 50, 500
GAMMA_MIN, GAMMA_MAX = 0.05, 0.5
R0_MIN, R0_MAX = 0.2, 6.0


def run_single_sim(i):
    print(f"[worker] starting simulation {i}")

    # ----- Sample random parameters -----
    gamma = np.random.uniform(GAMMA_MIN, GAMMA_MAX)
    R0 = np.random.uniform(R0_MIN, R0_MAX)
    beta = min(R0 * gamma, 0.95)

    N = np.random.randint(POP_MIN, POP_MAX + 1)
    I0 = max(1, int(np.random.uniform(0.01, 0.10) * N))
    S0 = N - I0
    generations = np.random.randint(GEN_MIN, GEN_MAX)

    total_pop = S0 + I0

    # ----- Full simulation -----
    try:
        history, _ = simulate_sis(
            n=total_pop,
            m=I0,
            beta=beta,
            gamma=gamma,
            generations=generations,
            x=0  # allow any last-gen infected
        )
    except Exception:
        return None

    infected_vals = []

    # ----- Subsimulations at chosen time points -----
    for tp in time_points:
        t = max(1, int(tp * generations))

        try:
            hist_tp, _ = simulate_sis(
                n=total_pop,
                m=I0,
                beta=beta,
                gamma=gamma,
                generations=t,
                x=0
            )
        except Exception:
            return None

        infected_vals.append(np.sum(hist_tp[-1]))

    # Only return valid rows
    if len(infected_vals) == len(time_points):
        return [generations, N] + infected_vals + [beta, gamma]

    return None


def main():
    workers = cpu_count()
    print(f"Using {workers} cores.")

    with Pool(workers) as pool:
        results = pool.map(run_single_sim, range(X))

    # Filter failed runs
    results = [r for r in results if r is not None]

    header = (
        "generations,population,"
        + ",".join([f"infected_t{int(tp*100)}" for tp in time_points])
        + ",beta,gamma"
    )

    np.savetxt(
        "training_data_with_time.csv",
        results,
        delimiter=",",
        header=header,
        comments=""
    )


if __name__ == "__main__":
    main()
