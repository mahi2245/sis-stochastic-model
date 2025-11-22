import sys
import numpy as np
import os

X = int(sys.argv[1])  # Number of simulations to generate
data = []

# use 6 time points like your training script expects
time_points = [0.25, 0.75, 0.10, 0.60, 0.50, 1.0]

# realistic epidemic parameter ranges
POP_MIN = 500
POP_MAX = 20000

GEN_MIN = 50
GEN_MAX = 500

GAMMA_MIN = 0.05
GAMMA_MAX = 0.5

R0_MIN = 0.2
R0_MAX = 6.0

for i in range(X):
    print(f"running {i}")

    # ----- sample gamma and target R0 -----
    gamma = np.random.uniform(GAMMA_MIN, GAMMA_MAX)
    R0_target = np.random.uniform(R0_MIN, R0_MAX)
    beta = R0_target * gamma

    # cap beta so it behaves like a probability
    beta = min(beta, 0.95)

    # ----- sample population and initial conditions -----
    N = np.random.randint(POP_MIN, POP_MAX + 1)

    # initial infected = 1%–10% of population
    I0 = max(1, int(np.random.uniform(0.01, 0.10) * N))
    S0 = N - I0

    generations = np.random.randint(GEN_MIN, GEN_MAX)

    # run full simulation first to ensure it doesn't fail
    result = os.popen(
        f'python3 simple.py --S {S0} --I {I0} --beta {beta} '
        f'--gamma {gamma} --generations {generations} '
        f'--sample 1 --minI 0'
    ).read().strip().split()

    if len(result) != 4:
        continue

    infected_vals = []

    # now run subsimulations for each time point
    for tp in time_points:
        t = max(1, int(tp * generations))

        rc = os.popen(
            f'python3 simple.py --S {S0} --I {I0} --beta {beta} '
            f'--gamma {gamma} --generations {t} '
            f'--sample 1 --minI 0'
        ).read().strip().split()

        if len(rc) != 4:
            infected_vals = []
            break

        infected_vals.append(float(rc[3]))

    if len(infected_vals) == len(time_points):
        row = [generations, N] + infected_vals + [beta, gamma]
        data.append(row)

header = (
    "generations,population,"
    + ",".join([f"infected_t{int(tp*100)}" for tp in time_points])
    + ",beta,gamma"
)

np.savetxt(
    "training_data_with_time.csv",
    data,
    delimiter=",",
    header=header,
    comments=""
)
