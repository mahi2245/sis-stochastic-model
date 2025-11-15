import sys
import numpy as np
import os

X = int(sys.argv[1])  # Number of trials
data = []

# choose time points
time_points = [0.25, 0.75, 0.1, 0.6, 0.5, 1.0]   # change to [0.10, 0.60] if needed

for i in range(X):
    print(f"running {i}")
    beta = np.random.uniform(0, 1)
    gamma = np.random.uniform(0, 1)
    generations = np.random.randint(50, 500)
    sample = np.random.randint(5, 50)

    result = os.popen(
        f'python3 simple.py --S 90 --I 10 --beta {beta} '
        f'--gamma {gamma} --generations {generations} '
        f'--sample {sample} --minI 0'
    ).read().strip().split()

    if len(result) == 4:
        beta_out, gamma_out, total_pop, infected_last = map(float, result)

        times = [int(tp * generations) for tp in time_points]

        infected_vals = []
        for t in times:
            rc = os.popen(
                f'python3 simple.py --S 90 --I 10 --beta {beta} '
                f'--gamma {gamma} --generations {t} --sample {sample} '
                f'--minI 0'
            ).read().strip().split()

            if len(rc) != 4:
                break

            infected_vals.append(float(rc[3]))

        if len(infected_vals) == len(time_points):
            data.append([generations, sample] + infected_vals + [beta, gamma])

# Save output
header = "generations,sample," + \
         ",".join([f"infected_t{int(tp*100)}" for tp in time_points]) + \
         ",beta,gamma"

np.savetxt("training_data_with_time.csv", data, delimiter=",",
           header=header, comments="")
