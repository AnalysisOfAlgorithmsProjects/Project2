import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt



# Greedy Airport Checkpoint Coverage

def greedy_checkpoint_coverage(U, checkpoints):
    """
    GreedyCheckpointCoverage(U, C) as in the report.

    U: iterable of route IDs (e.g., 0..m-1)
    checkpoints: list of sets C_i ⊆ U

    Returns:
        selected_indices: list of chosen checkpoint indices
        full_cover: True iff all routes are covered
        runtime_ms: running time of the greedy algorithm in ms
    """
    start = time.perf_counter()

    U_rem = set(U)
    cand_indices = set(range(len(checkpoints)))
    selected_indices = []

    while U_rem and cand_indices:
        best_idx = None
        best_gain = 0

        # scan all remaining checkpoints and pick the one
        # that covers the largest number of uncovered routes
        for idx in cand_indices:
            gain = len(checkpoints[idx] & U_rem)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        # if no remaining checkpoint covers any new route, stop
        if best_gain == 0:
            break

        selected_indices.append(best_idx)
        U_rem -= checkpoints[best_idx]
        cand_indices.remove(best_idx)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    full_cover = (len(U_rem) == 0)

    return selected_indices, full_cover, elapsed_ms


# Random instance generator


def generate_random_checkpoint_instance(m,
                                        density=0.2,
                                        n_factor=1.5,
                                        seed=None):
    """
    Generate a random Airport Checkpoint Coverage instance.

    m: number of routes |U|
    density: probability that a given checkpoint monitors a given route
    n_factor: initial number of checkpoints n ≈ n_factor * m
    seed: optional RNG seed for reproducibility

    Returns:
        U: set of routes {0, ..., m-1}
        checkpoints: list of sets C_i ⊆ U
    """
    if seed is not None:
        random.seed(seed)

    U = set(range(m))
    n = int(n_factor * m)

    checkpoints = []
    for _ in range(n):
        C = set()
        for r in U:
            if random.random() < density:
                C.add(r)
        # avoid completely empty checkpoints
        if not C:
            C.add(random.choice(list(U)))
        checkpoints.append(C)

    # Ensure a full cover exists: if some routes are not covered
    # by the randomly generated checkpoints, add one more checkpoint
    # that covers all missing routes (or all routes).
    covered = set()
    for C in checkpoints:
        covered |= C
    missing = U - covered
    if missing:
        checkpoints.append(U.copy())  # final "super" checkpoint
        n += 1

    return U, checkpoints


# Experimental driver

def run_experiments(ms, trials_per_m=10):
    """
    For each m in ms, run 'trials_per_m' random instances and
    collect statistics.

    Returns:
        avg_runtime[m]: average runtime (ms)
        avg_n[m]: average number of checkpoints
        full_cover_rate[m]: fraction of runs where greedy found a full cover
    """
    runtime_sums = defaultdict(float)
    n_sums = defaultdict(float)
    full_cover_counts = defaultdict(int)

    for m in ms:
        for trial in range(trials_per_m):
            # Use a different seed each time for reproducibility if desired
            seed = 1000 * m + trial
            U, checkpoints = generate_random_checkpoint_instance(
                m, density=0.2, n_factor=1.5, seed=seed
            )
            n = len(checkpoints)

            _, full_cover, runtime_ms = greedy_checkpoint_coverage(U, checkpoints)

            runtime_sums[m] += runtime_ms
            n_sums[m] += n
            if full_cover:
                full_cover_counts[m] += 1

    avg_runtime = {}
    avg_n = {}
    full_cover_rate = {}
    for m in ms:
        avg_runtime[m] = runtime_sums[m] / trials_per_m
        avg_n[m] = n_sums[m] / trials_per_m
        full_cover_rate[m] = full_cover_counts[m] / trials_per_m

    return avg_runtime, avg_n, full_cover_rate


def print_summary(avg_runtime, avg_n, full_cover_rate):
    print("Average per m:")
    for m in sorted(avg_runtime.keys()):
        print(
            f"m = {m:3d}, "
            f"avg n = {avg_n[m]:5.1f}, "
            f"full cover rate = {full_cover_rate[m]:.2f}, "
            f"avg runtime = {avg_runtime[m]:5.3f} ms"
        )


def plot_observed_vs_theoretical(avg_runtime, avg_n):
    """
    Plot average observed runtime vs a scaled theoretical curve
    T_theory(m, n) proportional to n^2 * m.
    """
    ms = sorted(avg_runtime.keys())
    observed = [avg_runtime[m] for m in ms]

    # Theoretical values up to a constant factor: n^2 * m
    theo_raw = [(avg_n[m] ** 2) * m for m in ms]

    # Scale so the last point matches observed runtime
    scale = observed[-1] / theo_raw[-1]
    theoretical = [scale * x for x in theo_raw]

    plt.figure(figsize=(6, 4))
    plt.plot(ms, observed, marker="o", label="Observed runtime (ms)")
    plt.plot(
        ms,
        theoretical,
        marker="s",
        linestyle="--",
        label=r"Scaled theoretical $O(n^2 m)$"
    )
    plt.xlabel("Number of routes m")
    plt.ylabel("Average runtime (ms)")
    plt.title("Greedy Airport Checkpoint Coverage runtime")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# Main


if __name__ == "__main__":
    ms = [50, 100, 150, 200, 250, 300]
    trials_per_m = 10

    avg_runtime, avg_n, full_cover_rate = run_experiments(ms, trials_per_m)
    print_summary(avg_runtime, avg_n, full_cover_rate)
    plot_observed_vs_theoretical(avg_runtime, avg_n)
