import random
import time
import csv
from collections import defaultdict
import matplotlib.pyplot as plt


# Edmonds Karp max flow

def edmonds_karp(num_nodes, edges, s, t):
    """
    Classic Edmonds Karp implementation.

    num_nodes: number of nodes in G'
    edges: list of (u, v, capacity)
    s: source index
    t: sink index

    Returns:
        flow_value: maximum s-t flow value
        flow_mat: 2D list where flow_mat[u][v] is the flow sent on edge u->v
    """
    adj = [[] for _ in range(num_nodes)]
    cap = [[0] * num_nodes for _ in range(num_nodes)]
    flow_mat = [[0] * num_nodes for _ in range(num_nodes)]

    for u, v, c in edges:
        adj[u].append(v)
        adj[v].append(u)           # reverse edge for residual graph
        cap[u][v] += c             # handle possible parallel edges

    flow = 0
    parent = [-1] * num_nodes

    while True:
        # BFS to find shortest augmenting path in residual graph
        for i in range(num_nodes):
            parent[i] = -1
        parent[s] = s
        q = [s]

        while q and parent[t] == -1:
            u = q.pop(0)
            for v in adj[u]:
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    q.append(v)
                    if v == t:
                        break

        if parent[t] == -1:
            break  # no augmenting path

        # find bottleneck capacity along the path
        path_flow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, cap[u][v])
            v = u

        # augment along the path and record flows
        v = t
        while v != s:
            u = parent[v]
            cap[u][v] -= path_flow
            cap[v][u] += path_flow
            flow_mat[u][v] += path_flow
            flow_mat[v][u] -= path_flow
            v = u

        flow += path_flow

    return flow, flow_mat


# Blood Supply Routing algorithm 

def blood_supply_routing(n, edges, B, H, supply, demand):
    """
    Implements Algorithm Blood Supply Routing from the report.

    n: number of original nodes |V|
    edges: list of (u, v, capacity) for original road network E
    B: list of blood bank node indices
    H: list of hospital node indices
    supply: dict b -> s(b)
    demand: dict h -> d(h)

    Returns:
      feasible (bool),
      totalDemand D,
      totalDelivered (max flow value),
      runtimeMs (total runtime of the algorithm),
      shipment_plan: dict mapping (u, v) in E to shipped units f(u, v)
    """
    start = time.perf_counter()

    # Step 1: compute total demand
    D = sum(demand[h] for h in H)

    # Step 2 and 3: build flow network G' with super source s and super sink t
    s = n
    t = n + 1
    num_nodes_prime = n + 2

    edges_prime = []
    # copy original edges with same capacities
    for (u, v, c) in edges:
        edges_prime.append((u, v, c))

    # add edges from super source to each bank
    for b in B:
        edges_prime.append((s, b, supply[b]))

    # add edges from each hospital to super sink
    for h in H:
        edges_prime.append((h, t, demand[h]))

    # Step 4: run max flow on G'
    value, flow_mat = edmonds_karp(num_nodes_prime, edges_prime, s, t)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    feasible = (value == D)
    total_delivered = value

    # Build shipment plan only on original road edges E
    # shipment_plan[(u, v)] = flow on edge u->v
    shipment_plan = {}
    for (u, v, c) in edges:
        shipped = max(flow_mat[u][v], 0)
        shipment_plan[(u, v)] = shipped

    return feasible, D, total_delivered, elapsed_ms, shipment_plan



# Random instance generator 

def generate_random_instance(
    n,
    edge_prob=0.08,
    supply_range=(50, 150),
    demand_range=(20, 80),
    seed=None,
):
    """
    Generate a random BSR instance with:
      - B subset of V as banks
      - H subset of V as hospitals
      - no edges into banks
      - no edges out of hospitals
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(range(n))
    random.shuffle(nodes)

    num_banks = max(1, n // 5)
    num_hosp = max(1, n // 5)

    B = nodes[:num_banks]
    H = nodes[num_banks:num_banks + num_hosp]
    middle = nodes[num_banks + num_hosp:]  

    supply = {b: random.randint(*supply_range) for b in B}
    demand = {h: random.randint(*demand_range) for h in H}

    edges = []
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if v in B:
                continue       # no edges into blood banks
            if u in H:
                continue       # no edges leaving hospitals
            if random.random() <= edge_prob:
                capacity = random.randint(10, 100)
                edges.append((u, v, capacity))

    return edges, B, H, supply, demand




def run_experiments(sizes, trials_per_size=10, csv_path="bsr_results.csv"):
    """
    For each n in sizes, run 'trials_per_size' random instances,
    record results in CSV and also return them as a list of rows.
    Each row: [n, m, totalDemand, totalDelivered, feasible, runtimeMs, shipment_plan]
    """
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["n", "m", "totalDemand", "totalDelivered", "feasible", "runtimeMs"]
        )

        for n in sizes:
            print(f"Running experiments for n = {n} ...")
            for trial_idx in range(trials_per_size):
                edges, B, H, supply, demand = generate_random_instance(n)
                m = len(edges)

                feasible, D, delivered, runtime_ms, shipment_plan = blood_supply_routing(
                    n, edges, B, H, supply, demand
                )

                row = [n, m, D, delivered, feasible, runtime_ms, shipment_plan]
                rows.append(row)
                # Only write the numeric fields to CSV
                writer.writerow([n, m, D, delivered, feasible, runtime_ms])

    return rows


def build_runtime_summary(rows):
    """
    Returns:
      avg_runtime[n]: average runtime over all runs with this n
      avg_m[n]: average number of edges m for this n
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    m_sums = defaultdict(float)

    for n, m, D, delivered, feasible, runtime_ms, shipment_plan in rows:
        n = int(n)
        m = int(m)
        runtime_ms = float(runtime_ms)
        sums[n] += runtime_ms
        m_sums[n] += m
        counts[n] += 1

    avg_runtime = {}
    avg_m = {}
    for n in sorted(counts.keys()):
        avg_runtime[n] = sums[n] / counts[n]
        avg_m[n] = m_sums[n] / counts[n]

    return avg_runtime, avg_m


def plot_observed_vs_theoretical(avg_runtime, avg_m):
    """
    Plot average observed total runtime vs a scaled theoretical curve
    T_theory(n, m) proportional to n (m + n)^2.
    """
    ns = sorted(avg_runtime.keys())
    observed = [avg_runtime[n] for n in ns]

    # raw theoretical values up to a constant factor
    theo_raw = [n * (avg_m[n] + n) ** 2 for n in ns]

    # scale so the last point matches observed runtime
    scale = observed[-1] / theo_raw[-1]
    theoretical = [scale * x for x in theo_raw]

    plt.figure(figsize=(6, 4))
    plt.plot(ns, observed, marker="o", label="Observed total runtime")
    plt.plot(ns, theoretical, marker="s", linestyle="--",
             label=r"Theoretical $n(m + n)^2$ (scaled)")
    plt.xlabel("Number of nodes n")
    plt.ylabel("Average runtime (ms)")
    plt.title("Blood Supply Routing using Edmonds-Karp")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    sizes = [20, 40, 60, 80, 100, 120, 140]
    rows = run_experiments(sizes, trials_per_size=10,
                           csv_path="bsr_results.csv")

    # Print summary results in the csv style you showed
    print("n,m,totalDemand,totalDelivered,feasible,runtimeMs")
    for n, m, D, delivered, feasible, runtime_ms, shipment_plan in rows:
        print(",".join(str(x) for x in [n, m, D, delivered, feasible, runtime_ms]))

    # Optionally, print one example shipment plan per n
    print("\nExample shipment plans (first trial for each n):")
    seen_n = set()
    for n, m, D, delivered, feasible, runtime_ms, shipment_plan in rows:
        if n in seen_n:
            continue
        seen_n.add(n)
        print(f"\n--- n = {n}, feasible = {feasible}, delivered = {delivered}, D = {D} ---")
        print("Shipment plan (u, v) -> flow:")
        for (u, v), fval in shipment_plan.items():
            if fval > 0:
                print(f"  ({u}, {v}) -> {fval}")

    avg_runtime, avg_m = build_runtime_summary(rows)
    plot_observed_vs_theoretical(avg_runtime, avg_m)
