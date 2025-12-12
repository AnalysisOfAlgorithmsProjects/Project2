COP 5536 – Analysis of Algorithms
Project 2 — Group 9
📌 Overview

This repository contains bonus experimental implementations for the two problems analyzed in our Project 2 submission:

Problem 1 — Blood Supply Routing (Polynomial-Time / Max-Flow)

We reduce a realistic blood transportation scenario to a maximum flow problem, solve it using an Edmonds–Karp–style algorithm, and measure runtime performance on randomly generated instances.

Problem 2 — Airport Checkpoint Coverage (NP-Complete / Greedy Approximation)

We prove NP-Completeness via reduction from Set Cover, then implement a greedy approximation algorithm. We also empirically study its runtime and compare it against the theoretical complexity bound.

All experiments included in the report (tables and plots) were produced using this code.

📁 Repository Structure
Project2/
│── Problem1.py        # Blood Supply Routing via Max-Flow (Bonus Implementation)
│── Problem2.py        # Airport Checkpoint Coverage Greedy Algorithm + Experiments
│── requirements.txt   # Optional Python dependencies
│── README.md          # This file

🔺 Problem 1 — Blood Supply Routing (Max-Flow)
📘 Description

This implementation follows the reduction described in the report:

Donation centers and hospitals become nodes

Transportation routes become directed edges with capacities

A super-source and super-sink encode total supply and demand

Max-flow determines whether the blood demand can be fully satisfied

A feasible routing plan is extracted from the flow

📂 File Included

Problem1.py implements:

Graph construction for the reduced network

Custom Edmonds–Karp maximum flow

Random instance generator

Runtime measurement using time.perf_counter()

Summary output used for Table I and Figure 1 in the report

▶️ Run Problem 1 Experiments
python3 Problem1.py


Example output:

Nodes=80, Edges=329, Demand=753, Delivered=742, Feasible=False, Time=1.10ms

🔺 Problem 2 — Airport Checkpoint Coverage (NP-Complete)
📘 Description

This implementation corresponds directly to the formalization in the report.
The greedy algorithm chooses, at each step, the checkpoint covering the largest number of uncovered routes.

📂 File Included

Problem2.py includes:

greedy_checkpoint_coverage() — greedy set-cover algorithm

generate_random_checkpoint_instance() — random ACC instance generator

Experiment driver over increasing numbers of routes

Collection of:

average runtime

average number of checkpoints

full-cover success rate

Plot comparing observed runtime to theoretical 
𝑂
(
𝑛
2
𝑚
)
O(n
2
m)

This code produced Table II and Figure 2 in the final report.

▶️ Run Problem 2 Experiments
python3 Problem2.py


Example summary:

m=200, avg_n=300.0, full_cover_rate=1.00, avg_runtime=4.03ms

📊 Dependencies

Install all dependencies (optional):

pip install -r requirements.txt


Minimal required packages:

matplotlib

Standard Python libraries (random, collections, time)

🧪 Reproducibility

All experiments:

Use fixed random seeds

Follow the same parameters as described in the Project 2 report

Generate the exact tables and plots included in the document

This ensures full reproducibility.
