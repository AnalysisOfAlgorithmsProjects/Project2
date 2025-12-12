Implementations for AOA Project 2
COP 5536 — Analysis of Algorithms
Project 2 — Group 9

📌 Overview

This repository contains the bonus experimental implementations for the two problems analyzed in our Project 2 submission:

Problem 1 — Blood Supply Routing (Polynomial-Time / Max-Flow)

We reduce a practical blood transportation network to a maximum-flow instance and evaluate the performance of an Edmonds–Karp–style algorithm on random synthetic graphs.

Problem 2 — Airport Checkpoint Coverage (NP-Complete / Greedy Approximation)

We show that the Airport Checkpoint Coverage problem is NP-Complete (via SET COVER) and implement the classical greedy approximation algorithm. We also empirically compare runtime growth vs. the theoretical upper bound.

All results presented in the final report (tables and plots) were generated using this code.

🔺 Problem 1 — Blood Supply Routing (Max-Flow)
📘 Description

The implementation in Problem1.py follows the reduction described in the report:

Donation centers → supply nodes

Hospitals → demand nodes

Transportation routes → directed edges with capacities

A super-source and super-sink encode total supply/demand

Edmonds–Karp is used to compute maximum flow

The script also:

Generates random test instances

Measures runtime

Prints feasibility outcomes

Produces results equivalent to Table I and Figure 1 in the report

▶️ Run Problem 1

From the project directory, execute:

python3 Problem1.py


Example output:

Nodes = 80, Edges = 329
Total Demand = 753, Delivered = 742
Feasible = False
Runtime = 1.10 ms

🔺 Problem 2 — Airport Checkpoint Coverage (NP-Complete)
📘 Description

Problem2.py implements:

greedy_checkpoint_coverage() — Greedy SET COVER algorithm

generate_random_checkpoint_instance() — Random instance generator

A complete experimental pipeline over various values of m

Aggregation of runtime statistics

Plot comparing observed runtime vs. scaled theoretical O(n²m)

This script produced Table II and Figure 2 in the final report.

▶️ Run Problem 2
python3 Problem2.py


Example output:

m = 200, avg_n = 300.0
Full cover rate = 1.00
Average runtime = 4.03 ms

📊 Dependencies

Install dependencies using:

pip install -r requirements.txt


Minimal packages required:

matplotlib

Python standard libraries (random, collections, time)


