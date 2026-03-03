# Multi-Agent Pathfinding: Pivot and Destination Algorithms

This repository contains the source code, execution scripts, maps, and results for a research project evaluating algorithms for multi-agent pathfinding scenarios. The project specifically compares a fast existence algorithm (Algorithm 1) against an optimal approach across different cases, evaluating execution time and makespan.

## 📂 Repository Structure

```text
📦 Project Root
 ┣ 📂 Algorithm       # Source code, execution scripts, and result plots
 ┣ 📂 Case1           # Scenarios and results for the first experiment
 ┣ 📂 Case2           # Results and comparisons for the second experiment (Warehouse)
 ┗ 📂 Map             # Map files used across all experiments


## 💻 Source Code (/Algorithm)

This directory contains the core implementation of the algorithms discussed in the paper, the scripts used to run the experiments, and the plots summarizing the data.

Pivot.py & PivotOptimal.py: Core implementations of the pivot algorithms.

Destination.py: Handles agent destination logic.

Algorithm.py: Wrapper and composition script that calls the above modules to calculate paths and metrics.

Execution Scripts:

Case1.py, Case1.bash, Case1.out: Scripts and standard output for running the first set of experiments.

Case2.py, Case2.bash, Case2.out: Scripts and standard output for running the second set of experiments.

Result Plots (PDFs):

1_exec_time_agents.pdf

2_exec_time_density.pdf

3_exec_time_distance.pdf

4_exec_time_warehouse.pdf

5_makespan_warehouse.pdf

## 🗺️ Maps (/Map)

Contains the raw map files used to generate the scenarios for both Case 1 and Case 2.

## 📊 Experiment 1 (/Case1)

This folder contains the setup and results for the first experimental case, testing various agents across different distances.

/scenarios
Contains the input configurations. The structure is: map_name/distance/number_of_agents.txt.
Each file defines the environment, the pivot point, and the starting coordinates for each agent.
Example (10.txt):

Plaintext
map
random512-10-0.map
pivot
254 254
agent & start
510 498
...
/results
Contains the output for the corresponding scenarios. The structure mirrors the scenarios folder: map_name/distance/res_number_of_agents.txt.
Each file outputs the safety status, the logical time steps taken by each agent to reach the pivot, and the maximum steps required.

execution_times.csv: A summary file located in the root of the results folder containing execution metrics (e.g., group;sub;agents;exec_time_half;max_piv_step). Note: Decimal values in the data use commas (e.g., 341,75).

## 🏭 Experiment 2 (/Case2)

This folder contains the results of the second experiment, which directly compares the fast existence algorithm (Algorithm 1) against the optimal algorithm within a warehouse context.

Subfolders by Map: Each subfolder represents a map and contains the raw output files for the agents:

Existance.txt: Results for Algorithm 1. Demonstrates very fast execution times (e.g., ≈15,79s) but results in a significantly higher makespan.

Optimal.txt: Results for the Optimal Algorithm. Demonstrates a minimized makespan (e.g., 5571 steps) at the cost of higher execution time (e.g., ≈98,61s).

warehouse_comparison.csv: A comprehensive summary file comparing the two algorithms across different node scales.

Format: nodes;exec_1;exec_2;makespan_1;makespan_2
