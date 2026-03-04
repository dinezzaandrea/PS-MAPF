import os
import sys
import time
import fcntl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from glob import glob
import Algorithm

def load_map_warehouse(map_path):
    """Loads the warehouse map and identifies all free cells."""
    obstacles = set()
    free_nodes = []
    width, height = 0, 0
    try:
        with open(map_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("height"): height = int(line.split()[1])
            if line.startswith("width"): width = int(line.split()[1])
            if line.startswith("map"): break

        # Start reading the grid after the 'map' line
        map_start_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "map":
                map_start_idx = i + 1
                break

        if map_start_idx != -1:
            for r, line in enumerate(lines[map_start_idx:]):
                for c, char in enumerate(line.strip()):
                    if char == 'T':
                        obstacles.add((c, r))
                    else:
                        free_nodes.append((c, r))
    except Exception as e:
        print(f"Error loading map {map_path}: {e}", flush=True)
    return obstacles, free_nodes, width, height

def save_detailed_results(folder, filename, is_safe, dict_times, formatted_exec_time):
    """Saves detailed logs to Existance.txt or Optimal.txt."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, 'w') as f:
        f.write(f"Safe: {is_safe}\n")
        f.write(f"Execution Time (s): {formatted_exec_time}\n")
        if is_safe and dict_times:
            max_step = max(dict_times.values())
            f.write(f"Max Steps (Makespan): {max_step}\n")
            f.write("-" * 30 + "\n")
            for ag, t in sorted(dict_times.items()):
                f.write(f"Agent {ag}: {t}\n")

def update_csv_row(csv_path, map_name, nodes, exec_1, exec_2, makespan_1, makespan_2):
    """Updates the corresponding row in the CSV file safely using file locks."""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("map_name;nodes;exec_1;exec_2;makespan_1;makespan_2\n")

    with open(csv_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX) # Lock the file for exclusive access
        lines = f.readlines()
        updated = False
        new_lines = []

        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 2 and parts[0] == map_name:
                new_lines.append(f"{map_name};{nodes};{exec_1};{exec_2};{makespan_1};{makespan_2}\n")
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f"{map_name};{nodes};{exec_1};{exec_2};{makespan_1};{makespan_2}\n")

        f.seek(0)
        f.truncate()
        f.writelines(new_lines)
        fcntl.flock(f, fcntl.LOCK_UN) # Unlock the file

def run_experiments(target_path=None):
    """Runs the comparison between Algorithm 1 (General) and Algorithm 2 (Optimal)."""
    results_root = os.path.join("..", "Case2")
    os.makedirs(results_root, exist_ok=True)
    csv_path = os.path.join(results_root, "warehouse_comparison.csv")

    map_files = []

    # Determine if a specific file, a directory, or nothing was passed
    if target_path and os.path.isfile(target_path):
        map_files = [target_path]
    elif target_path and os.path.isdir(target_path):
        map_files = glob(os.path.join(target_path, "warehouse-*.map"))
    else:
        map_folder = os.path.join("..", "Map")
        map_files = glob(os.path.join(map_folder, "warehouse-*.map"))

    if not map_files:
        print(f"No maps found for target '{target_path or '../Map'}'", flush=True)
        return

    pivot = (2, 2)
    print(f"Starting experiments on {len(map_files)} warehouse maps...", flush=True)

    for m_path in map_files:
        map_name = os.path.basename(m_path)
        obstacles, free_nodes, w, h = load_map_warehouse(m_path)

        num_free = len(free_nodes)
        num_occupied = len(obstacles)
        print(f"\nMap: {map_name}")
        print(f"  - Map Size: {w}x{h}")
        print(f"  - Free Cells: {num_free}")
        print(f"  - Occupied Cells: {num_occupied}")

        # Fully Occupied configuration (k = n)
        initial_config = {i: node for i, node in enumerate(free_nodes)}
        agent_ids = list(initial_config.keys())

        output_folder = os.path.join(results_root, f"{map_name}")

        # --- Algorithm 1: Pivot Visit via Cycle Rotation ---
        safe1, times1, exec1 = Algorithm.run_half_algorithm(
            agent_ids, initial_config, pivot, obstacles, w, h
        )
        save_detailed_results(output_folder, "Existance.txt", safe1, times1, exec1)
        makespan1 = max(times1.values()) if times1 else 0

        # --- Algorithm 2: Optimal Construction ---
        safe2, times2, exec2 = Algorithm.run_half_optimal_algorithm(
            agent_ids, initial_config, pivot, obstacles, w, h
        )
        save_detailed_results(output_folder, "Optimal.txt", safe2, times2, exec2)
        makespan2 = max(times2.values()) if times2 else 0

        if safe1 and safe2:
            exec1_float = float(str(exec1).replace(',', '.'))
            exec2_float = float(str(exec2).replace(',', '.'))

            # Row-by-row update instead of accumulating and overwriting everything at the end
            update_csv_row(csv_path, map_name, num_free, exec1_float, exec2_float, makespan1, makespan2)

            # Display execution times using a comma for decimals
            print(f"Completed: {map_name} ({num_free} agents) - Exec 1: {str(exec1_float).replace('.', ',')}s, Exec 2: {str(exec2_float).replace('.', ',')}s", flush=True)

def generate_plots_and_stats():
    """Generates PDF plots for the paper (Figures 4 and 5) as grouped bar charts with log scale."""
    results_root = os.path.join("..", "Case2")
    csv_path = os.path.join(results_root, "warehouse_comparison.csv")

    if not os.path.exists(csv_path):
        print(f"Nessun file trovato in {csv_path}", flush=True)
        return

    df = pd.read_csv(csv_path, sep=';')

    # Since insertion is now asynchronous/row-by-row, sort the dataframe before plotting
    df = df.sort_values('nodes').reset_index(drop=True)

    comma_formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}".replace('.', ','))

    x = np.arange(len(df['nodes']))
    width = 0.35

    color_alg1 = 'tab:blue'
    color_alg2 = 'tab:green'

    # --- Figure 4: Execution Time comparison ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, df['exec_1'], width, label='Algorithm 1 (General)', color=color_alg1)
    ax.bar(x + width/2, df['exec_2'], width, label='Algorithm 2 (Optimal)', color=color_alg2)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Agents ($k=n$)')
    ax.set_ylabel('Mean Execution Time (s) [Log Scale]')
    ax.set_xticks(x)
    ax.set_xticklabels(df['nodes'])
    ax.yaxis.set_major_formatter(comma_formatter)
    ax.legend()
    ax.grid(True, which="both", ls='--', alpha=0.5, axis='y')

    plt.savefig('4_exec_time_warehouse.pdf', bbox_inches='tight')
    plt.close()

    # --- Figure 5: Makespan cost comparison ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, df['makespan_1'], width, label='Algorithm 1 Makespan', color=color_alg1)
    ax.bar(x + width/2, df['makespan_2'], width, label='Algorithm 2 Makespan', color=color_alg2)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Agents ($k=n$)')
    ax.set_ylabel('Makespan ($M_o$) [Log Scale]')
    ax.set_xticks(x)
    ax.set_xticklabels(df['nodes'])
    ax.yaxis.set_major_formatter(comma_formatter)
    ax.legend()
    ax.grid(True, which="both", ls='--', alpha=0.5, axis='y')

    plt.savefig('5_makespan_warehouse.pdf', bbox_inches='tight')
    plt.close()

    print("\nCharts 4_exec_time_warehouse.pdf and 5_makespan_warehouse.pdf generated in the current directory.", flush=True)

if __name__ == "__main__":
    try:
        # Get the command line argument if present
        target = sys.argv[1] if len(sys.argv) > 1 else None

        run_experiments(target)

        # Regenerate plots at the end, only if we are not processing a single file in parallel
        if target is None or os.path.isdir(target):
            generate_plots_and_stats()

    except KeyboardInterrupt:
        print("\nExecution interrupted by the user.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Critical error during execution: {e}", flush=True)
        sys.exit(1)
