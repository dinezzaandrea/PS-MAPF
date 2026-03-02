import os
import sys
import time
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

def run_experiments():
    """Runs the comparison between Algorithm 1 (General) and Algorithm 2 (Optimal)."""
    map_folder = os.path.join("..", "Map")
    results_root = os.path.join("..", "Case2")

    # Point 1: Pattern captures warehouse-10-20-10-2-1.map and similar files
    map_files = glob(os.path.join(map_folder, "warehouse-*.map"))

    if not map_files:
        print(f"No maps found in {map_folder} with pattern 'warehouse-*.map'", flush=True)
        return

    pivot = (2, 2)
    data_summary = []

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
            data_summary.append({
                'nodes': len(free_nodes),
                'exec_1': float(exec1.replace(',', '.')),
                'exec_2': float(exec2.replace(',', '.')),
                'makespan_1': makespan1,
                'makespan_2': makespan2
            })
            print(f"Completed: {map_name} ({len(free_nodes)} agents)", flush=True)

    # Save CSV for plotting
    if data_summary:
        df = pd.DataFrame(data_summary).sort_values('nodes')
        df.to_csv(os.path.join(results_root, "warehouse_comparison.csv"), index=False, sep=';')

def generate_plots_and_stats():
    """Generates PDF plots for the paper (Figures 4 and 5)."""
    results_root = "Case2"
    csv_path = os.path.join(results_root, "warehouse_comparison.csv")

    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path, sep=';')
    # Use comma as decimal separator for labels
    comma_formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}".replace('.', ','))

    # Figure 4: Execution Time comparison
    plt.figure(figsize=(6, 4))
    plt.plot(df['nodes'], df['exec_1'], 'o-', label='Algorithm 1 (General)', color='tab:blue')
    plt.plot(df['nodes'], df['exec_2'], 's-', label='Algorithm 2 (Optimal)', color='tab:green')
    plt.xlabel('Number of Agents ($k=n$)')
    plt.ylabel('Mean Execution Time (s)')
    plt.gca().yaxis.set_major_formatter(comma_formatter)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.savefig('4_exec_time_warehouse.pdf', bbox_inches='tight')
    plt.close()

    # Figure 5: Makespan cost comparison
    plt.figure(figsize=(6, 4))
    plt.plot(df['nodes'], df['makespan_1'], 'o-', label='Algorithm 1 Makespan', color='tab:red')
    plt.plot(df['nodes'], df['makespan_2'], 's-', label='Algorithm 2 Makespan', color='tab:orange')
    plt.xlabel('Number of Agents ($k=n$)')
    plt.ylabel('Makespan ($M_o$)')
    plt.gca().yaxis.set_major_formatter(comma_formatter)
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.savefig('5_makespan_warehouse.pdf', bbox_inches='tight')
    plt.close()

    print("\nCharts 4_exec_time_warehouse.pdf and 5_makespan_warehouse.pdf generated.", flush=True)

# Point 2: Main entry point with exception handling
if __name__ == "__main__":
    try:
        run_experiments()
        generate_plots_and_stats()
    except KeyboardInterrupt:
        print("\nExecution interrupted by the user.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Critical error during execution: {e}", flush=True)
        sys.exit(1)
