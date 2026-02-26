import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import Algorithm

def load_map(map_path):
    """
    Parses the .map file to extract obstacle coordinates and grid dimensions.
    Note: The pivot coordinate is no longer extracted in this function.
    """
    obstacles = set()
    width, height = 0, 0

    try:
        with open(map_path, 'r') as f:
            lines = f.readlines()

        map_started = False
        row = 0

        for line in lines:
            if line.startswith("height"):
                height = int(line.split()[1])
            if line.startswith("width"):
                width = int(line.split()[1])

            if line.startswith("map"):
                map_started = True
                continue

            if map_started and row < height:
                for col, char in enumerate(line.strip()):
                    if char in ['@']:
                        obstacles.add((col, row))
                row += 1

    except Exception as e:
        print(f"Error loading map {map_path}: {e}", flush=True)

    return obstacles, width, height

def load_scenario(scen_path):
    """
    Parses the scenario file to extract the relative map path, the pivot coordinates,
    and the initial configurations of all agents.
    """
    with open(scen_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    map_rel_path = lines[1]

    pivot = None
    try:
        pivot_idx = lines.index("pivot")
        p_parts = lines[pivot_idx + 1].split()
        pivot = (int(p_parts[0]), int(p_parts[1]))
    except ValueError:
        print(f"Warning: 'pivot' keyword not found in {scen_path}", flush=True)

    starts_idx = lines.index("agent & start")

    try:
        dest_idx = lines.index("destination")
        end_agents = dest_idx
    except ValueError:
        end_agents = len(lines)

    initial_config = {}
    agent_id_counter = 0
    for i in range(starts_idx + 1, end_agents):
        parts = lines[i].split()
        if not parts:
            continue

        if len(parts) == 2:
            initial_config[agent_id_counter] = (int(parts[0]), int(parts[1]))
            agent_id_counter += 1

        elif len(parts) >= 3:
            initial_config[int(parts[0])] = (int(parts[1]), int(parts[2]))

    return map_rel_path, pivot, initial_config

def update_csv_row(csv_path, group, sub, agents_count, exec_time, max_piv_step):
    """
    Updates the corresponding row in the CSV file if it exists;
    otherwise, appends a new row with the provided data.
    """
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("group;sub;agents;exec_time_half;max_piv_step\n")

    with open(csv_path, 'r') as f:
        lines = f.readlines()

    updated = False
    with open(csv_path, 'w') as f:
        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 3 and parts[0] == group and parts[1] == sub and parts[2] == str(agents_count):
                f.write(f"{group};{sub};{agents_count};{exec_time};{max_piv_step}\n")
                updated = True
            else:
                f.write(line)

        if not updated:
            f.write(f"{group};{sub};{agents_count};{exec_time};{max_piv_step}\n")

def process_single_file(args):
    """
    Executes the Half Algorithm (simulating up to the pivot point)
    for a given scenario file.
    """
    scenarios_root, results_root, group, sub, scen_file = args
    folder_path = os.path.join(scenarios_root, group, sub)
    scen_path = os.path.join(folder_path, scen_file)

    res_folder_path = os.path.join(results_root, group, sub)
    os.makedirs(res_folder_path, exist_ok=True)

    output_res_path = os.path.join(res_folder_path, f"res_{scen_file}")

    base_name = os.path.splitext(scen_file)[0]
    try:
        agents_count = int(base_name)
    except ValueError:
        return False

    try:
        map_rel_path, pivot, init_config = load_scenario(scen_path)
        actual_map_path = os.path.abspath(os.path.join(folder_path, "..", "..", "..", "..", "Map", map_rel_path))
        obstacles, w, h = load_map(actual_map_path)
        agent_ids = list(init_config.keys())

        is_safe, dict_piv_times, formatted_time_piv = Algorithm.run_half_algorithm(
            agent_ids, init_config, pivot, obstacles, w, h
        )

        max_piv_step = max(dict_piv_times.values()) if dict_piv_times else 0

        with open(output_res_path, 'w') as out_res:
            out_res.write(f"Safe_to_Pivot: {is_safe}\n")
            if is_safe:
                out_res.write(f"\n--- LOGICAL TIME STEPS ---\n")
                if dict_piv_times:
                    for ag_id, t_piv in sorted(dict_piv_times.items()):
                        out_res.write(f"Agent {ag_id} steps to Pivot: {t_piv}\n")
                out_res.write(f"Max Steps to Pivot: {max_piv_step}\n")
            else:
                print(f"[NOT SAFE] File {scen_path} cannot reach the pivot.", flush=True)

        global_csv = os.path.join(results_root, "execution_times.csv")
        update_csv_row(global_csv, group, sub, agents_count, formatted_time_piv, max_piv_step)

    except Exception as e:
        print(f"Error processing {scen_path}: {e}", flush=True)

    return True

def run_experiments():
    """
    Iterates through the scenarios directory and triggers the processing
    of each valid scenario file.
    """
    scenarios_root = "../Case1/scenarios"
    results_root = "../Case1/results"

    if not os.path.exists(scenarios_root):
        print(f"Scenarios folder '{scenarios_root}' not found.", flush=True)
        return

    os.makedirs(results_root, exist_ok=True)

    tasks = []
    for group in os.listdir(scenarios_root):
        group_path = os.path.join(scenarios_root, group)
        if not os.path.isdir(group_path):
            continue

        for sub in os.listdir(group_path):
            sub_path = os.path.join(group_path, sub)
            if not os.path.isdir(sub_path):
                continue

            for scen_file in os.listdir(sub_path):
                if scen_file.startswith("res_") or not scen_file.endswith(".txt"):
                    continue
                tasks.append((scenarios_root, results_root, group, sub, scen_file))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No new scenarios to execute.", flush=True)
        return

    print(f"Starting processing of {total_tasks} scenarios (Half Algorithm)...", flush=True)

    completed = 0
    for t in tasks:
        process_single_file(t)
        completed += 1
        if completed % 10 == 0 or completed == total_tasks:
            print(f"{completed}/{total_tasks} scenarios completed...", flush=True)

def generate_plots_and_stats():
    """
    Reads the generated execution times CSV, calculates statistics,
    and generates performance plots.
    """
    results_root = "../Case1/results"
    csv_path = os.path.join(results_root, "execution_times.csv")

    if not os.path.exists(csv_path):
        print("No CSV file found to generate plots.", flush=True)
        return

    df = pd.read_csv(csv_path, sep=';')

    # Internal conversion: replacing commas with dots for valid Python numeric processing
    df['exec_time_half'] = pd.to_numeric(df['exec_time_half'].astype(str).str.replace(',', '.'), errors='coerce')
    df['max_piv_step'] = pd.to_numeric(df['max_piv_step'], errors='coerce')
    df = df.dropna()

    # Extract obstacle density percentage (e.g., 'random512-20-0' -> 20)
    df['obstacles'] = df['sub'].apply(lambda x: int(x.split('-')[1]) if '-' in x else 0)

    # Formats the terminal output to display decimals with a comma
    fmt = lambda x: str(round(x, 6)).replace('.', ',')

    print("\n" + "="*50, flush=True)
    print("MEAN HALF EXECUTION TIMES (Terminal output with comma as decimal separator)", flush=True)
    print(f"{'Group':<10} | {'Obstacles':<10} | {'Agents':<10} | {'Exec (s)':<12} | {'Max Piv'}", flush=True)
    print("-" * 50, flush=True)

    grouped_print = df.groupby(['group', 'obstacles', 'agents']).mean(numeric_only=True).reset_index()
    for _, row in grouped_print.iterrows():
        print(f"{row['group']:<10} | {int(row['obstacles']):<10} | {int(row['agents']):<10} | {fmt(row['exec_time_half']):<12} | {fmt(row['max_piv_step'])}", flush=True)

    # ==========================================
    # 1. Exec Time vs Agents (by Obstacle Density)
    # ==========================================
    grouped_time_obs = df.groupby(['obstacles', 'agents'])['exec_time_half'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for obs in sorted(grouped_time_obs['obstacles'].unique()):
        subset = grouped_time_obs[grouped_time_obs['obstacles'] == obs].sort_values('agents')
        plt.loglog(subset['agents'], subset['exec_time_half'], marker='o', linestyle='-', label=f"Obstacles {obs}%")
    plt.title('Execution Time vs Number of Agents (by Map Density)')
    plt.xlabel('Number of Agents')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('1.exec_time_by_density.png')
    print("\nSaved: 1.exec_time_by_density.png", flush=True)

    # ==========================================
    # 2. Exec Time vs Agents (by Max Distance)
    # ==========================================
    grouped_time_dist = df.groupby(['group', 'agents'])['exec_time_half'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for grp in sorted(grouped_time_dist['group'].unique()):
        subset = grouped_time_dist[grouped_time_dist['group'] == grp].sort_values('agents')
        plt.loglog(subset['agents'], subset['exec_time_half'], marker='^', linestyle='-', label=f"Max Dist Bound {grp}")
    plt.title('Execution Time vs Number of Agents (by Initial Distance)')
    plt.xlabel('Number of Agents')
    plt.ylabel('Mean Execution Time (seconds)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('2.exec_time_by_distance.png')
    print("Saved: 2.exec_time_by_distance.png", flush=True)

    # ==========================================
    # 3. Pivot Cost vs Agents (by Obstacle Density)
    # ==========================================
    grouped_piv_obs = df.groupby(['obstacles', 'agents'])['max_piv_step'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for obs in sorted(grouped_piv_obs['obstacles'].unique()):
        subset = grouped_piv_obs[grouped_piv_obs['obstacles'] == obs].sort_values('agents')
        plt.loglog(subset['agents'], subset['max_piv_step'], marker='s', linestyle='-', label=f"Obstacles {obs}%")
    plt.title('Max Logical Pivot Steps vs Number of Agents (by Map Density)')
    plt.xlabel('Number of Agents')
    plt.ylabel('Mean Logical Steps (time_pivot)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('3.pivot_cost_by_density.png')
    print("Saved: 3.pivot_cost_by_density.png", flush=True)

    # ==========================================
    # 4. Pivot Cost vs Agents (by Max Distance)
    # ==========================================
    grouped_piv_dist = df.groupby(['group', 'agents'])['max_piv_step'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for grp in sorted(grouped_piv_dist['group'].unique()):
        subset = grouped_piv_dist[grouped_piv_dist['group'] == grp].sort_values('agents')
        plt.loglog(subset['agents'], subset['max_piv_step'], marker='d', linestyle='-', label=f"Max Dist Bound {grp}")
    plt.title('Max Logical Pivot Steps vs Number of Agents (by Initial Distance)')
    plt.xlabel('Number of Agents')
    plt.ylabel('Mean Logical Steps (time_pivot)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig('4.pivot_cost_by_distance.png')
    print("Saved: 4.pivot_cost_by_distance.png", flush=True)


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
