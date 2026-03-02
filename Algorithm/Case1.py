import os
import sys
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fcntl
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

def update_csv_row(csv_path, distanza, mappa, agents_count, exec_time, max_piv_step):
    """
    Updates the corresponding row in the CSV file safely using file locks.
    """
    # If the file does not exist, prepare it first
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("group;sub;agents;exec_time_half;max_piv_step\n")

    # Open in read/write mode and apply an exclusive lock
    with open(csv_path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX) # Lock the file for other jobs

        lines = f.readlines()
        updated = False
        new_lines = []

        for line in lines:
            parts = line.strip().split(';')
            if len(parts) >= 3 and parts[0] == str(distanza) and parts[1] == str(mappa) and parts[2] == str(agents_count):
                new_lines.append(f"{distanza};{mappa};{agents_count};{exec_time};{max_piv_step}\n")
                updated = True
            else:
                new_lines.append(line)

        if not updated:
            new_lines.append(f"{distanza};{mappa};{agents_count};{exec_time};{max_piv_step}\n")

        # Rewrite the file from the beginning
        f.seek(0)
        f.truncate()
        f.writelines(new_lines)

        fcntl.flock(f, fcntl.LOCK_UN) # Unlock the file

def process_single_file(args):
    """
    Executes the Half Algorithm (simulating up to the pivot point)
    for a given scenario file.
    """
    scenarios_root, results_root, mappa, distanza, scen_file = args
    folder_path = os.path.join(scenarios_root, mappa, distanza)
    scen_path = os.path.join(folder_path, scen_file)

    res_folder_path = os.path.join(results_root, mappa, distanza)
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
        # Passing 'distanza' and 'mappa' in this order to maintain the 'group;sub' format of the CSV
        update_csv_row(global_csv, distanza, mappa, agents_count, formatted_time_piv, max_piv_step)

    except Exception as e:
        print(f"Error processing {scen_path}: {e}", flush=True)

    return True

def get_scenario_info(filepath):
    """Extracts map, distance, and filename based on the folder structure."""
    dir1 = os.path.dirname(filepath)
    distanza = os.path.basename(dir1)
    dir2 = os.path.dirname(dir1)
    mappa = os.path.basename(dir2)
    scen_file = os.path.basename(filepath)
    scenarios_root = os.path.dirname(dir2)
    return scenarios_root, mappa, distanza, scen_file

def run_experiments(target_path=None):
    """
    If target_path is specified, analyzes only that file or folder.
    Otherwise, uses the default folder.
    """
    default_root = "../Case1/scenarios"
    results_root = "../Case1/results"

    target_path = target_path or default_root

    if not os.path.exists(target_path):
        print(f"Path '{target_path}' not found.", flush=True)
        return

    os.makedirs(results_root, exist_ok=True)
    tasks = []

    # If the user passed a single file
    if os.path.isfile(target_path):
        if target_path.endswith(".txt") and not os.path.basename(target_path).startswith("res_"):
            scenarios_root, mappa, distanza, scen_file = get_scenario_info(target_path)
            tasks.append((scenarios_root, results_root, mappa, distanza, scen_file))

    # If the user passed a folder (e.g., the entire folder or a specific map)
    elif os.path.isdir(target_path):
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.endswith(".txt") and not file.startswith("res_"):
                    filepath = os.path.join(root, file)
                    scenarios_root, mappa, distanza, scen_file = get_scenario_info(filepath)
                    tasks.append((scenarios_root, results_root, mappa, distanza, scen_file))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No scenarios to execute.", flush=True)
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
    and generates exactly 3 performance plots (all in log-log scale).
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
    # In our case, 'sub' still contains the Map string
    df['obstacles'] = df['sub'].apply(lambda x: int(x.split('-')[1]) if '-' in x else 0)

    # Formats the terminal output to display decimals with a comma
    fmt = lambda x: str(round(x, 6)).replace('.', ',')

    print("\n" + "="*50, flush=True)
    print("MEAN HALF EXECUTION TIMES", flush=True)
    print(f"{'Group':<10} | {'Obstacles':<10} | {'Agents':<10} | {'Exec (s)':<12} | {'Max Piv'}", flush=True)
    print("-" * 50, flush=True)

    grouped_print = df.groupby(['group', 'obstacles', 'agents']).mean(numeric_only=True).reset_index()
    for _, row in grouped_print.iterrows():
        print(f"{row['group']:<10} | {int(row['obstacles']):<10} | {int(row['agents']):<10} | {fmt(row['exec_time_half']):<12} | {fmt(row['max_piv_step'])}", flush=True)

    # Formatter for Matplotlib axes to keep the comma on the plots
    comma_formatter = ticker.FuncFormatter(lambda x, pos: f"{x:g}".replace('.', ','))

    # ==========================================
    # 1. Exec Time vs Number of Agents (log-log)
    # ==========================================
    grouped_agents = df.groupby('agents')['exec_time_half'].mean().reset_index().sort_values('agents')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(grouped_agents['agents'], grouped_agents['exec_time_half'], marker='o', linestyle='-', color='blue')
    ax.set_title('Execution Time vs Number of Agents')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Mean Execution Time (seconds)')
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax.xaxis.set_major_formatter(comma_formatter)
    ax.yaxis.set_major_formatter(comma_formatter)

    plt.savefig('1_exec_time_agents.pdf', format='pdf', bbox_inches='tight')
    print("\nSaved: 1_exec_time_agents.pdf", flush=True)
    plt.close()

    # ==========================================
    # 2. Exec Time vs Map Structural Density (log-log)
    # ==========================================
    grouped_density = df.groupby('obstacles')['exec_time_half'].mean().reset_index().sort_values('obstacles', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(grouped_density['obstacles'], grouped_density['exec_time_half'], marker='s', linestyle='-', color='green')
    ax.set_title('Execution Time vs Map Structural Density')
    ax.set_xlabel('Percentage of Obstacles (%)')
    ax.set_ylabel('Mean Execution Time (seconds)')

    ticks_x = sorted(grouped_density['obstacles'].unique())
    ax.set_xticks(ticks_x)

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.xaxis.set_major_formatter(comma_formatter)
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(comma_formatter)

    plt.savefig('2_exec_time_density.pdf', format='pdf', bbox_inches='tight')
    print("Saved: 2_exec_time_density.pdf", flush=True)
    plt.close()

    # ==========================================
    # 3. Exec Time vs Initial Distance Bounds (log-log)
    # ==========================================
    df['group_numeric'] = pd.to_numeric(df['group'], errors='coerce')
    grouped_distance = df.dropna(subset=['group_numeric']).groupby('group_numeric')['exec_time_half'].mean().reset_index().sort_values('group_numeric', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(grouped_distance['group_numeric'], grouped_distance['exec_time_half'], marker='^', linestyle='-', color='red')
    ax.set_title('Execution Time vs Maximum Initial Distance Bounds')
    ax.set_xlabel('Maximum Initial Distance to Pivot')
    ax.set_ylabel('Mean Execution Time (seconds)')

    ax.set_xticks(grouped_distance['group_numeric'].unique())

    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax.xaxis.set_major_formatter(comma_formatter)
    ax.yaxis.set_major_formatter(comma_formatter)

    plt.savefig('3_exec_time_distance.pdf', format='pdf', bbox_inches='tight')
    print("Saved: 3_exec_time_distance.pdf", flush=True)
    plt.close()

if __name__ == "__main__":
    try:
        # Takes the command line argument if present
        target = sys.argv[1] if len(sys.argv) > 1 else None

        run_experiments(target)

        # Regenerate plots at the end (only if it's not a parallel run of a single file,
        # otherwise dozens of jobs will try to save the same PDF).
        if target is None or os.path.isdir(target):
            generate_plots_and_stats()

    except KeyboardInterrupt:
        print("\nExecution interrupted by the user.", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Critical error during execution: {e}", flush=True)
        sys.exit(1)
