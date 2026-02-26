import collections
import concurrent.futures

# --- WORKER GLOBAL VARIABLES ---
# These reside only in the individual core's memory.
# This prevents massive RAM duplication of obstacles and destinations for each task.
global_obstacles = None
global_width = None
global_height = None
global_v_free = None

def init_worker(obs, w, h, v_free_dests):
    """
    Initializes the worker by loading heavy static data only once.
    """
    global global_obstacles, global_width, global_height, global_v_free
    global_obstacles = obs
    global_width = w
    global_height = h
    global_v_free = set(v_free_dests)

def get_paths_to_destinations(start):
    """
    Optimized BFS. Reads obstacles, grid bounds, and targets directly
    from the global variables pre-loaded in the worker.
    """
    queue = collections.deque([(start, [start])])
    visited = {start}
    paths_found = {}

    while queue:
        curr, path = queue.popleft()

        if curr in global_v_free:
            paths_found[curr] = path

            # Optimized early exit: stop if we found all available destinations
            if len(paths_found) == len(global_v_free):
                break

        x, y = curr
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < global_width and 0 <= ny < global_height and \
               (nx, ny) not in global_obstacles and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

    return paths_found

def _get_paths_wrapper(args):
    """Ultra-lightweight wrapper for parallel mapping."""
    agent_id, start_pos = args
    return agent_id, get_paths_to_destinations(start_pos)

def extend_to_destination_set(agents, initial_config, destinations, obstacles, width, height):
    """
    Algorithm 1: Parallel Destination Pathfinding
    Calculates the shortest paths to all free destinations for all unresolved agents,
    then assigns them greedily. Returns an ordered list of paths.

    Returns:
        list: Ordered list of tuples (agent_id, destination_coord, path_list)
              sorted strictly by path length (shortest first).
    """
    x_pos = initial_config.copy()

    # Identify agents already at a destination vs agents that need moving
    agents_at_dest = {i for i in agents if x_pos[i] in destinations}
    agents_to_move = [i for i in agents if i not in agents_at_dest]

    # Identify which destinations are still available
    occupied_destinations = {x_pos[i] for i in agents_at_dest}
    v_free = set(destinations) - occupied_destinations

    ### PHASE 1: Parallel Pathfinding (Memory Optimized) ###
    path_args = [(a_idx, x_pos[a_idx]) for a_idx in agents_to_move]
    agent_to_paths = {}

    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_worker,
        initargs=(obstacles, width, height, v_free)
    ) as executor:

        results = executor.map(_get_paths_wrapper, path_args, chunksize=10)
        for a_idx, paths in results:
            agent_to_paths[a_idx] = paths

    ### PHASE 2: Global Greedy Assignment Based on Real Paths ###
    all_options = []
    for a_idx, paths in agent_to_paths.items():
        for dest, path in paths.items():
            all_options.append((a_idx, dest, path))

    # Sort all options strictly by path length
    all_options.sort(key=lambda item: len(item[2]))

    assignments = []
    assigned_agents = set()
    assigned_dests = set()

    # Build the final assignment list based on the shortest available paths
    for a_idx, dest, path in all_options:
        if a_idx not in assigned_agents and dest not in assigned_dests:
            assignments.append((a_idx, dest, path))
            assigned_agents.add(a_idx)
            assigned_dests.add(dest)

            # Stop once all agents have a target or all destinations are filled
            if len(assigned_agents) == len(agents_to_move) or len(assigned_dests) == len(v_free):
                break

    # Return the directly computed, ordered list of assigned paths
    return assignments
