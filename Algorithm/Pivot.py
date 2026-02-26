import collections
import concurrent.futures

# --- WORKER GLOBAL VARIABLES ---
# These reside only in the independent memory space of each worker core.
# We use this pattern to prevent massive RAM duplication of static data
# (like the obstacles set) across processes during parallel execution.
global_obstacles = None
global_width = None
global_height = None
global_pivot = None

def init_worker(obs, w, h, piv):
    """
    Initializer for the ProcessPoolExecutor.
    Executed exactly once per worker process upon startup to load heavy static data.
    """
    global global_obstacles, global_width, global_height, global_pivot
    global_obstacles = obs
    global_width = w
    global_height = h
    global_pivot = piv

def get_path(start):
    """
    Standard Breadth-First Search (BFS) to find the shortest path between two nodes.
    Uses worker-local global variables for grid dimensions, obstacles, and the goal (pivot).

    Args:
        start (tuple): The (x, y) starting coordinate.

    Returns:
        list: A list of (x, y) coordinates representing the path, or None if no path exists.
    """
    queue = collections.deque([(start, [start])])
    visited = {start}

    while queue:
        (x, y), path = queue.popleft()

        # Check against the global pivot instead of a passed goal argument
        if (x, y) == global_pivot:
            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Use global grid constraints and obstacles
            if 0 <= nx < global_width and 0 <= ny < global_height and \
               (nx, ny) not in global_obstacles and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

    return None

def get_cycle_path(v, u):
    """
    Finds a path in the graph minus the direct edge (u, v) from node v back to node u.
    This forces the algorithm to find an alternative loop to complete a full cycle.
    Uses worker-local global variables for grid constraints.

    Args:
        v (tuple): The node the agent is stepping INTO.
        u (tuple): The node the agent is stepping FROM.

    Returns:
        list: The loop path from v to u, or None if no loop is possible.
    """
    queue = collections.deque([(v, [v])])
    visited = {v}

    while queue:
        curr, path = queue.popleft()

        if curr == u:
            return path

        x, y = curr
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Use global grid constraints and obstacles
            if 0 <= nx < global_width and 0 <= ny < global_height and (nx, ny) not in global_obstacles:

                # CRITICAL STEP: Prevent taking the direct edge back to 'u'
                # at the start of the search to ensure a genuine loop.
                if curr == v and (nx, ny) == u:
                    continue

                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))

    return None

def compute_agent_moves(args):
    """
    Computes the forward cycle sequence for a single agent.

    Args:
        args (tuple): A packed tuple containing (agent_id, start_pos).

    Returns:
        tuple: (agent_id, list_of_cycles) containing the ordered cycle arrays.
    """
    agent_id, start_pos = args

    # Flush=True ensures progress is monitored in real-time in log files
    #print(f"Core working on agent {agent_id} for pivot visit...", flush=True)

    # Calculate the shortest path using global variables exclusively
    path = get_path(start_pos)
    cycles = []

    if not path:
        return agent_id, []

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]

        # Calculate the cycle path using global variables
        p_prime = get_cycle_path(v, u)

        if p_prime:
            # Construct the continuous cyclic sequence where 'u' moves to 'v'.
            # Omit the last element of p_prime (which is 'u') to avoid duplication.
            cycle = [u] + p_prime[:-1]
            cycles.append(cycle)

    return agent_id, cycles

def parallel_pivot_visit(agents_to_task, initial_config, pivot_o, obstacles, width, height):
    """
    Algorithm: Pivot Visit via Cycle Rotation
    Computes all spatial cycles in parallel to maximize performance.

    Args:
        agents_to_task (list): List of agent IDs to process.
        initial_config (dict): Mapping {agent_id: (x, y)}.
        pivot_o (tuple): The target pivot coordinate (x, y).
        obstacles (set): Static obstacles.
        width (int): Grid width.
        height (int): Grid height.

    Returns:
        dict: A mapping {agent_id: list_of_cycles} containing the computed paths.
    """
    task_args = ((a_idx, initial_config[a_idx]) for a_idx in agents_to_task)
    all_agent_cycles = {}

    with concurrent.futures.ProcessPoolExecutor(
        initializer=init_worker,
        initargs=(obstacles, width, height, pivot_o)
    ) as executor:

        results = executor.map(compute_agent_moves, task_args, chunksize=10)

        for a_idx, cycles in results:
            all_agent_cycles[a_idx] = cycles

    return all_agent_cycles
