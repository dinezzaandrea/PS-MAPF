import time
import collections

# Import the modules we previously built
import Destination as dest
import Pivot as piv
import PivotOptimal as pivOpt

def check_pivot_reachability_without_bridges(width, height, obstacles, pivot, initial_config):
    """
    Verifies if all agents are within the same "bridgeless" (2-edge-connected)
    component as the pivot point. It uses an iterative version of Tarjan's
    bridge-finding algorithm.
    """
    if not pivot or pivot in obstacles:
        return False

    # 1. Build the adjacency list for the grid graph
    adj = collections.defaultdict(list)
    nodes = set()
    for r in range(height):
        for c in range(width):
            if (c, r) not in obstacles:
                nodes.add((c, r))
                for dx, dy in [(0, 1), (1, 0)]:
                    nc, nr = c + dx, r + dy
                    if 0 <= nc < width and 0 <= nr < height and (nc, nr) not in obstacles:
                        adj[(c, r)].append((nc, nr))
                        adj[(nc, nr)].append((c, r))

    # 2. Iterative Tarjan's Bridge-Finding Algorithm
    tin = {n: -1 for n in nodes}
    low = {n: -1 for n in nodes}
    timer = 0
    bridges = set()

    stack = [(pivot, None, iter(adj[pivot]))]
    tin[pivot] = low[pivot] = timer
    timer += 1

    while stack:
        u, p, children = stack[-1]
        try:
            v = next(children)
            if v == p:
                continue

            if tin[v] != -1:
                low[u] = min(low[u], tin[v])
            else:
                tin[v] = low[v] = timer
                timer += 1
                stack.append((v, u, iter(adj[v])))
        except StopIteration:
            stack.pop()
            if stack:
                parent, _, _ = stack[-1]
                low[parent] = min(low[parent], low[u])
                if low[u] > tin[parent]:
                    bridges.add(tuple(sorted((parent, u))))

    # 3. BFS to explore the safe component
    visited = {pivot}
    pivot_component = set([pivot])
    q = collections.deque([pivot])

    while q:
        u = q.popleft()
        pivot_component.add(u)
        for v in adj[u]:
            if v not in visited and tuple(sorted((u, v))) not in bridges:
                visited.add(v)
                q.append(v)

    # 4. Check if all agents are inside
    for agent_pos in initial_config.values():
        if agent_pos not in pivot_component:
            return False

    return True

def check_optimality(width, height, obstacles, initial_config):
    """
    Verifies two optimality conditions:
    1. Fully occupied graph: every non-obstacle node hosts a task at the start.
    2. 2-vertex-connected: the graph of free nodes has no articulation points and is connected.
    """
    # 1. Identify all free nodes (non-obstacles)
    nodes = set()
    for r in range(height):
        for c in range(width):
            if (c, r) not in obstacles:
                nodes.add((c, r))

    if not nodes:
        return False

    # Verify Condition 1: Is the graph fully occupied by tasks?
    starting_positions = set(initial_config.values())
    if nodes != starting_positions:
        return False

    # 2. Build the adjacency list (same logic used previously)
    adj = collections.defaultdict(list)
    for c, r in nodes:
        for dx, dy in [(0, 1), (1, 0)]:
            nc, nr = c + dx, r + dy
            if 0 <= nc < width and 0 <= nr < height and (nc, nr) not in obstacles:
                adj[(c, r)].append((nc, nr))
                adj[(nc, nr)].append((c, r))

    # 3. Iterative Tarjan's Algorithm for Articulation Points
    start_node = next(iter(nodes))
    tin = {n: -1 for n in nodes}
    low = {n: -1 for n in nodes}
    timer = 0
    articulation_points = set()

    # stack: (current_node, parent_node, children_iterator)
    stack = [(start_node, None, iter(adj[start_node]))]
    tin[start_node] = low[start_node] = timer
    timer += 1

    children_of_root = 0

    while stack:
        u, p, children = stack[-1]
        try:
            v = next(children)
            if v == p:
                continue

            if tin[v] != -1:
                # Back-edge: node v has already been visited
                low[u] = min(low[u], tin[v])
            else:
                # Tree-edge: we discover a new node
                if p is None:
                    children_of_root += 1

                tin[v] = low[v] = timer
                timer += 1
                stack.append((v, u, iter(adj[v])))

        except StopIteration:
            # All neighbors of 'u' have been explored
            stack.pop()
            if stack:
                parent, _, _ = stack[-1]
                # Update the low value of the parent with that of the completed child (u)
                low[parent] = min(low[parent], low[u])

                # Condition for articulation points (if the parent is NOT the root)
                if parent != start_node and low[u] >= tin[parent]:
                    articulation_points.add(parent)

    # Specific condition for the root node
    if children_of_root > 1:
        articulation_points.add(start_node)

    # Final verification:
    # - All nodes must have been visited (the graph is connected)
    if any(t == -1 for t in tin.values()):
        return False

    # - There must be no articulation points (the graph is 2-vertex-connected)
    if len(articulation_points) > 0:
        return False

    return True

def calculate_pivot_times(agent_ids, initial_config, pivot, all_agent_cycles):
    """
    Calculates the exact logical time step when each individual agent reaches
    the pivot for the very first time. Uses a lightweight spatial tracker to
    detect if an agent is pushed onto the pivot by another agent's cycle.

    Returns:
        tuple: (dict of {agent_id: time_reached}, total_pivot_time)
    """
    current_time = 0
    agent_reach_times = {}

    # Lightweight spatial tracking
    x_pos = initial_config.copy()
    cell_to_agent = {pos: i for i, pos in x_pos.items()}

    # Record any agents already sitting on the pivot at t=0
    for a_idx, pos in x_pos.items():
        if pos == pivot:
            agent_reach_times[a_idx] = 0

    for current_executing_agent in agent_ids:
        cycles = all_agent_cycles.get(current_executing_agent, [])

        # --- Forward Phase ---
        for cycle in cycles:
            current_time += 1
            moving_agents = [cell_to_agent.get(v) for v in cycle]
            cycle_length = len(cycle)

            # Clear old footprint
            for agent in moving_agents:
                if agent is not None:
                    del cell_to_agent[x_pos[agent]]

            # Apply new footprint and check pivot intersection
            for i in range(cycle_length):
                agent = moving_agents[i]
                if agent is not None:
                    next_v = cycle[(i + 1) % cycle_length]
                    x_pos[agent] = next_v
                    cell_to_agent[next_v] = agent

                    # Detect first-time pivot reach
                    if next_v == pivot and agent not in agent_reach_times:
                        agent_reach_times[agent] = current_time

        # --- Backward Phase ---
        for cycle in reversed(cycles):
            current_time += 1
            rev_cycle = cycle[::-1]
            moving_agents = [cell_to_agent.get(v) for v in rev_cycle]
            cycle_length = len(rev_cycle)

            # Clear old footprint
            for agent in moving_agents:
                if agent is not None:
                    del cell_to_agent[x_pos[agent]]

            # Apply new footprint and check pivot intersection
            for i in range(cycle_length):
                agent = moving_agents[i]
                if agent is not None:
                    next_v = rev_cycle[(i + 1) % cycle_length]
                    x_pos[agent] = next_v
                    cell_to_agent[next_v] = agent

                    # Detect first-time pivot reach
                    if next_v == pivot and agent not in agent_reach_times:
                        agent_reach_times[agent] = current_time

    return agent_reach_times, current_time

def calculate_pivot_optimal_times(initial_config, pivot, cycles_list):
    """
    Simulates the cycles generated by the optimal algorithm to calculate the
    exact moment (logical step) when each agent reaches the pivot for the
    first time.

    Returns:
        tuple: (dict of {agent_id: time_reached}, total_pivot_time, final_configuration)
    """
    current_time = 0
    agent_reach_times = {}

    # Lightweight spatial tracking
    x_pos = initial_config.copy()
    cell_to_agent = {pos: i for i, pos in x_pos.items()}

    # Record any agents already sitting on the pivot at time t=0
    for a_idx, pos in x_pos.items():
        if pos == pivot:
            agent_reach_times[a_idx] = 0

    # Apply the cycles sequentially
    for cycle in cycles_list:
        current_time += 1
        moving_agents = [cell_to_agent.get(v) for v in cycle]
        cycle_length = len(cycle)

        # Remove old positions
        for agent in moving_agents:
            if agent is not None:
                del cell_to_agent[x_pos[agent]]

        # Apply new positions and check for pivot intersection
        for i in range(cycle_length):
            agent = moving_agents[i]
            if agent is not None:
                next_v = cycle[(i + 1) % cycle_length]
                x_pos[agent] = next_v
                cell_to_agent[next_v] = agent

                # If the agent touches the pivot for the first time, record the time
                if next_v == pivot and agent not in agent_reach_times:
                    agent_reach_times[agent] = current_time

    # Return x_pos as well, since it acts as the final configuration needed for Phase 2
    return agent_reach_times, current_time, x_pos

def calculate_destination_times(agent_ids, initial_config, destinations, total_pivot_time, assignments):
    """
    Calculates the logical time step when each individual agent makes its final
    movement and settles permanently.

    Returns:
        tuple: (dict of {agent_id: settle_time}, final_total_time)
    """
    current_time = total_pivot_time
    agent_settle_times = {}

    x_pos = initial_config.copy()
    cell_to_agent = {pos: i for i, pos in x_pos.items()}
    agents_at_dest = {i for i in agent_ids if x_pos[i] in destinations}

    # Initialize default times for agents already resolved
    for a_idx in agent_ids:
        agent_settle_times[a_idx] = current_time

    for u_idx, target_dest, path in assignments:
        path_length = len(path) - 1

        chain = []
        curr_agent = u_idx
        curr_t = 0

        while True:
            chain.append((curr_agent, curr_t))
            next_v = path[curr_t + 1]
            w_idx = cell_to_agent.get(next_v)

            if w_idx is not None and w_idx in agents_at_dest:
                curr_agent = w_idx
                curr_t += 1
            else:
                break

            if curr_t + 1 >= len(path):
                break

        final_leader = chain[-1][0]
        agent_settle_times[final_leader] = current_time + path_length
        agents_at_dest.add(final_leader)

        for i in range(len(chain) - 1):
            agent = chain[i][0]
            swap_time_relative = chain[i+1][1]
            agent_settle_times[agent] = current_time + swap_time_relative

        for agent, idx in reversed(chain):
            next_v = path[idx + 1]
            old_pos = x_pos[agent]

            if old_pos in cell_to_agent and cell_to_agent[old_pos] == agent:
                del cell_to_agent[old_pos]

            x_pos[agent] = next_v
            cell_to_agent[next_v] = agent

        current_time += path_length

    return agent_settle_times, current_time

def run_half_algorithm(agent_ids, init_config, pivot, obstacles, w, h):
    """
    Executes only the Pivot Phase.

    Returns:
        tuple: (is_safe (bool), dict_of_pivot_times, formatted_exec_time (str))
    """
    is_safe = check_pivot_reachability_without_bridges(w, h, obstacles, pivot, init_config)

    if not is_safe:
        return False, {}, "0,0"

    start_piv = time.perf_counter()
    all_agent_cycles = piv.parallel_pivot_visit(agent_ids, init_config, pivot, obstacles, w, h)
    exec_time_piv = time.perf_counter() - start_piv

    # Updated signature to pass initial_config and pivot
    dict_of_pivot_times, _ = calculate_pivot_times(agent_ids, init_config, pivot, all_agent_cycles)

    # Decimal comma format applied
    formatted_time_piv = str(exec_time_piv).replace('.', ',')

    return True, dict_of_pivot_times, formatted_time_piv


def run_half_optimal_algorithm(agent_ids, init_config, pivot_o, obstacles, w, h):
    """
    Executes the optimal construction (Pivot Phase only for fully occupied instances).

    Returns:
        tuple: (is_safe (bool), dict_of_pivot_times, formatted_exec_time (str))
    """
    is_safe = check_optimality(w, h, obstacles, init_config)

    if not is_safe:
        return False, {}, "0,0"

    start_opt = time.perf_counter()
    # Generate the list of optimal cycles
    cycles_list = pivOpt.optimal_construction(agent_ids, init_config, pivot_o, obstacles, w, h)
    exec_time_opt = time.perf_counter() - start_opt

    # Calculate times offline, ignoring the final configuration which is not needed here
    dict_of_pivot_times, _, _ = calculate_pivot_optimal_times(init_config, pivot_o, cycles_list)

    formatted_time_opt = str(exec_time_opt).replace('.', ',')

    return True, dict_of_pivot_times, formatted_time_opt

def run_destination_only_algorithm(agent_ids, init_config, dest_set, pivot, obstacles, w, h):
    """
    Executes only the Destination Filling Phase.

    Returns:
        tuple: (is_safe (bool), dict_of_dest_times, formatted_time_dest (str))
    """
    # Optional safety check, but consistent with other functions
    is_safe = check_pivot_reachability_without_bridges(w, h, obstacles, pivot, init_config)

    if not is_safe:
        return False, {}, "0,0"

    # --- DESTINATION PHASE ONLY ---
    start_dest = time.perf_counter()
    assignments = dest.extend_to_destination_set(agent_ids, init_config, dest_set, obstacles, w, h)
    exec_time_dest = time.perf_counter() - start_dest

    # Since we skip the Pivot phase, the previously accumulated time is 0
    total_pivot_time = 0

    dict_of_dest_times, final_time = calculate_destination_times(
        agent_ids, init_config, dest_set, total_pivot_time, assignments
    )

    # Replace dot with comma for decimal values
    formatted_time_dest = str(exec_time_dest).replace('.', ',')

    return True, dict_of_dest_times, formatted_time_dest

def run_full_algorithm(agent_ids, init_config, dest_set, pivot, obstacles, w, h):
    """
    Executes both the Pivot Phase and the Destination Filling Phase.

    Returns:
        tuple: (is_safe (bool), dict_of_pivot_times, dict_of_dest_times,
                formatted_time_piv (str), formatted_time_dest (str))
    """
    is_safe = check_pivot_reachability_without_bridges(w, h, obstacles, pivot, init_config)

    if not is_safe:
        return False, {}, {}, "0,0", "0,0"

    # --- PHASE 1: PIVOT ---
    start_piv = time.perf_counter()
    all_agent_cycles = piv.parallel_pivot_visit(agent_ids, init_config, pivot, obstacles, w, h)
    exec_time_piv = time.perf_counter() - start_piv

    # Updated signature to pass initial_config and pivot
    dict_of_pivot_times, total_pivot_time = calculate_pivot_times(agent_ids, init_config, pivot, all_agent_cycles)

    # --- PHASE 2: DESTINATION ---
    start_dest = time.perf_counter()
    assignments = dest.extend_to_destination_set(agent_ids, init_config, dest_set, obstacles, w, h)
    exec_time_dest = time.perf_counter() - start_dest

    dict_of_dest_times, final_time = calculate_destination_times(
        agent_ids, init_config, dest_set, total_pivot_time, assignments
    )

    # Decimal comma format applied
    formatted_time_piv = str(exec_time_piv).replace('.', ',')
    formatted_time_dest = str(exec_time_dest).replace('.', ',')

    return True, dict_of_pivot_times, dict_of_dest_times, formatted_time_piv, formatted_time_dest

def run_full_optimal_algorithm(agent_ids, init_config, dest_set, pivot_o, obstacles, w, h):
    """
    Executes the full algorithm for fully occupied instances:
    1. Optimal Construction (Pivot Phase).
    2. Destination Filling (Destination Filling Phase).

    Returns:
        tuple: (is_safe (bool), dict_of_pivot_times, dict_of_dest_times,
                formatted_time_piv (str), formatted_time_dest (str))
    """
    is_safe = check_optimality(w, h, obstacles, init_config)

    if not is_safe:
        return False, {}, {}, "0,0", "0,0"

    # --- PHASE 1: PIVOT (OPTIMAL) ---
    start_piv = time.perf_counter()
    cycles_list = pivOpt.optimal_construction(agent_ids, init_config, pivot_o, obstacles, w, h)
    exec_time_piv = time.perf_counter() - start_piv

    # Calculate arrival times and get the final configuration for Phase 2
    dict_of_pivot_times, total_pivot_time, final_config = calculate_pivot_optimal_times(init_config, pivot_o, cycles_list)

    # --- PHASE 2: DESTINATION ---
    start_dest = time.perf_counter()
    # Use final_config instead of history[-1]
    assignments = dest.extend_to_destination_set(agent_ids, final_config, dest_set, obstacles, w, h)
    exec_time_dest = time.perf_counter() - start_dest

    dict_of_dest_times, final_time = calculate_destination_times(
        agent_ids, init_config, dest_set, total_pivot_time, assignments
    )

    formatted_time_piv = str(exec_time_piv).replace('.', ',')
    formatted_time_dest = str(exec_time_dest).replace('.', ',')

    return True, dict_of_pivot_times, dict_of_dest_times, formatted_time_piv, formatted_time_dest
