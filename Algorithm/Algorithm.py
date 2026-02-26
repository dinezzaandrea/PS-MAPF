import time
import collections

# Import the modules we previously built
import Destination as dest
import Pivot as piv

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
