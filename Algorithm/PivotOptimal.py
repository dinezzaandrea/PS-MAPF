import collections
import time

# --- GRAPH UTILITY FUNCTIONS ---

def get_neighbors(node, width, height, obstacles_set):
    """Computes valid neighbors for a given node dynamically."""
    x, y = node
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles_set:
            neighbors.append((nx, ny))
    return neighbors

def get_articulation_points(nodes_set, width, height, obstacles_set, start_node):
    """
    Finds the articulation points using Tarjan's algorithm.
    Iterative version (non-recursive) to avoid RecursionError on huge maps.
    Returns a set of nodes whose removal would disconnect the graph.
    """
    discovery = {}
    low = {}
    parent = {start_node: None}
    ap_set = set()
    time_counter = 0

    stack = [start_node]
    neighbors_dict = {}
    ptr = {start_node: 0}
    neighbors_dict[start_node] = [n for n in get_neighbors(start_node, width, height, obstacles_set) if n in nodes_set]

    discovery[start_node] = low[start_node] = 0
    time_counter += 1
    children_of_root = 0

    while stack:
        u = stack[-1]

        # If there are still neighbors to explore for node 'u'
        if ptr[u] < len(neighbors_dict[u]):
            v = neighbors_dict[u][ptr[u]]
            ptr[u] += 1

            if v not in discovery:
                parent[v] = u
                if u == start_node:
                    children_of_root += 1

                discovery[v] = low[v] = time_counter
                time_counter += 1

                # Initialize the state for the newly explored node
                neighbors_dict[v] = [n for n in get_neighbors(v, width, height, obstacles_set) if n in nodes_set]
                ptr[v] = 0
                stack.append(v)
            elif v != parent.get(u):
                # Back-edge found
                low[u] = min(low[u], discovery[v])
        else:
            # All neighbors of 'u' have been explored, backtrack
            stack.pop()
            if parent.get(u) is not None:
                p = parent[u]
                low[p] = min(low[p], low[u])

                # Tarjan's condition for articulation points
                if p != start_node and low[u] >= discovery[p]:
                    ap_set.add(p)

    # The root node is an articulation point only if it has more than one independent child in the DFS tree
    if children_of_root > 1:
        ap_set.add(start_node)

    return ap_set

def find_path(start, goal, allowed_nodes, width, height, obstacles_set):
    """
    Finds the shortest path between start and goal using an optimized BFS.
    """
    if start == goal:
        return [start]

    queue = collections.deque([start])
    parents = {start: None}

    while queue:
        curr = queue.popleft()

        for neighbor in get_neighbors(curr, width, height, obstacles_set):
            if neighbor in allowed_nodes and neighbor not in parents:
                parents[neighbor] = curr
                queue.append(neighbor)

                if neighbor == goal:
                    path = []
                    node = goal
                    while node is not None:
                        path.append(node)
                        node = parents[node]
                    path.reverse()
                    return path

    return None

# --- MAIN ALGORITHM ---

def optimal_construction(agents_to_task, initial_config, pivot_o, obstacles, width, height):
    """
    Optimized version: Strictly sequential memory-efficient graph traversal.
    Returns a list of cycles generated at each step.
    """
    obstacles_set = set(obstacles)
    cycles_list = []

    V = set()
    for x in range(width):
        for y in range(height):
            if (x, y) not in obstacles_set:
                V.add((x, y))

    if pivot_o not in V:
        raise ValueError("pivot_o is on an obstacle or out of bounds.")

    S = V - {pivot_o}
    R = {pivot_o}

    step_counter = 0
    start_time = time.time() # Added this to ensure the variable in the print statement is initialized

    while S:
        # Monitoring print every 100 processed agents
        if step_counter > 0 and step_counter % 100 == 0:
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed:.2f}".replace('.', ',')
            print(f"Processed {step_counter} agents... Elapsed time: {elapsed_str} seconds", flush=True)

        # Find the set of candidate nodes (adjacent to R)
        partial_S = [u for u in S if any(n in R for n in get_neighbors(u, width, height, obstacles_set))]

        # Calculate the articulation points for the entire subgraph (S union pivot_o) ONLY ONCE
        nodes_S_union_o = S.union({pivot_o})
        aps = get_articulation_points(nodes_S_union_o, width, height, obstacles_set, pivot_o)

        best_v_t = None
        # Any node in partial_S that is NOT an articulation point is safe to remove
        for u in partial_S:
            if u not in aps:
                best_v_t = u
                break

        if best_v_t is None:
            raise ValueError(f"Could not find a valid non-articulation point at step {step_counter}.")

        v_t = best_v_t
        u_t = next(n for n in get_neighbors(v_t, width, height, obstacles_set) if n in R)

        # PHASE 2: Immediate Sequential Pathfinding
        beta_path = find_path(u_t, pivot_o, R, width, height, obstacles_set)
        gamma_path = find_path(pivot_o, v_t, nodes_S_union_o, width, height, obstacles_set)

        if not beta_path or not gamma_path:
            raise ValueError(f"Pathfinding failed at step {step_counter}")

        cycle = gamma_path + beta_path[:-1]

        # PHASE 3: Save only the cycle
        cycles_list.append(cycle)

        S.remove(v_t)
        R.add(v_t)
        step_counter += 1

    return cycles_list
