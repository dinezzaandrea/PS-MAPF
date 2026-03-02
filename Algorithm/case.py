import os
import shutil
import glob
import random
import networkx as nx
from collections import defaultdict

def parse_map(filepath):
    """
    Legge un file .map standard (tipo MovingAI) e lo converte in un grafo
    non orientato considerando la connettività a 4 direzioni per le celle libere.
    """
    G = nx.Graph()
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('map'):
            start_idx = i + 1
            break

    grid = [list(line.strip()) for line in lines[start_idx:]]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in ['.', 'G', 'S']:
                G.add_node((c, r)) # <--- (c, r) invece di (r, c)
                if r > 0 and grid[r-1][c] in ['.', 'G', 'S']:
                    G.add_edge((c, r), (c, r-1))
                if c > 0 and grid[r][c-1] in ['.', 'G', 'S']:
                    G.add_edge((c, r), (c-1, r))
    return G

def main():
    # Configurazioni iniziali
    main_folders = [500, 600, 700, 800]
    agent_counts = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    map_dir = 'map'

    # Pattern dei file da cercare
    patterns = ['random512-15-?.map', 'random512-25-?.map', 'random512-35-?.map', 'random512-40-?.map']
    map_files = []
    for pattern in patterns:
        map_files.extend(glob.glob(os.path.join(map_dir, pattern)))

    if not map_files:
        print(f"Nessuna mappa trovata in '{map_dir}' corrispondente ai pattern.")
        return

    for map_path in map_files:
        map_filename = os.path.basename(map_path)
        print(f"Elaborazione mappa: {map_filename}")

        # 1. Parsing della mappa e creazione grafo
        G = parse_map(map_path)
        if G.number_of_nodes() == 0:
            print(f" -> Mappa vuota o formato non riconosciuto.")
            continue

        # 2. Calcola la più grande componente 2-edge-connected
        # Identifichiamo i "ponti" e li rimuoviamo per estrarre le componenti sicure
        bridges = list(nx.bridges(G))
        G_no_bridges = G.copy()
        G_no_bridges.remove_edges_from(bridges)

        components = list(nx.connected_components(G_no_bridges))

        if not components:
            largest_comp = set(G.nodes())
        else:
            largest_comp = max(components, key=len)

        H = G.subgraph(largest_comp)

        # 3. Seleziona un pivot casuale dalla componente
        pivot = random.choice(list(H.nodes()))

        # 4. Calcola le distanze minime dal pivot a tutti gli altri nodi nella componente
        lengths = nx.single_source_shortest_path_length(H, pivot)

        dist_to_nodes = defaultdict(list)
        for node, dist in lengths.items():
            dist_to_nodes[dist].append(node)

        # 5. Generazione cartelle e file
        for distance in main_folders:
            folder_name = str(distance)
            subfolder_name = os.path.join("Case3", folder_name, map_filename)

            # Crea la struttura di directory
            os.makedirs(subfolder_name, exist_ok=True)

            for count in agent_counts:
                selected_agents = []

                # Partiamo dalla distanza target esatta e andiamo a ritroso verso 0
                for d in range(distance, -1, -1):
                    if d in dist_to_nodes:
                        nodes_at_d = list(dist_to_nodes[d])
                        random.shuffle(nodes_at_d)  # Randomizza la selezione a pari distanza
                        needed = count - len(selected_agents)
                        selected_agents.extend(nodes_at_d[:needed])

                    if len(selected_agents) >= count:
                        break

                # Verifica ed eventuale avviso
                if len(selected_agents) < count:
                    print(f" [!] ATTENZIONE: In {map_filename}, cartella {distance}, richiesti {count} agenti ma trovati solo {len(selected_agents)} nodi a distanza <= {distance}.")

                # 6. Scrittura del file .txt
                txt_path = os.path.join(subfolder_name, f"{count}.txt")
                with open(txt_path, 'w') as f:
                    f.write("map\n")
                    f.write(f"{map_filename}\n")
                    f.write("pivot\n")
                    f.write(f"{pivot[0]} {pivot[1]}\n")
                    f.write("agent & start\n")
                    for agent in selected_agents:
                        f.write(f"{agent[0]} {agent[1]}\n")
                    f.write("destination\n")

def reorganize_folders(base_path):
    """
    Trasforma la struttura da base_path/DISTANZA/MAPPA/file
    a base_path/MAPPA/DISTANZA/file.
    """
    if not os.path.exists(base_path):
        print(f"Cartella non trovata: {base_path}", flush=True)
        return

    print(f"Inizio riorganizzazione di {base_path}...", flush=True)
    temp_dir = base_path + "_temp_migration"
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Spostiamo tutto nella cartella temporanea con la nuova struttura
    for distanza in os.listdir(base_path):
        distanza_path = os.path.join(base_path, distanza)

        # Saltiamo i file (come execution_times.csv) e la cartella temporanea
        if not os.path.isdir(distanza_path) or distanza.endswith("_temp_migration"):
            continue

        for mappa in os.listdir(distanza_path):
            mappa_path = os.path.join(distanza_path, mappa)
            if not os.path.isdir(mappa_path):
                continue

            # Nuova destinazione: temp_dir / MAPPA / DISTANZA
            new_dest_dir = os.path.join(temp_dir, mappa, distanza)
            os.makedirs(new_dest_dir, exist_ok=True)

            for file_name in os.listdir(mappa_path):
                old_file_path = os.path.join(mappa_path, file_name)
                new_file_path = os.path.join(new_dest_dir, file_name)
                shutil.move(old_file_path, new_file_path)

    # 2. Eliminiamo le vecchie cartelle ormai vuote in base_path
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if item_path != temp_dir and os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # 3. Spostiamo le nuove cartelle da temp_dir alla cartella originale
    for mappa in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, mappa), base_path)

    # 4. Rimuoviamo la cartella temporanea vuota
    os.rmdir(temp_dir)
    print(f"Riorganizzazione completata con successo per: {base_path}\n", flush=True)

if __name__ == '__main__':
    main()
    reorganize_folders("Case3")
