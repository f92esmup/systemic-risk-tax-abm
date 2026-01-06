import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import os
from parametros import Parametros


def load_simulation_data(mode, run_id):
    """
    Loads the simulation history for a specific run.
    """
    file_path = f"output_data/{mode}/run_{run_id:05d}.npz"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Simulation file not found: {file_path}")

    print(f"Loading data from {file_path}...")
    return np.load(file_path)


def setup_static_graph():
    """
    Creates the base NetworkX graph with all nodes (Banks, Firms, Households)
    and assigns them fixed positions using a concentric layout.

    Returns:
        G (nx.DiGraph): Graph with nodes and 'pos' attributes.
        node_lists (dict): Dictionary identifying node IDs by type.
    """
    B = Parametros.B
    F = Parametros.F
    H = Parametros.H

    G = nx.DiGraph()

    # --- ID RANGES ---
    # Banks: 0 to B-1
    # Firms: B to B+F-1
    # Households: B+F to B+F+H-1
    ids_banks = list(range(0, B))
    ids_firms = list(range(B, B + F))
    ids_households = list(range(B + F, B + F + H))

    node_lists = {"banks": ids_banks, "firms": ids_firms, "households": ids_households}

    # --- POSITIONS (Concentric) ---
    def get_coords(n, radius, noise=0.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        if noise > 0:
            x += np.random.normal(0, noise, n)
            y += np.random.normal(0, noise, n)
        return np.column_stack((x, y))

    pos_b = get_coords(B, radius=1.0)
    pos_f = get_coords(F, radius=3.5)

    # Households scattered in an outer ring
    angles_h = np.linspace(0, 2 * np.pi, H, endpoint=False)
    radii_h = np.random.uniform(6.0, 8.0, H)
    x_h = radii_h * np.cos(angles_h)
    y_h = radii_h * np.sin(angles_h)
    pos_h = np.column_stack((x_h, y_h))

    # Add Nodes to Graph
    for i, idx in enumerate(ids_banks):
        G.add_node(idx, type="bank", pos=pos_b[i], color="blue", size=100)

    for i, idx in enumerate(ids_firms):
        G.add_node(idx, type="firm", pos=pos_f[i], color="green", size=30)

    for i, idx in enumerate(ids_households):
        G.add_node(idx, type="household", pos=pos_h[i], color="orange", size=5)

    return G, node_lists


def draw_subgraph(ax, G, source_ids, target_ids, adjacency, title, edge_color):
    """
    Updates edges in G for the specific subgraph and draws it using NetworkX.
    Note: To be efficient, we only draw the relevant subset of nodes/edges on 'ax'.
    """
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    # 1. Identify active edges from adjacency matrix
    # adjacency shape: (len(source_ids), len(target_ids))
    rows, cols = np.where(adjacency > 1e-9)

    active_edges = []
    weights = []

    if len(rows) > 0:
        # Scale weights for visualization
        w_raw = adjacency[rows, cols]
        w_norm = 0.1 + 0.9 * (w_raw - w_raw.min()) / (w_raw.max() - w_raw.min() + 1e-9)

        for r, c, w in zip(rows, cols, w_norm):
            u = source_ids[r]
            v = target_ids[c]
            active_edges.append((u, v))
            weights.append(w)

    # 2. Draw Nodes (Background)
    # We draw ALL nodes of the relevant types to show the structure, even if unconnected

    # Source Nodes
    nodelist_s = source_ids
    pos = nx.get_node_attributes(G, "pos")
    colors_s = [G.nodes[n]["color"] for n in nodelist_s]
    sizes_s = [G.nodes[n]["size"] for n in nodelist_s]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodelist_s,
        node_color=colors_s,
        node_size=sizes_s,
        ax=ax,
        alpha=0.8,
    )

    # Target Nodes (if different)
    if source_ids != target_ids:
        nodelist_t = target_ids
        colors_t = [G.nodes[n]["color"] for n in nodelist_t]
        sizes_t = [G.nodes[n]["size"] for n in nodelist_t]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist_t,
            node_color=colors_t,
            node_size=sizes_t,
            ax=ax,
            alpha=0.8,
        )

    # 3. Draw Edges
    if active_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=active_edges,
            width=1.0,
            edge_color=edge_color,
            alpha=weights,
            ax=ax,
            arrowsize=5,
        )


def crear_gif_redes(mode, n_run, steps_limit=None, output_filename=None):
    """
    Main function to generate the GIF.
    """
    # 1. Load Data
    data = load_simulation_data(mode, n_run)

    m_interbank = data["matriz_interbancaria"]
    m_credit = data["matriz_credito_firmas"]
    m_deposits = data["matriz_depositos"]
    m_consumo = data["matriz_consumo"]

    total_steps = m_interbank.shape[0]
    if steps_limit is None:
        steps_limit = total_steps
    else:
        steps_limit = min(steps_limit, total_steps)

    print(f"Generating GIF for {steps_limit} steps with NetworkX...")

    # 2. Setup Graph & Positions
    G, node_lists = setup_static_graph()

    # 3. Output Config
    if output_filename is None:
        os.makedirs("output_data/gifs", exist_ok=True)
        output_filename = f"output_data/gifs/{mode}_run{n_run}_network_nx.gif"

    temp_dir = f"output_data/temp_frames_nx_{mode}_{n_run}"
    os.makedirs(temp_dir, exist_ok=True)

    filenames = []
    step_jump = max(1, steps_limit // 40)  # Target ~40 frames

    # 4. Loop
    for t in range(0, steps_limit, step_jump):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Network Dynamics (NetworkX) - Step {t}", fontsize=16)

        # A. Interbank (Banks -> Banks)
        draw_subgraph(
            axes[0, 0],
            G,
            node_lists["banks"],
            node_lists["banks"],
            m_interbank[t],
            "Interbank Market",
            "blue",
        )

        # B. Credit (Firms -> Banks)
        # Note: In the matrix, Rows=Firms, Cols=Banks. Flow is technically Credit Line (Bank->Firm) or Debt (Firm->Bank).
        # We visualize the relationship.
        draw_subgraph(
            axes[0, 1],
            G,
            node_lists["firms"],
            node_lists["banks"],
            m_credit[t],
            "Credit Market (Firms-Banks)",
            "green",
        )

        # C. Deposits (Households -> Banks)
        draw_subgraph(
            axes[1, 0],
            G,
            node_lists["households"],
            node_lists["banks"],
            m_deposits[t],
            "Deposit Market (HH-Banks)",
            "orange",
        )

        # D. Consumption (Households -> Firms)
        draw_subgraph(
            axes[1, 1],
            G,
            node_lists["households"],
            node_lists["firms"],
            m_consumo[t],
            "Consumption Market (HH-Firms)",
            "purple",
        )

        # Save
        frame_path = f"{temp_dir}/frame_{t:05d}.png"
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        plt.savefig(frame_path, dpi=70)  # Lower DPI for speed
        plt.close(fig)
        filenames.append(frame_path)

        if t % step_jump == 0:
            print(f"Rendered step {t}/{steps_limit}")

    # 5. Compile
    print("Compiling GIF...")
    images = [Image.open(fn) for fn in filenames]
    if images:
        images[0].save(
            output_filename,
            save_all=True,
            append_images=images[1:],
            duration=250,
            loop=0,
        )

    # Cleanup
    for filename in filenames:
        os.remove(filename)
    os.rmdir(temp_dir)
    print(f"GIF saved to: {output_filename}")
