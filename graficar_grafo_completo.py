import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from parametros import Parametros
import os


def visualizar_grafo_completo(run_id=0, step=50, output_file=None):
    """
    Reconstructs and visualizes the full multilayer network (Banks, Firms, Households)
    at a specific simulation step `t`.

    Layers:
    1. Interbank (Bank -> Bank) [Red]
    2. Credit (Firm -> Bank) [Blue]
    3. Ownership (Household -> Firm/Bank) [Gold, Dashed]
    4. Labor (Household -> Firm) [Grey, Dotted - Sampled]
    """

    # --- 1. Load Data ---
    # Try different folders if 'none' doesn't exist
    filepath = f"output_data/none/run_{run_id:05d}.npz"
    if not os.path.exists(filepath):
        filepath = f"output_data/srt/run_{run_id:05d}.npz"
        if not os.path.exists(filepath):
            print(f"Error: Could not find data for run {run_id}")
            return

    print(f"Loading data from {filepath}...")
    data = np.load(filepath)

    # Check T limit
    T_max = data["estado_bancos"].shape[0]
    if step >= T_max:
        print(f"Step {step} exceeds simulation length {T_max}. Using last step.")
        step = T_max - 1

    # Extract Snapshots
    state_banks = data["estado_bancos"][step]  # (B, N_feat)
    state_firms = data["estado_firmas"][step]  # (F, N_feat)
    state_hh = data["estado_hogares"][step]  # (H, N_feat)

    adj_interbank = data["matriz_interbancaria"][step]  # (B, B)
    adj_credit = data["matriz_credito_firmas"][step]  # (F, B)

    # --- 2. Setup Graph & Offsets ---
    G = nx.DiGraph()

    B = Parametros.B
    F = Parametros.F
    H = Parametros.H

    # Node ID Offsets
    OFFSET_BANK = 0
    OFFSET_FIRM = B
    OFFSET_HH = B + F

    # --- 3. Add Nodes ---
    print("Adding nodes...")

    # Banks
    for i in range(B):
        assets = state_banks[i, Parametros.IDX_BANK_TOTAL_ASSETS]
        G.add_node(
            OFFSET_BANK + i, type="bank", size=assets, color="red", label=f"B{i}"
        )

    # Firms
    for i in range(F):
        # Use Liquidity or Assets as size proxy
        size = state_firms[i, Parametros.IDX_FIRM_LIQUIDITY]
        G.add_node(OFFSET_FIRM + i, type="firm", size=size, color="blue", label=f"F{i}")

    # Households (Filter: Owners + Sample Workers)
    # Ideally, plot all owners.
    owners_mask = state_hh[:, Parametros.IDX_HH_IS_OWNER] == 1.0
    owner_indices = np.where(owners_mask)[0]

    for idx in owner_indices:
        G.add_node(OFFSET_HH + idx, type="hh_owner", size=10, color="gold", label="")

    # --- 4. Add Edges ---
    print("Adding edges...")

    # A. Interbank (Red)
    # Rows=Borrower, Cols=Lender. Edge: Borrower -> Lender (Liability direction)
    rows, cols = np.where(adj_interbank > 0)
    for r, c in zip(rows, cols):
        weight = adj_interbank[r, c]
        G.add_edge(
            OFFSET_BANK + r, OFFSET_BANK + c, weight=weight, color="red", style="solid"
        )

    # B. Credit (Blue)
    # Rows=Firm (Borrower), Cols=Bank (Lender). Edge: Firm -> Bank
    rows, cols = np.where(adj_credit > 0)
    for r, c in zip(rows, cols):
        weight = adj_credit[r, c]
        G.add_edge(
            OFFSET_FIRM + r, OFFSET_BANK + c, weight=weight, color="blue", style="solid"
        )

    # C. Ownership (Gold, Dashed)
    # HH -> Entity
    # Vectorized check
    # Iterate owner indices
    for idx in owner_indices:
        owned_type = int(state_hh[idx, Parametros.IDX_HH_OWNED_TYPE])
        owned_entity_idx = int(state_hh[idx, Parametros.IDX_HH_OWNED_ENTITY_IDX])

        target_node = -1
        if owned_type == 1:  # Firm
            target_node = OFFSET_FIRM + owned_entity_idx
        elif owned_type == 2:  # Bank
            target_node = OFFSET_BANK + owned_entity_idx

        if target_node != -1:
            G.add_edge(
                OFFSET_HH + idx, target_node, color="gold", style="dashed", weight=1
            )

    # D. Labor (Grey, Dotted)
    # Only for Owners to avoid clutter? Or sample workers?
    # Let's verify graph reconstruction capability by plotting labor for the Owners (who are also workers/consumers)
    for idx in owner_indices:
        emp_idx = int(state_hh[idx, Parametros.IDX_HH_EMPLOYER_IDX])
        target_node = OFFSET_FIRM + emp_idx
        G.add_edge(
            OFFSET_HH + idx, target_node, color="grey", style="dotted", weight=0.5
        )

    # --- 5. Visualization ---
    print("Calculating layout...")
    # Multipartite Layout: Banks Top, Firms Middle, HH Bottom
    pos = {}

    # Random x for spread, fixed y for layers
    # Banks (Y=3)
    for i in range(B):
        pos[OFFSET_BANK + i] = np.array([np.random.uniform(-1, 1), 1.0])

    # Firms (Y=0)
    for i in range(F):
        pos[OFFSET_FIRM + i] = np.array([np.random.uniform(-1.5, 1.5), 0.0])

    # HH (Y=-1)
    for idx in owner_indices:
        pos[OFFSET_HH + idx] = np.array([np.random.uniform(-1.5, 1.5), -1.0])

    # Improve x-positions using spring layout constrained to y?
    # Or just use spring layout for the whole thing, initialized with these pos?
    pos = nx.spring_layout(G, pos=pos, fixed=None, k=0.3, iterations=50)

    # Drawing
    plt.figure(figsize=(12, 12))

    # Nodes
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    # Normalize sizes
    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]
    # Scale for plot
    # Avoid 0 size
    max_s = max(node_sizes) if node_sizes else 1
    node_sizes = [
        100 + (s / (max_s + 1e-9)) * 500 if G.nodes[n]["type"] != "hh_owner" else 50
        for s, n in zip(node_sizes, G.nodes())
    ]

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8
    )

    # Labels (Only Banks/Firms)
    labels = {
        n: G.nodes[n]["label"] for n in G.nodes() if G.nodes[n]["type"] != "hh_owner"
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white")

    # Edges
    edges = G.edges()
    edge_colors = [G[u][v]["color"] for u, v in edges]
    edge_styles = [G[u][v]["style"] for u, v in edges]

    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, style=edge_styles, alpha=0.6, arrows=True
    )

    plt.title(f"Full Multilayer Network Reconstruction (t={step})")
    plt.axis("off")

    if output_file is None:
        output_file = f"grafo_completo_t{step}.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Graph saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Ensure data exists first (Run main if needed, but we assume prev steps ran)
    # We will look for data generated by verify_phase5 if it saved, OR
    # we need to run a small sim first.

    # Run a quick sim to generate data
    from simulacion import Modelo_CRISIS

    m = Modelo_CRISIS(seed=42)
    print("Running simulation for 5 steps...")
    for t in range(5):
        m.ejecutar_paso()
    m.guardar_simulacion_disco(run_id=999, folder="output_data/none")

    visualizar_grafo_completo(run_id=999, step=4)
