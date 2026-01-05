import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from parametros import Parametros


def generar_grafo_multicapa(run_id, step, folder="output_data"):
    """
    Reconstructs and plots the multilayer network for a specific step of a simulation run.
    Uses persistent matrices: Interbank, Credit, Labor, Consumption, Ownership.
    """
    filename = f"{folder}/run_{run_id:05d}.npz"
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    print(f"Loading data from {filename}...")
    data = np.load(filename)

    # Check if step is valid
    n_steps = data["matriz_interbancaria"].shape[0]
    if step >= n_steps:
        print(f"Error: Step {step} out of range (Max {n_steps - 1}).")
        return

    # Extract Matrices for the step
    mat_ib = data["matriz_interbancaria"][step]
    mat_credit = data["matriz_credito_firmas"][step]
    mat_labor = data["matriz_laboral"][step]
    mat_consum = data["matriz_consumo"][step]
    mat_owner_f = data["matriz_propiedad_firmas"][step]
    mat_owner_b = data["matriz_propiedad_bancos"][step]

    # Initialize Graph
    G = nx.DiGraph()

    # --- ADD NODES ---
    # Layers (Y-coordinates): Banks=3, Firms=2, Households=1

    # Banks (0..B-1)
    for b in range(Parametros.B):
        G.add_node(f"B{b}", layer=3, color="red", pos=(np.random.rand(), 3))

    # Firms (0..F-1)
    for f in range(Parametros.F):
        G.add_node(f"F{f}", layer=2, color="blue", pos=(np.random.rand(), 2))

    # Households (0..H-1) -> Sample subset for visualization if H is huge
    # Visualizing 1300 nodes is messy. Let's sample 50 active ones or just plot edges blindly.
    # For full graph, we include all but maybe render them small.
    # Let's plot only a subset of Households to keep it sane: First 50.
    H_subset = 50
    for h in range(H_subset):
        G.add_node(f"H{h}", layer=1, color="green", pos=(np.random.rand(), 1))

    # --- ADD EDGES (Layer by Layer) ---

    # 1. Interbank (Bank -> Bank) [RED]
    rows, cols = np.where(mat_ib > 0)
    for r, c in zip(rows, cols):
        weight = mat_ib[r, c]
        G.add_edge(f"B{r}", f"B{c}", weight=weight, type="interbank", color="red")

    # 2. Credit (Firm -> Bank) [BLUE]
    # Matrix rows=Firms, Cols=Banks. Meaning Firm BORROWS from Bank.
    # Edge direction: usually Flow of Money? Bank -> Firm.
    # Or Liability? Firm -> Bank.
    # Let's draw Credit Relationship: Bank -> Firm (Lending)
    rows, cols = np.where(mat_credit > 0)
    for f_idx, b_idx in zip(rows, cols):
        weight = mat_credit[f_idx, b_idx]
        G.add_edge(f"B{b_idx}", f"F{f_idx}", weight=weight, type="credit", color="blue")

    # 3. Labor (Household -> Firm) [GREY]
    # Matrix H x F. 1 if H works for F.
    # Only for subset of H
    rows, cols = np.where(mat_labor[:H_subset] > 0)
    for h_idx, f_idx in zip(rows, cols):
        G.add_edge(f"H{h_idx}", f"F{f_idx}", type="labor", color="grey", style="dashed")

    # 4. Consumption (Household -> Firm) [GREEN]
    # Matrix H x F.
    rows, cols = np.where(mat_consum[:H_subset] > 0)
    for h_idx, f_idx in zip(rows, cols):
        weight = mat_consum[h_idx, f_idx]
        G.add_edge(
            f"H{h_idx}", f"F{f_idx}", weight=weight, type="consumption", color="green"
        )

    # 5. Ownership (Household -> Firm/Bank) [GOLD]
    # Firms
    rows, cols = np.where(mat_owner_f[:H_subset] > 0)
    for h_idx, f_idx in zip(rows, cols):
        G.add_edge(f"H{h_idx}", f"F{f_idx}", type="owner", color="gold")

    # Banks
    rows, cols = np.where(mat_owner_b[:H_subset] > 0)
    for h_idx, b_idx in zip(rows, cols):
        G.add_edge(f"H{h_idx}", f"B{b_idx}", type="owner", color="gold")

    # --- DRAWING ---
    plt.figure(figsize=(12, 10))

    # Position layout
    # Use Multipartite layout
    pos = {}
    # Banks: Top line
    for i in range(Parametros.B):
        pos[f"B{i}"] = (i / Parametros.B, 0.9)

    # Firms: Middle line
    for i in range(Parametros.F):
        pos[f"F{i}"] = (i / Parametros.F, 0.5)

    # Households: Bottom line (Subset)
    for i in range(H_subset):
        pos[f"H{i}"] = (i / H_subset, 0.1)

    # Draw Nodes
    # Banks
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n in G.nodes if n.startswith("B")],
        node_color="red",
        node_size=100,
    )
    # Firms
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n in G.nodes if n.startswith("F")],
        node_color="blue",
        node_size=50,
    )
    # Households
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[n for n in G.nodes if n.startswith("H")],
        node_color="green",
        node_size=20,
    )

    # Draw Edges by Type
    edges = G.edges(data=True)

    # Interbank
    ib_edges = [(u, v) for u, v, d in edges if d.get("type") == "interbank"]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=ib_edges,
        edge_color="red",
        width=1.5,
        alpha=0.7,
        connectionstyle="arc3,rad=0.1",
    )

    # Credit
    cr_edges = [(u, v) for u, v, d in edges if d.get("type") == "credit"]
    nx.draw_networkx_edges(
        G, pos, edgelist=cr_edges, edge_color="blue", width=1.0, alpha=0.5
    )

    # Labor
    lb_edges = [(u, v) for u, v, d in edges if d.get("type") == "labor"]
    nx.draw_networkx_edges(
        G, pos, edgelist=lb_edges, edge_color="grey", style="dashed", alpha=0.3
    )

    # Consumption
    co_edges = [(u, v) for u, v, d in edges if d.get("type") == "consumption"]
    nx.draw_networkx_edges(G, pos, edgelist=co_edges, edge_color="green", alpha=0.3)

    # Owner
    ow_edges = [(u, v) for u, v, d in edges if d.get("type") == "owner"]
    nx.draw_networkx_edges(
        G, pos, edgelist=ow_edges, edge_color="gold", width=2.0, alpha=0.6
    )

    plt.title(f"Multilayer Economic Network (Run {run_id}, Step {step})")
    plt.axis("off")

    output_path = f"grafo_run{run_id}_t{step}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Example Usage:
    # First, run a small simulation to generate data
    from simulacion import Modelo_CRISIS

    print("Running simulation for visualization data...")
    model = Modelo_CRISIS(seed=42)
    # Run enough steps to get interbank activity (usually after a few steps of deficits)
    for _ in range(10):
        model.ejecutar_paso()

    model.guardar_simulacion_disco(run_id=999, folder="output_data_vis")

    # Generate Graph for Step 5
    generar_grafo_multicapa(run_id=999, step=5, folder="output_data_vis")
