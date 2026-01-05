import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import functions as fn
from parameters import Params

def plot_network_topology(mode="none", run_id=0, step_t=-1):
    """
    Reconstruye la red interbancaria y la visualiza usando NetworkX.
    Nodos: Bancos.
    Color: DebtRank (Rojo = Alto Riesgo, Verde = Bajo Riesgo).
    Tamaño: Activos Totales.
    Aristas: Pasivos (i -> j significa i debe a j).
    """
    file_path = f"output_data/{mode}/run_{run_id:05d}.npz"
    if not os.path.exists(file_path):
        print(f"Archivo no encontrado: {file_path}")
        return

    # 1. Cargar datos
    data = np.load(file_path)
    L_bb_history = data["L_bb"]
    banks_history = data["banks_state"]
    
    # Seleccionar el paso de tiempo (último por defecto)
    L = L_bb_history[step_t]
    banks = banks_history[step_t]
    
    # Extraer métricas para DebtRank y Visualización
    # banks_state: [Liq, Equity, Dep, Bad, CHI, PSI, DefProb, TotalAssets]
    equity = banks[:, 1]
    total_assets = banks[:, 7]
    
    # Calcular DebtRank en tiempo real para el coloreado
    # v = Importancia relativa (Assets / Total System Assets)
    v = total_assets / (np.sum(total_assets) + 1e-9)
    R = fn.compute_debtrank(L, equity, v)

    # 2. Construir Grafo NetworkX
    G = nx.DiGraph()
    B = L.shape[0]
    G.add_nodes_from(range(B))
    
    # Añadir aristas
    # L[i, j] > 0 significa que Banco i (Fila) debe a Banco j (Columna).
    # La arista va de Deudor a Acreedor (Flujo de pago o Impacto de Default).
    # Si i quiebra, golpea a j. Dibujamos i -> j.
    max_liability = np.max(L) if np.max(L) > 0 else 1.0

    for i in range(B):
        for j in range(B):
            if L[i, j] > 1e-3: # Filtrar préstamos muy pequeños
                G.add_edge(i, j, weight=L[i, j])

    # 3. Configuración Visual
    plt.figure(figsize=(12, 10))
    
    # Layout: Kamada-Kawai suele ser bueno para ver clusters, Circular para ver densidad global
    # Usaremos Kamada-Kawai si es posible, fallback a Circular
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        pos = nx.circular_layout(G)

    # Nodos
    # Tamaño proporcional a log(Assets) o Assets directo escalado
    # Evitar log(0)
    safe_assets = np.maximum(total_assets, 1.0)
    node_sizes = np.log(safe_assets) * 300 
    
    # Color: Mapa de calor invertido (Verde=Bajo R, Rojo=Alto R)
    # DebtRank está entre 0 y 1 (o aprox).
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_size=node_sizes, 
                                   node_color=R, 
                                   cmap=plt.cm.RdYlGn_r, 
                                   alpha=0.9, 
                                   edgecolors="#333333")

    # Aristas
    # Ancho proporcional al peso
    edges_list = list(G.edges(data=True))
    if edges_list:
        weights = [d['weight'] for u, v, d in edges_list]
        # Escalar anchos para que sean visibles pero no enormes
        widths = [2.0 * (w / max_liability) + 0.5 for w in weights]
        
        nx.draw_networkx_edges(G, pos, 
                               width=widths, 
                               edge_color="grey", 
                               alpha=0.6, 
                               arrowstyle="-|>", 
                               arrowsize=15,
                               connectionstyle="arc3,rad=0.1") # Curvar ligeramente para ver bidireccionales

    # Etiquetas (IDs de bancos)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_color="black")

    # Barra de Color
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(vmin=0, vmax=np.max(R) if np.max(R)>0 else 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label="Systemic Risk (DebtRank)", shrink=0.8)
    
    plt.title(f"Interbank Network Topology\nMode: {mode.upper()} | Step: {step_t}", fontsize=16)
    plt.axis("off")
    
    # Guardar
    output_filename = f"network_nx_{mode}_run{run_id}_t{step_t}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Grafo (NetworkX) guardado: {output_filename}")
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Modo específico
        plot_network_topology(mode=sys.argv[1])
    else:
        # Generar ambos por defecto para demostración
        plot_network_topology(mode="none")
        plot_network_topology(mode="srt")
