import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
from parametros import Param as p

# CONFIGURACIÓN
# Apunta al directorio de una simulación específica generada por el orquestador
DATA_DIR = "outputdata/run_SRT_sim_0" 
OUTPUT_GIF = "evolucion_red_interbancaria.gif"
TEMP_DIR = "temp_frames"

def generar_gif_red():
    print(f"--- Generando GIF de Red Interbancaria desde {DATA_DIR} ---")
    
    # 1. Cargar Datos
    try:
        # Mi logger guarda redes como edgelist: source, target, weight, t
        net_path = os.path.join(DATA_DIR, "net_BB.parquet")
        # Mi logger guarda agentes bancos con: id, liq, eq, dr, t
        banks_path = os.path.join(DATA_DIR, "banks.parquet")
        
        if not os.path.exists(net_path) or not os.path.exists(banks_path):
            raise FileNotFoundError("No se encuentran los archivos .parquet necesarios.")
            
        df_net = pd.read_parquet(net_path)
        df_banks = pd.read_parquet(banks_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Preparar directorios
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 2. Configurar Layout Estático
    # Usamos todos los bancos (B) para que la posición sea fija
    G_base = nx.DiGraph()
    G_base.add_nodes_from(range(p.B))
    pos = nx.circular_layout(G_base) # Layout circular es muy limpio para 20 nodos
    
    frames = []
    
    # Determinar pasos de tiempo únicos
    steps = sorted(df_banks['t'].unique())
    print(f"Procesando {len(steps)} pasos de tiempo...")

    # 3. Bucle temporal
    for t in steps:
        # A. Extraer red en t (Edgelist)
        edges_t = df_net[df_net['t'] == t]
        
        # B. Extraer estado de los bancos en t
        banks_t = df_banks[df_banks['t'] == t].set_index('id')
        
        # C. Crear Grafo
        G = nx.DiGraph()
        G.add_nodes_from(range(p.B))
        
        weights = []
        for _, row in edges_t.iterrows():
            if row['weight'] > 0:
                G.add_edge(int(row['source']), int(row['target']), weight=row['weight'])
                weights.append(row['weight'])

        # D. Definir Colores (Verde = Vivo, Rojo = Quebrado)
        node_colors = []
        for i in range(p.B):
            if i in banks_t.index:
                eq = banks_t.loc[i, 'eq']
                if eq <= 0:
                    node_colors.append('#FF4C4C') # Rojo suave
                else:
                    node_colors.append('#4CFF4C') # Verde suave
            else:
                node_colors.append('#A0A0A0') # Gris (desconocido)

        # E. DIBUJAR
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('#F0F0F0')
        
        # Dibujar nodos
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, 
                               edgecolors='black', linewidths=1.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Dibujar aristas
        if weights:
            # Normalizar grosores (max 4.0)
            max_w = max(weights) if max(weights) > 0 else 1
            scaled_widths = [ (w / max_w) * 4 for w in weights ]
            nx.draw_networkx_edges(G, pos, width=scaled_widths, arrowstyle='->', 
                                   arrowsize=20, edge_color='#606060', alpha=0.5, ax=ax)
        
        ax.set_title(f"Evolución del Riesgo Sistémico (t={t})\nVerde: Solvente | Rojo: Default", 
                     fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Guardar frame
        filename = f"{TEMP_DIR}/frame_{int(t):03d}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        frames.append(filename)
        
        if t % 50 == 0:
            print(f"   Renderizado t={t}")

    # 4. Crear GIF
    if not frames:
        print("❌ No se generaron frames.")
        return
        
    print("Ensamblando GIF...")
    with imageio.get_writer(OUTPUT_GIF, mode='I', duration=0.2) as writer:
        for filename in frames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Limpieza
    for filename in frames:
        os.remove(filename)
    os.rmdir(TEMP_DIR)
    
    print(f"✅ ¡Éxito! GIF animado guardado como: {OUTPUT_GIF}")

if __name__ == "__main__":
    generar_gif_red()
