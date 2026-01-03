import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def dibujar_grafo_paso(csv_path, paso_objetivo):
    print(f"Cargando datos de red desde {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {csv_path}")
        return
    
    # Filtrar solo el paso deseado
    df_step = df[df['Step'] == paso_objetivo]
    
    if df_step.empty:
        print(f"Error: No hay datos para el paso {paso_objetivo}.")
        return

    G = nx.DiGraph()
    
    # Añadir nodos y aristas
    for _, row in df_step.iterrows():
        tipo_rel = row['Relation_Type']
        
        # Crear IDs únicos combinando Tipo y ID original
        origen = f"{row['Source_Type']}_{int(row['Source_ID'])}"
        destino = f"{row['Target_Type']}_{int(row['Target_ID'])}"
        
        # Filtrar qué mostrar: El paper se centra en la red INTERBANCARIA (Fig 2)
        if tipo_rel == 'INTERBANCARIO':
            G.add_edge(origen, destino, weight=row['Weight'], type=tipo_rel)
            # Asegurar que los nodos existen con atributo tipo
            G.add_node(origen, tipo=row['Source_Type'])
            G.add_node(destino, tipo=row['Target_Type'])

    if G.number_of_nodes() == 0:
        print("No hay conexiones interbancarias en este paso (o el filtro es muy estricto).")
        return

    # --- DISEÑO VISUAL (Replicando estética Paper) ---
    print(f"Generando grafo con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas...")
    pos = nx.spring_layout(G, k=0.5, seed=42)  # Layout de fuerza
    
    # Calcular tamaños basados en grado (conectividad)
    d = dict(G.degree)
    node_sizes = [v * 50 + 100 for v in d.values()]
    
    # Calcular colores (DebtRank proxy: grado de salida ponderado)
    # En el paper: Rojo = Sistémico (alto impacto), Verde = Seguro.
    out_strength = [G.out_degree(n, weight='weight') for n in G.nodes()]
    
    plt.figure(figsize=(12, 10))
    
    # Dibujar nodos
    nodes = nx.draw_networkx_nodes(G, pos, 
                                   node_size=node_sizes, 
                                   node_color=out_strength, 
                                   cmap=plt.cm.RdYlGn_r, # Rojo a Verde invertido
                                   alpha=0.9)
    
    # Dibujar aristas (grosor según monto)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Normalizar grosores para que no tapen todo
    max_w = max(weights) if weights else 1
    widths = [(w / max_w) * 3 + 0.5 for w in weights]
    
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4, edge_color='gray', arrowsize=15)
    
    # Etiquetas (solo IDs numéricos para limpieza)
    labels = {n: n.split('_')[1] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

    plt.colorbar(nodes, label="Importancia Sistémica (Proxy: Exposiciones)")
    plt.title(f"Red Interbancaria - Paso {paso_objetivo} (Replicando Fig. 2)", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    
    output_file = f"red_interbancaria_paso_{paso_objetivo}.png"
    plt.savefig(output_file)
    print(f"Grafo guardado en {output_file}")
    # plt.show() # Deshabilitado para ejecución headless

if __name__ == "__main__":
    # Ajusta el nombre del archivo según tu ejecución
    # Buscamos el archivo más reciente o uno específico
    archivo = "relaciones_IRS_Impuesto_Riesgo_Sistémico.csv" 
    paso = 30 # Un paso avanzado donde la red ya evolucionó
    dibujar_grafo_paso(archivo, paso)
