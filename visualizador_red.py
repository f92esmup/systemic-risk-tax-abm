import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import argparse
from parametros import Param as p

# CONFIGURACIÓN POR DEFECTO
DEFAULT_DATA_DIR = "outputdata/run_SRT_sim_0"
DEFAULT_DURATION = 0.5  # Segundos por frame (más lento)


def cargar_datos(data_dir):
    try:
        data = {}
        data["net_BB"] = pd.read_parquet(os.path.join(data_dir, "net_BB.parquet"))
        data["banks"] = pd.read_parquet(os.path.join(data_dir, "banks.parquet"))

        # Intentar cargar datos extra para modo macro, pero no fallar si no existen (para compatibilidad)
        path_fb = os.path.join(data_dir, "net_FB.parquet")
        path_firms = os.path.join(data_dir, "firms.parquet")

        if os.path.exists(path_fb) and os.path.exists(path_firms):
            data["net_FB"] = pd.read_parquet(path_fb)
            data["firms"] = pd.read_parquet(path_firms)
            data["has_macro"] = True
        else:
            data["has_macro"] = False

        return data
    except Exception as e:
        print(f"❌ Error cargando datos desde {data_dir}: {e}")
        return None


def generar_gif_interbancario(data, output_file, duration):
    print(f"--- Generando GIF Interbancario -> {output_file} ---")
    temp_dir = "temp_frames_ib"
    os.makedirs(temp_dir, exist_ok=True)

    df_net = data["net_BB"]
    df_banks = data["banks"]

    # Layout Estático
    G_base = nx.DiGraph()
    G_base.add_nodes_from(range(p.B))
    pos = nx.circular_layout(G_base)

    steps = sorted(df_banks["t"].unique())
    frames = []

    for t in steps:
        edges_t = df_net[df_net["t"] == t]
        banks_t = df_banks[df_banks["t"] == t].set_index("id")

        G = nx.DiGraph()
        G.add_nodes_from(range(p.B))

        weights = []
        for _, row in edges_t.iterrows():
            if row["weight"] > 0:
                G.add_edge(int(row["source"]), int(row["target"]), weight=row["weight"])
                weights.append(row["weight"])

        node_colors = []
        for i in range(p.B):
            if i in banks_t.index:
                if banks_t.loc[i, "eq"] <= 0:
                    node_colors.append("#FF4C4C")  # Rojo
                else:
                    node_colors.append("#4CFF4C")  # Verde
            else:
                node_colors.append("gray")

        _, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("#F0F0F0")

        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_colors,
            node_size=500,
            edgecolors="black",
            linewidths=1.5,
            ax=ax,
        )
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

        if weights:
            max_w = max(weights) if max(weights) > 0 else 1
            widths = [(w / max_w) * 4 for w in weights]
            nx.draw_networkx_edges(
                G,
                pos,
                width=widths,
                arrowstyle="->",
                arrowsize=20,
                edge_color="#606060",
                alpha=0.5,
                ax=ax,
            )

        ax.set_title(
            f"Red Interbancaria (t={t})\nVerde: Solvente | Rojo: Default", fontsize=14
        )
        plt.axis("off")

        if t % 20 == 0:
            print(f"   Renderizando t={t}")

    print("Ensamblando GIF...")
    with imageio.get_writer(output_file, mode="I", duration=duration) as writer:
        for fname in frames:
            writer.append_data(imageio.imread(fname))  # type: ignore

    for fname in frames:
        os.remove(fname)
    os.rmdir(temp_dir)
    print("✅ GIF Interbancario completado.")


def generar_gif_macro(data, output_file, duration):
    if not data["has_macro"]:
        print("⚠️ No se encontraron datos macro (net_FB/firms) para este GIF.")
        return

    print(f"--- Generando GIF Macro-Financiero -> {output_file} ---")
    temp_dir = "temp_frames_macro"
    os.makedirs(temp_dir, exist_ok=True)

    df_net_BB = data["net_BB"]
    df_net_FB = data["net_FB"]
    df_banks = data["banks"]
    df_firms = data["firms"]

    # Layout Concéntrico
    nodes_B = list(range(p.B))
    nodes_F = list(range(p.B, p.B + p.F))

    pos = {}
    for i, node in enumerate(nodes_B):
        theta = 2 * np.pi * i / p.B
        pos[node] = np.array([np.cos(theta), np.sin(theta)]) * 0.3

    for i, node in enumerate(nodes_F):
        theta = 2 * np.pi * i / p.F
        pos[node] = np.array([np.cos(theta), np.sin(theta)]) * 1.0

    G_layout = nx.Graph()  # Dummy graph for drawing

    steps = sorted(df_banks["t"].unique())
    frames = []

    for t in steps:
        plt.figure(figsize=(10, 10), facecolor="white")
        ax = plt.gca()

        banks_t = df_banks[df_banks["t"] == t].set_index("id")
        firms_t = df_firms[df_firms["t"] == t].set_index("id")

        colors_B = [
            "#FF4444"
            if (i in banks_t.index and banks_t.loc[i, "eq"] < 0)
            else "#4444FF"
            for i in range(p.B)
        ]
        colors_F = [
            "#FF4444"
            if (i in firms_t.index and firms_t.loc[i, "eq"] < 0)
            else "#44FF44"
            for i in range(p.F)
        ]

        # Nodos
        nx.draw_networkx_nodes(
            G_layout,
            pos,
            nodelist=nodes_B,
            node_color=colors_B,
            node_size=400,
            edgecolors="black",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G_layout,
            pos,
            nodelist=nodes_F,
            node_color=colors_F,
            node_size=100,
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_labels(
            G_layout,
            pos,
            labels={n: str(n) for n in nodes_B},
            font_size=8,
            font_color="white",
            ax=ax,
        )

        # Aristas
        edges_bb_t = df_net_BB[df_net_BB["t"] == t]
        edges_fb_t = df_net_FB[df_net_FB["t"] == t]

        # B-B
        valid_bb = edges_bb_t[edges_bb_t["weight"] > 10.0]
        eb = list(zip(valid_bb["source"], valid_bb["target"]))
        if eb:
            nx.draw_networkx_edges(
                G_layout,
                pos,
                edgelist=eb,
                width=1.5,
                edge_color="black",
                alpha=0.6,
                ax=ax,
            )

        # F-B (Source=Firma 0..99 -> Layout ID = Source + p.B)
        valid_fb = edges_fb_t[edges_fb_t["weight"] > 10.0]
        ef = [(r + p.B, c) for r, c in zip(valid_fb["source"], valid_fb["target"])]
        if ef:
            nx.draw_networkx_edges(
                G_layout,
                pos,
                edgelist=ef,
                width=0.3,
                edge_color="gray",
                alpha=0.3,
                ax=ax,
            )

        plt.title(f"Ecosistema Macro-Financiero (t={t})", fontsize=15)
        plt.axis("off")

        fname = f"{temp_dir}/step_{int(t):03d}.png"
        plt.savefig(fname, dpi=80, bbox_inches="tight")
        plt.close()
        frames.append(fname)

        if t % 20 == 0:
            print(f"   Renderizando t={t}")

    print("Ensamblando GIF...")
    with imageio.get_writer(output_file, mode="I", duration=duration) as writer:
        for fname in frames:
            writer.append_data(imageio.imread(fname))  # type: ignore

    for fname in frames:
        os.remove(fname)
    os.rmdir(temp_dir)
    print("✅ GIF Macro completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de visualizaciones de red (GIF)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interbank", "macro", "both"],
        default="both",
        help="Tipo de visualización a generar",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directorio de datos de la simulación",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_DURATION,
        help="Duración por frame en segundos (mayor = más lento)",
    )

    args = parser.parse_args()

    print(f"Cargando datos de: {args.dir}")
    data = cargar_datos(args.dir)

    if data:
        os.makedirs("output_plots", exist_ok=True)
        if args.mode in ["interbank", "both"]:
            generar_gif_interbancario(
                data, "output_plots/red_interbancaria.gif", args.speed
            )

        if args.mode in ["macro", "both"]:
            generar_gif_macro(data, "output_plots/red_macro_financiera.gif", args.speed)
