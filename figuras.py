import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parametros import Param as p

OUTPUT_DIR = "output_plots"

DATA_DIR = "outputdata"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Configuración de estilo

plt.style.use("ggplot")

PARAMS = {
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": (10, 6),
}

plt.rcParams.update(PARAMS)


MODES_ORDER = ["NINGUNO", "TOBIN", "SRT"]

MODE_LABELS = {"NINGUNO": "No Tax", "TOBIN": "Tobin Tax", "SRT": "SRT"}

MODE_COLORS = {"NINGUNO": "red", "TOBIN": "blue", "SRT": "green"}


def load_simulation_data(data_dir=DATA_DIR):
    """

    Explora el directorio outputdata y agrega métricas para graficar.

    Ahora optimizado para registros Parquet consolidados (SimulationLogger).

    """

    results = {
        m: {
            "debtrank_profiles": [],
            "scatter_data_x": [],  # Tamaño relativo del préstamo [%]
            "scatter_data_y": [],  # Delta EL relativo [%]
            "total_losses": [],
            "cascade_sizes": [],
            "volumes": [],
        }
        for m in MODES_ORDER
    }

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")

        return results

    # Escanear directorios

    subdirs = [d for d in os.listdir(data_dir) if d.startswith("run_")]

    for run_dir in subdirs:
        # Analizar modo desde el nombre de run_dir

        parts = run_dir.split("_")

        if len(parts) < 4:
            continue

        mode = parts[1]  # NINGUNO, TOBIN, SRT

        if mode not in results:
            continue

        full_run_path = os.path.join(data_dir, run_dir)

        # --- 1. Métricas Globales (Toda la simulación) ---

        globals_path = os.path.join(full_run_path, "globals.parquet")

        if os.path.exists(globals_path):
            run_metrics = pd.read_parquet(globals_path)

            # Fig 4a: Pérdidas Totales (Suma sobre todo el tiempo)

            total_loss = run_metrics["contagion_loss"].sum()

            results[mode]["total_losses"].append(total_loss)

            # Fig 4b: Cascadas (Eventos individuales > 0)

            cascades = run_metrics[run_metrics["cascade_size"] > 0][
                "cascade_size"
            ].tolist()

            if cascades:
                results[mode]["cascade_sizes"].extend(cascades)

            else:
                results[mode]["cascade_sizes"].append(0)

            # Fig 4c: Volúmenes (Media sobre el tiempo)

            avg_vol = run_metrics["volume_ib"].mean()

            results[mode]["volumes"].append(avg_vol)

        # --- 2. Perfil de DebtRank (En el paso T_final) ---

        banks_path = os.path.join(full_run_path, "banks.parquet")

        if os.path.exists(banks_path):
            banks_df = pd.read_parquet(banks_path)

            if "t" in banks_df.columns and "dr" in banks_df.columns:
                max_t = banks_df["t"].max()

                # Tomar un paso intermedio estable, o el promedio de los últimos 50 pasos

                target_t = max(0, max_t - 10)

                final_banks = banks_df[banks_df["t"] == target_t]

                # Extraer vector DebtRank

                if not final_banks.empty:
                    results[mode]["debtrank_profiles"].append(
                        np.array(final_banks["dr"])
                    )

        # --- 3. Dispersión SRT (Delta EL vs Tamaño del Préstamo) ---

        trans_path = os.path.join(full_run_path, "transactions.parquet")

        if os.path.exists(trans_path):
            trans_df = pd.read_parquet(trans_path)

            if (
                "amount" in trans_df.columns
                and "marginal_sr" in trans_df.columns
                and "t" in trans_df.columns
            ):
                # Agrupar por paso de tiempo para calcular tamaño relativo por estado de mercado

                for _, group in trans_df.groupby("t"):
                    total_vol_t = group["amount"].sum()

                    if total_vol_t < 1e-9:
                        continue

                    # Tamaño relativo del préstamo [%] = (Cantidad / Volumen total en t) * 100

                    # Delta EL relativo [%] = marginal_sr * 100 (asumiendo que H es 0-1)

                    mask = group["amount"] > 0

                    valid = group[mask]

                    if not valid.empty:
                        rel_loans = (np.array(valid["amount"]) / total_vol_t) * 100

                        rel_deltas = np.array(valid["marginal_sr"]) * 100

                        results[mode]["scatter_data_x"].extend(rel_loans)

                        results[mode]["scatter_data_y"].extend(rel_deltas)

    return results


def plot_fig_3b(results):
    plt.figure()

    B = p.B

    x = np.arange(1, B + 1)

    bar_width = 0.25

    offsets = [-bar_width, 0, bar_width]

    # Ordenar modos para alinear con los desplazamientos (offsets)

    for i, mode in enumerate(MODES_ORDER):
        data = results[mode]["debtrank_profiles"]

        if not data:
            continue

        # Apilar para promediar

        data_array = np.array(data)

        if data_array.ndim != 2:
            continue

        # Ordenar cada ejecución descendente (regla Rank-Size)

        sorted_runs = np.sort(data_array, axis=1)[:, ::-1]

        avg_profile = np.mean(sorted_runs, axis=0)

        # std_profile = np.std(sorted_runs, axis=0)

        plt.bar(
            x + offsets[i],
            avg_profile,
            width=bar_width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
            alpha=0.8,
        )

    plt.xlabel("Bank Rank (by DebtRank)")

    plt.ylabel("DebtRank $R_i$")

    plt.title("Fig 3b: Perfil de Riesgo Sistémico (Rank-Ordered)")

    plt.legend()

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/Fig_3b_DebtRank.png")

    plt.close()


def plot_fig_3d(results):
    plt.figure()

    for mode in MODES_ORDER:
        xs = results[mode]["scatter_data_x"]

        ys = results[mode]["scatter_data_y"]

        if not xs:
            continue

        xs = np.array(xs)

        ys = np.array(ys)

        # La escala logarítmica necesita valores positivos

        mask = (xs > 0) & (ys > 0)

        if np.sum(mask) == 0:
            continue

        plt.scatter(
            xs[mask],
            ys[mask],
            color=MODE_COLORS[mode],
            alpha=0.3,
            label=MODE_LABELS[mode],
            s=15,
            edgecolors="none",
        )

    plt.xscale("log")

    plt.yscale("log")

    plt.xlabel("Relative Loan Size [%]")

    plt.ylabel(r"Relative $\Delta EL^{syst}$ [%]")

    plt.title("Fig 3d: Contribución Marginal al Riesgo")

    plt.legend()

    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/Fig_3d_Scatter.png")

    plt.close()


def plot_fig_4(results):
    # Preparar listas de datos para graficar en paralelo

    # Ayudante para limpiar Nones o vacíos

    def get_clean_list(metric_key):
        data_list = []

        labels = []

        colors = []

        for mode in MODES_ORDER:
            d = results[mode][metric_key]

            if d:
                data_list.append(d)

                labels.append(MODE_LABELS[mode])

                colors.append(MODE_COLORS[mode])

            else:
                # Marcador de posición para mantener la alineación si es necesario, o saltar

                # El histograma de Matplotlib maneja bien diferentes longitudes

                pass

        return data_list, labels, colors

    # 4a Pérdidas

    plt.figure()

    data_list, labels, colors = get_clean_list("total_losses")

    if data_list:
        plt.hist(
            data_list, bins=10, color=colors, label=labels, density=True, histtype="bar"
        )

    plt.title("Fig 4a: Pérdidas Totales a Bancos")

    plt.xlabel("Total Loss")

    plt.ylabel("Frequency")

    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/Fig_4a_Losses.png")

    plt.close()

    # 4b Cascadas

    plt.figure()

    data_list, labels, colors = get_clean_list("cascade_sizes")

    if data_list:
        bins = np.arange(0, p.B + 2) - 0.5

        plt.hist(
            data_list,
            bins=bins.tolist(),
            color=colors,
            label=labels,
            density=True,
            histtype="bar",
        )

    plt.title("Fig 4b: Tamaño de Cascadas")

    plt.xlabel("Banks defaulted in one step")

    plt.ylabel("Frequency")

    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/Fig_4b_Cascades.png")

    plt.close()

    # 4c Volumen mercado interbancario

    plt.figure()

    data_list, labels, colors = get_clean_list("volumes")

    if data_list:
        plt.hist(
            data_list, bins=10, color=colors, label=labels, density=True, histtype="bar"
        )

    plt.title("Fig 4c: Volumen Mercado Interbancario")

    plt.xlabel("Avg Transaction Volume")

    plt.ylabel("Frequency")

    plt.legend()

    plt.savefig(f"{OUTPUT_DIR}/Fig_4c_Volume.png")

    plt.close()


def generar_graficas():
    print("--- Generando Gráficos Finales ---")

    print("Cargando datos...")
    data = load_simulation_data()

    print("Generando Fig 3b (DebtRank Profile)...")
    plot_fig_3b(data)

    print("Generando Fig 3d (Scatter Relative)...")
    plot_fig_3d(data)

    print("Generando Fig 4 (Distributions)...")
    plot_fig_4(data)

    print(f"Listo. Resultados guardados en ./{OUTPUT_DIR}")


if __name__ == "__main__":
    generar_graficas()
