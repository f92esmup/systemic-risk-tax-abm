import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import funciones as fn
from parametros import Parametros

plt.style.use("default")
sns.set_theme(style="whitegrid")


def obtener_datos_seguros(archivo_npz):
    """Intenta cargar datos usando claves en español o inglés."""
    d = np.load(archivo_npz)

    # Mapeo de claves: (Español, Inglés)
    claves_map = {
        "L": ["matriz_interbancaria", "L_bb"],
        "Bancos": ["estado_bancos", "banks_state"],
    }

    resultado = {}

    for meta_key, opciones in claves_map.items():
        found = False
        for key in opciones:
            if key in d:
                resultado[meta_key] = d[key]
                found = True
                break
        if not found:
            raise KeyError(
                f"No se encontró datos para {meta_key} (buscado: {opciones})"
            )

    return resultado["L"], resultado["Bancos"]


def calculate_systemic_risk_metrics(L, equity, assets):
    B = L.shape[0]
    # Pasivos Interbancarios
    interbank_liabilities = np.sum(L, axis=1)  # Filas = Deudores

    # Estimación de Pasivos Totales (Assets - Equity)
    # Protegemos contra equity <= 0 para evitar divisiones
    equity_safe = np.maximum(equity, 0.001)
    total_liabilities = np.maximum(0, assets - equity)

    leverage = total_liabilities / equity_safe

    # Probabilidad de Default (Eq A4)
    p_default = 0.01 * np.tanh(Parametros.K_mu * leverage)

    v = assets
    V_total = np.sum(v)

    # DebtRank Base
    R_base = fn.calcular_debtrank(L, equity, v)

    # Expected Loss Base
    EL_base = np.sum(p_default * R_base) * V_total

    marginals = []
    rows, cols = np.where(L > 1e-5)  # Solo enlaces significativos

    for i, j in zip(rows, cols):
        loan_val = L[i, j]

        # Análisis Contrafactual: Remover préstamo
        L_temp = L.copy()
        L_temp[i, j] = 0.0

        R_temp = fn.calcular_debtrank(L_temp, equity, v)
        EL_temp = np.sum(p_default * R_temp) * V_total

        delta = EL_base - EL_temp

        # Tamaño relativo (Loan / Equity del Prestamista)
        lender_equity = equity_safe[j]
        rel_size = loan_val / lender_equity

        marginals.append((rel_size, delta))

    return R_base, marginals


def plot_figure3_corrected():
    modes = ["none", "tobin", "srt"]
    colors = {"none": "red", "tobin": "blue", "srt": "green"}
    labels = {"none": "No Tax", "tobin": "Tobin Tax", "srt": "SRT"}

    data_store = {}
    print(">>> Generando Figura 3 (Versión Robusta) <<<")

    hay_datos = False

    for mode in modes:
        files = sorted(glob.glob(f"output_data/{mode}/*.npz"))
        if not files:
            print(f" [WARN] No hay datos para el modo '{mode}'")
            continue

        try:
            # Usar el último run disponible
            f = files[-1]
            print(f" Procesando {mode}: {f}")

            L_hist, Banks_hist = obtener_datos_seguros(f)

            # Usar el último paso temporal
            L = L_hist[-1]
            banks = Banks_hist[-1]

            # Verificar si hay actividad
            if np.sum(L) == 0:
                print(
                    f" [WARN] El modo '{mode}' tiene VOLUMEN 0 en el paso final. Gráficas serán vacías."
                )

            # Indices: 1=Equity, 7=Assets (asegurarse que coinciden con simulacion.py)
            equity = banks[:, 1]
            assets = banks[:, 7]

            R, margs = calculate_systemic_risk_metrics(L, equity, assets)
            data_store[mode] = {"R": R, "marginals": margs}
            hay_datos = True

        except Exception as e:
            print(f" [ERROR] Fallo procesando {mode}: {e}")

    if not hay_datos:
        print(
            " [!] No se pudieron extraer datos válidos de ningún modo. Abortando gráficas."
        )
        return

    # --- PLOT 1: Perfiles DebtRank ---
    plt.figure(figsize=(10, 6))

    # Ordenamiento base (No Tax)
    if "none" in data_store:
        rank_indices = np.argsort(data_store["none"]["R"])[::-1]
    else:
        rank_indices = np.arange(Parametros.B)

    x_axis = np.arange(1, Parametros.B + 1)

    for mode in modes:
        if mode in data_store:
            R = data_store[mode]["R"]
            # Si R es todo ceros, avisar
            if np.sum(R) == 0:
                print(
                    f" [INFO] DebtRank es 0 para todos en '{mode}' (Sistema muy seguro o sin deuda)."
                )

            plt.plot(
                x_axis,
                R[rank_indices],
                marker="o",
                label=labels[mode],
                color=colors[mode],
            )

    plt.xlabel("Bancos (Ordenados por Riesgo No-Tax)")
    plt.ylabel("DebtRank")
    plt.title("Perfil de Riesgo Sistémico")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs("output_data/graficas_finales", exist_ok=True)
    plt.savefig("output_data/graficas_finales/figura3_ab_debtrank.png")
    plt.close()

    # --- PLOT 2: Scatter Marginal ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)

    for i, mode in enumerate(modes):
        ax = axes[i]
        if mode in data_store:
            margs = data_store[mode]["marginals"]
            if margs:
                loans, deltas = zip(*margs)
                ax.scatter(loans, deltas, color=colors[mode], alpha=0.6)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{labels[mode]} (n={len(loans)})")
            else:
                ax.text(0.5, 0.5, "Sin Préstamos", ha="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Sin Datos", ha="center", transform=ax.transAxes)

    axes[0].set_ylabel("Contribución Marginal (Delta EL)")
    axes[1].set_xlabel("Tamaño Relativo del Préstamo (Log)")

    plt.tight_layout()
    plt.savefig("output_data/graficas_finales/figura3_cd_marginal.png")
    plt.close()
    print(">>> Gráficas guardadas en output_data/graficas_finales/ <<<")


if __name__ == "__main__":
    plot_figure3_corrected()

