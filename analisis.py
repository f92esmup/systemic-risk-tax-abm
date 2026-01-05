import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import funciones as fn
from parametros import Parametros

# Configure plotting style
plt.style.use("default")
sns.set_theme(style="whitegrid")


def obtener_datos_seguros(archivo_npz):
    """Loads simulation data safely, handling key variations."""
    try:
        d = np.load(archivo_npz)
    except Exception as e:
        raise ValueError(f"Cannot load file {archivo_npz}: {e}")

    # Key mapping: (Meta-Key -> [Possible Keys in NPZ])
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
            # Check available keys for debugging
            raise KeyError(f"Key for {meta_key} not found. Available: {list(d.keys())}")

    return resultado["L"], resultado["Bancos"]


def calculate_systemic_risk_metrics(L, equity, assets):
    """Calculates DebtRank and Marginal Contributions."""
    # Ensure equity is positive to avoid division by zero
    equity_safe = np.maximum(equity, 0.001)

    # DebtRank Base
    # v = assets (proxy for economic value)
    v = assets
    V_total = np.sum(v)
    if V_total == 0:
        return np.zeros_like(equity), []

    R_base = fn.calcular_debtrank(L, equity, v)

    # Probability of Default (approximate for Expected Loss)
    # Total Liabilities approx Assets - Equity
    total_liabilities = np.maximum(0, assets - equity)
    leverage = total_liabilities / equity_safe
    p_default = 0.01 * np.tanh(Parametros.K_mu * leverage)

    # Expected Loss Base
    EL_base = np.sum(p_default * R_base) * V_total

    marginals = []
    rows, cols = np.where(L > 1e-5)  # Only significant links

    for i, j in zip(rows, cols):
        loan_val = L[i, j]

        # Counterfactual: Remove loan
        L_temp = L.copy()
        L_temp[i, j] = 0.0

        R_temp = fn.calcular_debtrank(L_temp, equity, v)
        EL_temp = np.sum(p_default * R_temp) * V_total

        delta = EL_base - EL_temp

        # Relative Size (Loan / Lender Equity)
        lender_equity = equity_safe[j]
        rel_size = loan_val / lender_equity

        marginals.append((rel_size, delta))

    return R_base, marginals


def generar_figura_3(output_folder_base="output_data"):
    """
    Generates Figure 3 (DebtRank Profiles and Marginal Contributions).
    Reads .npz files from output_folder_base/{mode}/
    """
    modes = ["none", "tobin", "srt"]
    colors = {"none": "red", "tobin": "blue", "srt": "green"}
    labels = {"none": "No Tax", "tobin": "Tobin Tax", "srt": "SRT"}

    data_store = {}
    print("\n>>> Generating Figure 3 (Systemic Risk Profiles) <<<")

    os.makedirs("output_data/graficas_finales", exist_ok=True)
    hay_datos = False

    for mode in modes:
        search_path = os.path.join(output_folder_base, mode, "*.npz")
        files = sorted(glob.glob(search_path))

        if not files:
            print(f" [WARN] No data found for mode '{mode}' in {search_path}")
            continue

        # Process valid runs (iterate backwards to find a populated one)
        found_good_run = False
        for f in reversed(files):
            try:
                L_hist, Banks_hist = obtener_datos_seguros(f)
                # Use last step
                L = L_hist[-1]
                if np.sum(L) > 1e-3:  # Ensure some volume exists
                    banks = Banks_hist[-1]
                    # IDX_BANK_EQUITY = 1, IDX_BANK_TOTAL_ASSETS = 7 (Simulacion convention)
                    equity = banks[:, 1]
                    assets = banks[:, 7]

                    R, margs = calculate_systemic_risk_metrics(L, equity, assets)
                    data_store[mode] = {"R": R, "marginals": margs}
                    hay_datos = True
                    found_good_run = True
                    print(f" Mode '{mode}': Processed {os.path.basename(f)}")
                    break
            except Exception as e:
                print(f" [DEBUG] Skipping {f}: {e}")
                continue

        if not found_good_run:
            print(f" [WARN] Mode '{mode}': No valid run with volume found.")

    if not hay_datos:
        print(" [!] No valid data found for any mode. Skipping Figure 3.")
        return

    # --- PLOT 3A: DebtRank Profile ---
    plt.figure(figsize=(10, 6))

    # Base ordering (No Tax) for x-axis
    if "none" in data_store:
        rank_indices = np.argsort(data_store["none"]["R"])[::-1]
    else:
        # Fallback if 'none' is missing
        available_keys = list(data_store.keys())
        rank_indices = np.argsort(data_store[available_keys[0]]["R"])[::-1]

    x_axis = np.arange(1, Parametros.B + 1)

    for mode in modes:
        if mode in data_store:
            R = data_store[mode]["R"]
            plt.plot(
                x_axis,
                R[rank_indices],
                marker="o",
                label=labels[mode],
                color=colors[mode],
                alpha=0.8,
            )

    plt.xlabel("Banks (Sorted by Risk)")
    plt.ylabel("DebtRank")
    plt.title("Systemic Risk Profile (DebtRank)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("output_data/graficas_finales/figura3_ab_debtrank.png")
    plt.close()

    # --- PLOT 3B: Marginal Contribution vs Size ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)

    for i, mode in enumerate(modes):
        ax = axes[i]
        if mode in data_store:
            margs = data_store[mode]["marginals"]
            if margs:
                loans, deltas = zip(*margs)
                ax.scatter(loans, deltas, color=colors[mode], alpha=0.6, s=15)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"{labels[mode]} (n={len(loans)})")
            else:
                ax.text(
                    0.5, 0.5, "No Active Loans", ha="center", transform=ax.transAxes
                )
        else:
            ax.text(0.5, 0.5, "No Data", ha="center", transform=ax.transAxes)

    axes[0].set_ylabel("Marginal Contribution (Delta EL)")
    axes[1].set_xlabel("Relative Loan Size (Log Scale)")

    plt.tight_layout()
    plt.savefig("output_data/graficas_finales/figura3_cd_marginal.png")
    plt.close()
    print(">>> Figure 3 saved in output_data/graficas_finales/")


def generar_figura_4(results_dict):
    """
    Generates Figure 4 (Distributions of Loss, Cascades, Volume).
    results_dict format: {'mode': {'losses': [], 'cascades': [], 'volumes': []}}
    """
    print("\n>>> Generating Figure 4 (Comparative Distributions) <<<")
    os.makedirs("output_data/graficas_finales", exist_ok=True)

    modes = ["none", "tobin", "srt"]
    colors = {"none": "red", "tobin": "blue", "srt": "green"}
    labels = {"none": "No Tax", "tobin": "Tobin Tax", "srt": "SRT"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 4A: Losses Distribution (KDE) ---
    ax_loss = axes[0]
    has_loss_data = False
    for mode in modes:
        if mode in results_dict and len(results_dict[mode]["losses"]) > 1:
            try:
                sns.kdeplot(
                    results_dict[mode]["losses"],
                    ax=ax_loss,
                    label=labels[mode],
                    fill=True,
                    log_scale=(True, False),  # Log x-axis for losses
                    color=colors[mode],
                    warn_singular=False,
                )
                has_loss_data = True
            except Exception as e:
                print(f" [WARN] Could not plot KDE for {mode}: {e}")

    ax_loss.set_title("Distribution of Systemic Losses (L)")
    ax_loss.set_xlabel("Loss Amount (Log Scale)")
    if has_loss_data:
        ax_loss.legend()

    # --- 4B: Cascade Size (Histogram) ---
    ax_casc = axes[1]
    bins = np.arange(1, Parametros.B + 2) - 0.5  # Center bars on integers

    for mode in ["none", "srt"]:  # Usually compare baseline vs SRT
        if mode in results_dict and len(results_dict[mode]["cascades"]) > 0:
            ax_casc.hist(
                results_dict[mode]["cascades"],
                bins=bins,
                alpha=0.5,
                label=labels[mode],
                color=colors[mode],
                density=True,
            )

    ax_casc.set_title("Cascade Size Distribution")
    ax_casc.set_xlabel("Number of Defaults")
    ax_casc.legend()

    # --- 4C: Volume (Boxplot) ---
    ax_vol = axes[2]
    data_vol = []
    labels_vol = []
    palette_vol = []

    for mode in modes:
        if mode in results_dict and len(results_dict[mode]["volumes"]) > 0:
            data_vol.append(results_dict[mode]["volumes"])
            labels_vol.append(labels[mode])
            palette_vol.append(colors[mode])

    if data_vol:
        sns.boxplot(data=data_vol, ax=ax_vol, palette=palette_vol)
        ax_vol.set_xticklabels(labels_vol)
        ax_vol.set_title("Interbank Market Volume")
        ax_vol.set_ylabel("Average Volume per Step")

    plt.tight_layout()
    plt.savefig("output_data/graficas_finales/figura4_resultados.png", dpi=300)
    plt.close()
    print(">>> Figure 4 saved in output_data/graficas_finales/")
