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
    # v = Total Interbank Liabilities (Row Sum)
    v = np.sum(L, axis=1)
    
    # If no interbank debt exists, v is 0. Avoid issues.
    if np.sum(v) == 0:
         # Fallback to Assets if network is empty? Or just 0.
         # If v=0, DebtRank is 0.
         pass

    V_total = np.sum(v)
    if V_total == 0:
        return np.zeros_like(equity), [], 0.0

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

    return R_base, marginals, EL_base


def generar_figura_3(output_folder_base="output_data"):
    """
    Generates Figure 3 (DebtRank Profiles and Marginal Contributions).
    (b) R_i vs Rank (Bar chart)
    (d) Relative Delta EL vs Relative Loan Size (Scatter)
    """
    modes = ["none", "tobin", "srt"]
    colors = {"none": "red", "tobin": "blue", "srt": "green"}
    labels = {"none": "No Tax", "tobin": "Tobin Tax", "srt": "SRT"}
    bar_width = 0.25

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

                    R, margs, EL_base = calculate_systemic_risk_metrics(L, equity, assets)
                    data_store[mode] = {"R": R, "marginals": margs, "EL_base": EL_base}
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

    # Create 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- PLOT 3B: DebtRank Profile (Bar Chart) ---
    ax_bar = axes[0]
    
    # Base ordering (No Tax) for x-axis
    if "none" in data_store:
        base_R = data_store["none"]["R"]
        rank_indices = np.argsort(base_R)[::-1]
    else:
        available_keys = list(data_store.keys())
        rank_indices = np.argsort(data_store[available_keys[0]]["R"])[::-1]

    # X positions
    x = np.arange(len(rank_indices))

    # Plot bars
    offsets = {"none": -bar_width, "tobin": 0, "srt": bar_width}
    
    for mode in modes:
        if mode in data_store:
            R = data_store[mode]["R"]
            # Reorder R based on the 'none' ranking
            R_sorted = R[rank_indices]
            
            ax_bar.bar(x + offsets[mode], R_sorted, width=bar_width, label=labels[mode], color=colors[mode], alpha=0.7)

    ax_bar.set_xlabel("Banks (Sorted by Risk in 'No Tax')")
    ax_bar.set_ylabel("DebtRank ($R_i$)")
    ax_bar.set_title("(b) Model results for $R_i$")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(x + 1) # 1-based indexing for display
    ax_bar.legend()
    ax_bar.grid(True, axis='y', alpha=0.3)

    # --- PLOT 3D: Marginal Contributions (Scatter) ---
    ax_scat = axes[1]
    
    for mode in modes:
        if mode in data_store:
            margs = data_store[mode]["marginals"]
            EL_base = data_store[mode]["EL_base"]
            
            if margs and EL_base > 0:
                loans_rel, deltas = zip(*margs)
                
                # Convert to percentages
                x_pct = np.array(loans_rel) * 100
                y_pct = (np.array(deltas) / EL_base) * 100
                
                ax_scat.scatter(x_pct, y_pct, color=colors[mode], alpha=0.6, s=20, label=labels[mode])
            else:
                pass # No data or EL_base 0

    ax_scat.set_xlabel("Relative Loan Size [%] ($L_{ij} / E_j$)")
    ax_scat.set_ylabel("Relative Increment EL sys [%]")
    ax_scat.set_title("(d) Marginal contributions")
    ax_scat.set_xscale("log")
    ax_scat.set_yscale("log")
    ax_scat.legend()
    ax_scat.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig("output_data/graficas_finales/figura3_combined.png", dpi=150)
    plt.close()
    print(">>> Figure 3 saved in output_data/graficas_finales/")


def generar_figura_4(results_dict):
    """
    Generates Figure 4 (Distributions of Loss, Cascades, Volume).
    (a) Distribution of total losses L
    (b) Distribution of cascade sizes C
    (c) Distribution of total transaction volume V
    """
    print("\n>>> Generating Figure 4 (Comparative Distributions) <<<")
    os.makedirs("output_data/graficas_finales", exist_ok=True)

    modes = ["none", "tobin", "srt"]
    colors = {"none": "red", "tobin": "blue", "srt": "green"}
    labels = {"none": "No Tax", "tobin": "Tobin Tax", "srt": "SRT"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Helper function to prepare data for side-by-side (dodged) histograms
    def prepare_hist_data(metric_key):
        data = []
        lbls = []
        cols = []
        for mode in modes:
            if mode in results_dict and len(results_dict[mode][metric_key]) > 0:
                data.append(results_dict[mode][metric_key])
                lbls.append(labels[mode])
                cols.append(colors[mode])
        return data, lbls, cols

    # --- 4A: Losses Distribution L ---
    ax_loss = axes[0]
    data_L, lbls_L, cols_L = prepare_hist_data("losses")
    
    if data_L:
        ax_loss.hist(
            data_L,
            bins=20,
            label=lbls_L,
            color=cols_L,
            density=True,
            histtype='bar',  # Side-by-side bars
            log=True         # Log scale for y-axis
        )
        
    ax_loss.set_title("(a) Distribution of total losses $L$")
    ax_loss.set_xlabel("Total Losses to Banks")
    ax_loss.set_ylabel("Frequency (Density, Log Scale)")
    ax_loss.legend()
    ax_loss.grid(True, axis='y', alpha=0.2)

    # --- 4B: Cascade Size Distribution C ---
    ax_casc = axes[1]
    data_C, lbls_C, cols_C = prepare_hist_data("cascades")
    bins_c = np.arange(1, Parametros.B + 2) - 0.5
    
    if data_C:
        ax_casc.hist(
            data_C,
            bins=bins_c,
            label=lbls_C,
            color=cols_C,
            density=True,
            histtype='bar' # Side-by-side bars
        )
        
    ax_casc.set_title("(b) Distribution of cascade sizes $C$")
    ax_casc.set_xlabel("Number of Defaulting Banks")
    ax_casc.set_ylabel("Frequency (Density)")
    ax_casc.set_xticks(np.arange(0, Parametros.B + 1, 5))
    ax_casc.legend()
    ax_casc.grid(True, axis='y', alpha=0.2)

    # --- 4C: Volume Distribution V ---
    ax_vol = axes[2]
    data_V, lbls_V, cols_V = prepare_hist_data("volumes")
    
    if data_V:
        ax_vol.hist(
            data_V,
            bins=20,
            label=lbls_V,
            color=cols_V,
            density=True,
            histtype='bar' # Side-by-side bars
        )
        
    ax_vol.set_title("(c) Distribution of transaction volume $V$")
    ax_vol.set_xlabel("Total IB Transaction Volume")
    ax_vol.set_ylabel("Frequency (Density)")
    ax_vol.legend()
    ax_vol.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig("output_data/graficas_finales/figura4_completa.png", dpi=300)
    plt.close()
    print(">>> Figure 4 saved in output_data/graficas_finales/")
