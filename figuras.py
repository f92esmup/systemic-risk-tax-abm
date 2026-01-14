import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parametros import Param as p

OUTPUT_DIR = "output_plots"
DATA_DIR = "outputdata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style configuration
plt.style.use('ggplot')
PARAMS = {
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': (10, 6)
}
plt.rcParams.update(PARAMS)

def load_simulation_data(data_dir=DATA_DIR):
    """
    Crawls the outputdata directory and aggregates metrics for plotting.
    Now optimized for Consolidated Parquet Logs (SimulationLogger).
    """
    modes = ["NINGUNO", "TOBIN", "SRT"]
    results = {m: {
        "debtrank_profiles": [],
        "scatter_data": [],
        "total_losses": [],
        "cascade_sizes": [],
        "volumes": []
    } for m in modes}
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return results

    # Scan directories
    # Expected: run_{mode}_sim_{i}
    subdirs = [d for d in os.listdir(data_dir) if d.startswith("run_")]
    
    for run_dir in subdirs:
        # Parse mode from run_dir name
        parts = run_dir.split("_")
        if len(parts) < 4: continue
        
        mode = parts[1] # NINGUNO, TOBIN, SRT
        if mode not in results: continue
        
        full_run_path = os.path.join(data_dir, run_dir)


        # --- 1. Global Metrics (Sim-wide) ---
        globals_path = os.path.join(full_run_path, "globals.parquet")
        if os.path.exists(globals_path):
            run_metrics = pd.read_parquet(globals_path)
            
            # Fig 4a: Total Losses (Sum over all time)
            total_loss = run_metrics["contagion_loss"].sum()
            results[mode]["total_losses"].append(total_loss)
            
            # Fig 4b: Cascades (Individual events > 0)
            cascades = run_metrics[run_metrics["cascade_size"] > 0]["cascade_size"].tolist()
            if cascades:
                results[mode]["cascade_sizes"].extend(cascades)
            else:
                results[mode]["cascade_sizes"].append(0)
                
            # Fig 4c: Volumes (Mean over time)
            avg_vol = run_metrics["volume_ib"].mean()
            results[mode]["volumes"].append(avg_vol)

        # --- 2. DebtRank Profile (At step T_final) ---
        banks_path = os.path.join(full_run_path, "banks.parquet")
        if os.path.exists(banks_path):
            banks_df = pd.read_parquet(banks_path)
            if "t" in banks_df.columns and "dr" in banks_df.columns:
                last_t = banks_df["t"].max()
                # Get data for last step
                final_banks = banks_df[banks_df["t"] == last_t]
                # Extract DebtRank vector (sorted by bank id usually, but let's just take values)
                results[mode]["debtrank_profiles"].append(final_banks["dr"].values)

        # --- 3. SRT Scatter (Delta EL vs Loan Size) ---
        # Data needed: Transactions list (Loan Amount vs Marginal Delta EL)
        
        trans_path = os.path.join(full_run_path, "transactions.parquet")

        if os.path.exists(trans_path):
            trans_df = pd.read_parquet(trans_path)
            
            # Filter for SRT relevant data (where marginal_sr is recorded)
            # Depending on mode, it might be 0, but we want to plot if it exists.
            if "amount" in trans_df.columns and "marginal_sr" in trans_df.columns:
                # Filter out tiny loans or zero deltas if desired, but Fig 3d plots everything usually
                mask = (trans_df["amount"] > 0)
                valid = trans_df[mask]
                
                if not valid.empty:
                    results[mode]["scatter_data"].append((
                        valid["amount"].values, 
                        valid["marginal_sr"].values
                    ))

    return results

def plot_fig_3b(results, colors):
    plt.figure()
    B = p.B
    x = np.arange(1, B + 1)
    bar_width = 0.25
    offsets = {"NINGUNO": -bar_width, "TOBIN": 0, "SRT": bar_width}
    
    for mode in results:
        data = results[mode]["debtrank_profiles"]
        if not data: continue
        
        # Handle varying lengths if any error, but should be B
        # Stack
        data_array = np.array(data)
        if data_array.ndim != 2: continue # Skip if empty
        
        sorted_runs = np.sort(data_array, axis=1)[:, ::-1]
        avg_profile = np.mean(sorted_runs, axis=0)
        std_profile = np.std(sorted_runs, axis=0)
        
        plt.bar(x + offsets[mode], avg_profile, width=bar_width, 
                color=colors[mode], label=mode, alpha=0.8, yerr=std_profile, capsize=2)
        
    plt.xlabel('Bank Rank (by DebtRank)')
    plt.ylabel('DebtRank $R_i$')
    plt.title('Fig 3b: Perfil de Riesgo Sistémico')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_3b_DebtRank.png")
    plt.close()

def plot_fig_3d(results, colors):
    plt.figure()
    has_data = False
    for mode in results:
        if not results[mode]["scatter_data"]: continue
        
        all_l, all_d = [], []
        for l, d in results[mode]["scatter_data"]:
            all_l.extend(l)
            all_d.extend(d)
        
        if not all_l: continue
        
        # [FIX] Filter out zeros/negative values for log-log plot to avoid errors
        all_l = np.array(all_l)
        all_d = np.array(all_d)
        mask = (all_l > 0) & (all_d > 0)
        
        if np.sum(mask) == 0: continue

        plt.scatter(all_l[mask], all_d[mask], color=colors[mode], alpha=0.3, label=mode, s=10, edgecolors='none')
        has_data = True
        
    if has_data:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Loan Size (Log)')
        plt.ylabel(r'$\Delta EL^{syst}$ (Log)')
        plt.title('Fig 3d: Contribución Marginal al Riesgo')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/Fig_3d_Scatter.png")
    else:
        print("⚠️ ADVERTENCIA: No hay datos válidos para el gráfico 3D (Posible DebtRank=0 o no SRT run)")
    plt.close()

def plot_fig_4(results, colors):
    # 4a Losses
    plt.figure()
    for mode in results:
        if not results[mode]["total_losses"]: continue
        plt.hist(results[mode]["total_losses"], bins=15, color=colors[mode], alpha=0.5, label=mode, density=True)
    plt.title('Fig 4a: Pérdidas Totales')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/Fig_4a_Losses.png")
    plt.close()

    # 4b Cascades
    plt.figure()
    bins = np.arange(0, p.B + 2) - 0.5
    for mode in results:
        if not results[mode]["cascade_sizes"]: continue
        plt.hist(results[mode]["cascade_sizes"], bins=bins, color=colors[mode], alpha=0.5, label=mode, density=True, histtype='stepfilled')
    plt.title('Fig 4b: Tamaño de Cascadas')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/Fig_4b_Cascades.png")
    plt.close()

    # 4c Volume
    plt.figure()
    for mode in results:
        if not results[mode]["volumes"]: continue
        plt.hist(results[mode]["volumes"], bins=15, color=colors[mode], alpha=0.5, label=mode, density=True)
    plt.title('Fig 4c: Volumen Interbancario')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/Fig_4c_Volume.png")
    plt.close()

def generar_graficas():
    print("--- Generando Gráficos desde Parquet ---")
    colors = {"NINGUNO": "red", "TOBIN": "blue", "SRT": "green"}
    
    print("Cargando datos...")
    data = load_simulation_data()
    
    print("Plotting Fig 3b...")
    plot_fig_3b(data, colors)
    
    print("Plotting Fig 3d...")
    plot_fig_3d(data, colors)
    
    print("Plotting Fig 4...")
    plot_fig_4(data, colors)
    
    print(f"Listo. Resultados en ./{OUTPUT_DIR}")

if __name__ == "__main__":
    generar_graficas()