import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from simulation import CRISIS_Model
from parameters import Params
import functions as fn

# Configure Plots
sns.set_theme(style="whitegrid")


def run_experiment(tax_mode="none", tax_param=0.0, n_runs=10, steps=200):
    """
    Run N simulations with specific tax configuration.
    Returns aggregated data for plotting.
    """
    print(f"Starting Experiment: Mode={tax_mode}, Param={tax_param}, Runs={n_runs}")

    all_losses = []
    all_cascade_sizes = []
    all_volumes = []

    # Run loop
    for r in tqdm(range(n_runs), desc=f"{tax_mode.upper()} Runs"):
        # Unique run_id for file naming needs to include mode to avoid collisions if we run modes sequentially
        # OR we just rely on r and separate folders?
        # User prompt said "run_00042.npz" in "output_data".
        # If we run 3 experiments (None, SRT, Tobin) we might overwrite run_00000.
        # Let's add a prefix or separate folders.
        # For strict compliance with prompt "flush_data(run_id, output_folder)":
        # We will use "output_data/{tax_mode}" to keep them clean.

        model = CRISIS_Model(seed=1000 + r, tax_mode=tax_mode, tax_param=tax_param)

        for t in range(steps):
            model.run_step()

            # Collect Metrics (Post-step)
            # 1. Losses (Capital destroyed in cascades)
            if model.current_step_loss > 0:
                all_losses.append(model.current_step_loss)

            # 2. Cascade Sizes
            if model.current_step_defaults > 0:
                all_cascade_sizes.append(model.current_step_defaults)

            # 3. Volume
            all_volumes.append(model.current_step_volume)

        # --- FLUSH DATA TO DISK ---
        # Save run data
        # We use a global counter or just 'r'. But 'r' resets for each mode.
        # Let's use a unique name or careful folder management.
        output_subfolder = f"output_data/{tax_mode}"
        model.save_run_to_disk(run_id=r, folder=output_subfolder)

    return {
        "losses": np.array(all_losses),
        "cascades": np.array(all_cascade_sizes),
        "volumes": np.array(all_volumes),
    }


def main():
    # Configuration matches Paper (roughly)
    # T=500, Runs=50 (Paper uses more, but we want speed for demo)
    # But for prompt compliance, I will try n_runs=15 to balance speed/results
    N_RUNS = 200
    STEPS = 500  # Paper uses 500-1000. 200 should be enough to see crises.

    # 1. Run Baseline (No Tax)
    data_base = run_experiment("none", 0.0, n_runs=N_RUNS, steps=STEPS)

    # 2. Run SRT (Systemic Risk Tax)
    # Zeta = 0.02 (Main text), 1.0 (Strong / Appendix B)
    # We use 0.02 to match Figure 4a
    data_srt = run_experiment("srt", 0.02, n_runs=N_RUNS, steps=STEPS)

    # 3. (Optional) Tobin Tax
    # Rate = 0.2% = 0.002
    data_tobin = run_experiment("tobin", 0.002, n_runs=N_RUNS, steps=STEPS)

    # --- PLOTTING ---
    print("Generating Plots...")
    # Figure 4a: Losses Distribution (Log Scale)
    # El uso de escala logarítmica permite visualizar las "fat tails" (colas pesadas),
    # fundamentales para analizar eventos de riesgo sistémico extremo.
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data_base["losses"], label="No Tax", fill=True, log_scale=(True, False))
    sns.kdeplot(data_srt["losses"], label="SRT", fill=True, log_scale=(True, False))
    sns.kdeplot(
        data_tobin["losses"],
        label="Tobin (0.2%)",
        fill=True,
        log_scale=(True, False),
        linestyle="--",
    )
    plt.title("Distribución de Pérdidas Financieras (Escala Log-X)")
    plt.xlabel("Tamaño de la Pérdida (Capital Destruido)")
    plt.legend()
    plt.savefig("fig_losses.png")
    plt.close()

    # Figure 4b: Cascade Sizes
    # Comparamos la frecuencia de defaults simultáneos. Un impuesto eficiente debería
    # reducir la probabilidad de cascadas que involucren una gran fracción del sistema (B).
    plt.figure(figsize=(10, 6))
    bins = range(1, Params.B + 2)
    plt.hist(data_base["cascades"], alpha=0.3, label="No Tax", bins=bins, density=True)
    plt.hist(data_srt["cascades"], alpha=0.3, label="SRT", bins=bins, density=True)
    plt.hist(
        data_tobin["cascades"],
        alpha=0.3,
        label="Tobin (0.2%)",
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
    )
    plt.title("Distribución de Tamaños de Cascada (Defaults)")
    plt.xlabel("Número de Bancos en Default")
    plt.ylabel("Densidad de Frecuencia")
    plt.legend()
    plt.savefig("fig_cascades.png")
    plt.close()

    # Figure 4c: Volume (Transaction Volume)
    # Analizamos la eficiencia del mercado. El SRT busca mitigar el riesgo sin
    # contraer excesivamente la liquidez, a diferencia de impuestos transaccionales ciegos.
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data_base["volumes"], label="No Tax", fill=True)
    sns.kdeplot(data_srt["volumes"], label="SRT", fill=True)
    sns.kdeplot(data_tobin["volumes"], label="Tobin (0.2%)", fill=True, linestyle="--")
    plt.title("Volumen de Transacciones Interbancarias")
    plt.xlabel("Volumen Agregado")
    plt.legend()
    plt.savefig("fig_volume.png")
    plt.close()
    # Summary Statistics
    print("\n--- RESULTS SUMMARY ---")
    print(
        f"Avg Loss (No Tax): {np.mean(data_base['losses']) if len(data_base['losses']) > 0 else 0:.2f}"
    )
    print(
        f"Avg Loss (SRT):    {np.mean(data_srt['losses']) if len(data_srt['losses']) > 0 else 0:.2f}"
    )
    print(
        f"Max Loss (No Tax): {np.max(data_base['losses']) if len(data_base['losses']) > 0 else 0:.2f}"
    )
    print(
        f"Max Loss (SRT):    {np.max(data_srt['losses']) if len(data_srt['losses']) > 0 else 0:.2f}"
    )
    print(f"Avg Vol  (No Tax): {np.mean(data_base['volumes']):.2f}")
    print(f"Avg Vol  (SRT):    {np.mean(data_srt['volumes']):.2f}")
    print(f"Avg Vol  (Tobin):  {np.mean(data_tobin['volumes']):.2f}")


if __name__ == "__main__":
    main()
