import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

from simulacion import Modelo_CRISIS
from parametros import Parametros
import visualizacion
import analisis

# --- CONFIGURATION ---
DEFAULT_STEPS_DEMO = 60
DEFAULT_STEPS_PAPER = 200
DEFAULT_RUNS_PAPER = 50


def run_simulation_single(mode, param, steps, run_id, output_folder, save_disk=True):
    """
    Runs a single simulation.
    Returns: (model instance, metrics_dict)
    """
    model = Modelo_CRISIS(seed=2000 + run_id, tax_mode=mode, tax_param=param)

    losses = []
    defaults = []
    volumes = []

    for t in range(steps):
        model.ejecutar_paso()

        # Metrics for Figure 4
        if model.current_step_loss > 0:
            losses.append(model.current_step_loss)
        if model.current_step_defaults > 0:
            defaults.append(model.current_step_defaults)
        volumes.append(model.current_step_volume)

    if save_disk:
        os.makedirs(output_folder, exist_ok=True)
        model.guardar_simulacion_disco(run_id=run_id, folder=output_folder)

    metrics = {
        "losses": np.array(losses),
        "cascades": np.array(defaults),
        "mean_volume": np.mean(volumes) if volumes else 0.0,
        "volumes_ts": np.array(volumes),
    }

    return model, metrics


def run_mode_single_demo():
    """
    Mode: SINGLE_DEMO
    Runs 1 simulation and generates network graphs.
    """
    print("\n>>> MODE: SINGLE_DEMO <<<\n")
    mode = "srt"  # Use SRT to show the full mechanism
    param = Parametros.TAX_SRT_ZETA
    folder = "output_data/demo"

    print(f"Running 1 simulation (Mode: {mode}, Steps: {DEFAULT_STEPS_DEMO})...")
    model, _ = run_simulation_single(
        mode, param, DEFAULT_STEPS_DEMO, run_id=0, output_folder=folder
    )

    print("Generating Network Graphs...")
    steps_to_plot = [20, 40, 50]
    for s in steps_to_plot:
        if s < DEFAULT_STEPS_DEMO:
            visualizacion.generar_grafo_multicapa(run_id=0, step=s, folder=folder)

    print(f"\nDemo complete. Data saved to {folder}/\n")
    print(f"Graphs saved in {os.getcwd()}")


def run_mode_paper_replication():
    """
    Mode: PAPER_REPLICATION
    Runs batch simulations and generates Figure 3 & 4.
    """
    print("\n>>> MODE: PAPER_REPLICATION <<<\n")
    print(
        f"Configuration: {DEFAULT_RUNS_PAPER} runs per scenario, {DEFAULT_STEPS_PAPER} steps."
    )

    scenarios = [
        ("none", 0.0),
        ("tobin", Parametros.TAX_TOBIN_RATE),
        ("srt", Parametros.TAX_SRT_ZETA),
    ]

    results_for_fig4 = {}  # Store aggregated data

    for mode, param in scenarios:
        print(f"\n--- Scenario: {mode.upper()} ---")
        output_folder = f"output_data/{mode}"

        # Aggregators
        all_losses = []
        all_cascades = []
        all_avg_volumes = []

        for r in tqdm(range(DEFAULT_RUNS_PAPER), desc=f"Simulating {mode}"):
            _, metrics = run_simulation_single(
                mode, param, DEFAULT_STEPS_PAPER, run_id=r, output_folder=output_folder
            )

            if len(metrics["losses"]) > 0:
                all_losses.extend(metrics["losses"])
            if len(metrics["cascades"]) > 0:
                all_cascades.extend(metrics["cascades"])
            all_avg_volumes.append(metrics["mean_volume"])

        results_for_fig4[mode] = {
            "losses": np.array(all_losses),
            "cascades": np.array(all_cascades),
            "volumes": np.array(all_avg_volumes),
        }

    print("\n>>> Simulations Complete. Starting Analysis... <<<")

    # Generate Figure 3 (Reads from disk)
    analisis.generar_figura_3()

    # Generate Figure 4 (Uses memory data)
    analisis.generar_figura_4(results_for_fig4)

    print(
        "\nReplication complete. Figures 3 and 4 saved in output_data/graficas_finales."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Antigravity Systemic Risk Model - Main Execution Script"
    )
    parser.add_argument(
        "--mode",
        choices=["SINGLE_DEMO", "PAPER_REPLICATION"],
        default="SINGLE_DEMO",
        help="Execution mode: SINGLE_DEMO (fast, visualization) or PAPER_REPLICATION (batch, stats)",
    )

    args = parser.parse_args()

    if args.mode == "SINGLE_DEMO":
        run_mode_single_demo()
    elif args.mode == "PAPER_REPLICATION":
        run_mode_paper_replication()


if __name__ == "__main__":
    main()


