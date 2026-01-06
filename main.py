import os
import numpy as np
from tqdm import tqdm

from simulacion import Modelo_CRISIS
from parametros import Parametros
import analisis
import grafos

# --- CONFIGURATION ---
DEFAULT_STEPS_PAPER = 50

DEFAULT_RUNS_PAPER = 100


def run_simulation_single(mode, param, steps, run_id, output_folder, save_disk=True):
    """
    Runs a single simulation.
    Returns: (model instance, metrics_dict)
    """
    model = Modelo_CRISIS(seed=2000 + run_id, tax_mode=mode, tax_param=param)

    losses = []
    defaults = []
    volumes = []

    for _ in range(steps):
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


def run_mode_paper_replication():
    """
    Runs batch simulations and generates Figure 3 & 4.
    """
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

    # Generate GIF for Run 0 (Scenario: None)
    print("\n>>> Generating Network Dynamics GIF (Run 0, Scenario: None) <<<")
    grafos.crear_gif_redes("none", n_run=0, steps_limit=DEFAULT_STEPS_PAPER)

    # Generate GIF for Run 0 (Scenario: Tobin)
    print("\n>>> Generating Network Dynamics GIF (Run 0, Scenario: Tobin) <<<")
    grafos.crear_gif_redes("tobin", n_run=0, steps_limit=DEFAULT_STEPS_PAPER)

    # Generate GIF for Run 0 (Scenario: SRT)
    print("\n>>> Generating Network Dynamics GIF (Run 0, Scenario: SRT) <<<")
    grafos.crear_gif_redes("srt", n_run=0, steps_limit=DEFAULT_STEPS_PAPER)


if __name__ == "__main__":
    run_mode_paper_replication()
