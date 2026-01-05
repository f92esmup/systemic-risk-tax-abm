import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from simulacion import Modelo_CRISIS
from parametros import Parametros
import os

# Configuración de Estilo
sns.set_theme(style="whitegrid")


def run_batch(mode, param, n_runs=10, n_steps=200):
    """Ejecuta un lote de simulaciones y guarda resultados."""
    print(f"\n>>> Iniciando Experimento: Modo={mode.upper()}, Parámetro={param}")

    all_losses = []
    all_cascades = []
    all_volumes = []

    output_folder = f"output_data/{mode}"
    os.makedirs(output_folder, exist_ok=True)

    for r in tqdm(range(n_runs), desc=f"Simulando {mode}"):
        model = Modelo_CRISIS(seed=2000 + r, tax_mode=mode, tax_param=param)

        run_volumes = []
        for t in range(n_steps):
            model.ejecutar_paso()

            # Recolectar métricas del paso
            if model.current_step_loss > 0:
                all_losses.append(model.current_step_loss)
            if model.current_step_defaults > 0:
                all_cascades.append(model.current_step_defaults)

            run_volumes.append(model.current_step_volume)

        all_volumes.append(np.mean(run_volumes))

        # Guardar snapshot completo a disco
        model.guardar_simulacion_disco(run_id=r, folder=output_folder)

    return {
        "losses": np.array(all_losses),
        "cascades": np.array(all_cascades),
        "volumes": np.array(all_volumes),
    }


def main():
    # Parámetros de ejecución
    # Ajustar según necesidad de velocidad vs precisión
    N_RUNS = 2  # Reducido para testing rápido
    STEPS = 50

    # 1. Ejecutar Escenarios
    data_none = run_batch("none", 0.0, N_RUNS, STEPS)
    data_tobin = run_batch("tobin", 0.002, N_RUNS, STEPS)
    data_srt = run_batch("srt", 0.02, N_RUNS, STEPS)

    # 2. Generar Figura 4 (Distribuciones)
    print("\nGenerando Figura 4: Comparativa de Riesgo Sistémico...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A. Pérdidas (Log Scale)
    if len(data_none["losses"]) > 0:
        sns.kdeplot(
            data_none["losses"],
            ax=axes[0],
            label="No Tax",
            fill=True,
            log_scale=(True, False),
            color="red",
        )
    if len(data_srt["losses"]) > 0:
        sns.kdeplot(
            data_srt["losses"],
            ax=axes[0],
            label="SRT",
            fill=True,
            log_scale=(True, False),
            color="green",
        )
    axes[0].set_title("Distribución de Pérdidas (L)")
    axes[0].legend()

    # B. Cascadas
    bins = np.arange(1, Parametros.B + 2)
    axes[1].hist(
        data_none["cascades"],
        bins=bins,
        alpha=0.5,
        label="No Tax",
        color="red",
        density=True,
    )
    axes[1].hist(
        data_srt["cascades"],
        bins=bins,
        alpha=0.5,
        label="SRT",
        color="green",
        density=True,
    )
    axes[1].set_title("Tamaño de Cascadas (C)")
    axes[1].set_xlabel("Nº de Bancos en Default")
    axes[1].legend()

    # C. Volumen
    sns.boxplot(
        data=[data_none["volumes"], data_tobin["volumes"], data_srt["volumes"]],
        ax=axes[2],
    )
    axes[2].set_xticklabels(["No Tax", "Tobin", "SRT"])
    axes[2].set_title("Volumen del Mercado Interbancario (V)")

    plt.tight_layout()
    plt.savefig("figura4_resultados.png", dpi=300)
    print("Métricas guardadas en 'figura4_resultados.png'")


if __name__ == "__main__":
    main()
