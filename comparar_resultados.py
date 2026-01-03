import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def graficar_comparativa():
    # Cargar resultados (asumiendo que ejecutaste todos los escenarios y guardaste en un solo CSV)
    try:
        df = pd.read_csv("resultados_simulacion.csv")
    except FileNotFoundError:
        print("No encuentro 'resultados_simulacion.csv'. Ejecuta la simulación primero.")
        return

    print("Generando gráficas comparativas...")
    # Estilo del paper
    sns.set_theme(style="whitegrid")
    
    # FIGURA 3b y 4c (Aprox): Evolución del Riesgo Sistémico y Volúmenes
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Riesgo Sistémico Total (DebtRank Agregado)
    sns.lineplot(data=df, x='paso', y='riesgo_sistemico', hue='Escenario', ax=axes[0,0], linewidth=2)
    axes[0,0].set_title("Evolución del Riesgo Sistémico (SR)")
    axes[0,0].set_ylabel("Total DebtRank")
    
    # 2. Volumen de Crédito (Deuda Total) - Proxy de Eficiencia
    sns.lineplot(data=df, x='paso', y='deuda_total', hue='Escenario', ax=axes[0,1], linewidth=2)
    axes[0,1].set_title("Volumen Total de Crédito (Eficiencia)")
    axes[0,1].set_ylabel("Crédito Pendiente")

    # 3. Impuesto Recaudado (Acumulado en Fondo)
    # Si implementaste fondo_rescate, úsalo. Si no, impuesto_recaudado (flujo).
    if 'fondo_rescate' in df.columns:
        sns.lineplot(data=df, x='paso', y='fondo_rescate', hue='Escenario', ax=axes[1,0], linewidth=2)
        axes[1,0].set_title("Tamaño del Fondo de Rescate (Acumulado)")
    else:
        sns.lineplot(data=df, x='paso', y='impuesto_recaudado', hue='Escenario', ax=axes[1,0])
        axes[1,0].set_title("Recaudación Diaria")

    # 4. Distribución de Pérdidas (Defaults Empresas) - Proxy de Fig 4a
    # Usamos histograma (KDE) para ver la cola de distribución de quiebras
    # Nota: defaults_empresas es acumulado en el historial, necesitamos el diferencial o usar el acumulado para ver la tendencia total
    # Para la distribución de IMPACTO (tamaño de cascadas), sería mejor ver los defaults por paso.
    # Pero si el CSV tiene acumulados, calculamos la diferencia.
    
    # Agrupamos por escenario para calcular diferencias
    df['defaults_diarios'] = df.groupby('Escenario')['defaults_empresas'].diff().fillna(0)
    
    sns.kdeplot(data=df, x='defaults_diarios', hue='Escenario', ax=axes[1,1], fill=True, common_norm=False)
    axes[1,1].set_title("Distribución de Quiebras Diarias (Cola de Riesgo)")
    axes[1,1].set_xlabel("Número de defaults por paso")

    plt.tight_layout()
    output_file = "comparativa_resultados.png"
    plt.savefig(output_file)
    print(f"Gráficas guardadas en {output_file}")
    # plt.show()

if __name__ == "__main__":
    graficar_comparativa()
