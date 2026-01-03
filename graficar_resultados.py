
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def graficar_resultados_simulacion(nombre_archivo="resultados_simulacion.csv"):
    try:
        df = pd.read_csv(nombre_archivo)
    except FileNotFoundError:
        print(f"No se encontró el archivo {nombre_archivo}")
        return

    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resultados de Simulación: Impuesto de Riesgo Sistémico (ABM)', fontsize=16)
    
    # 1. Riesgo Sistémico
    # Columnas nuevas: 'riesgo_sistemico', 'Escenario', 'paso'
    sns.lineplot(ax=axes[0, 0], data=df, x='paso', y='riesgo_sistemico', hue='Escenario', marker="o")
    axes[0, 0].set_title('Evolución del Riesgo Sistémico (DebtRank Total)')
    axes[0, 0].set_ylabel('Riesgo Sistémico')
    
    # 2. Deuda Total
    sns.lineplot(ax=axes[0, 1], data=df, x='paso', y='deuda_total', hue='Escenario', marker="o")
    axes[0, 1].set_title('Volumen Total de Crédito (Bancos -> Empresas)')
    axes[0, 1].set_ylabel('Deuda Total')
    
    # 3. Impuesto Recaudado
    sns.lineplot(ax=axes[1, 0], data=df, x='paso', y='impuesto_recaudado', hue='Escenario', marker="o")
    axes[1, 0].set_title('Recaudación de Impuestos')
    axes[1, 0].set_ylabel('Impuesto Recaudado por Paso')
    
    # 4. Patrimonio Bancario Total
    sns.lineplot(ax=axes[1, 1], data=df, x='paso', y='patrimonio_bancos_total', hue='Escenario', marker="o")
    axes[1, 1].set_title('Patrimonio Total Bancario')
    axes[1, 1].set_ylabel('Patrimonio Total')
    
    plt.tight_layout()
    plt.savefig('graficos_simulacion.png')
    print("Gráficos guardados en 'graficos_simulacion.png'")
    # plt.show() # No gui in headless

if __name__ == "__main__":
    graficar_resultados_simulacion()
