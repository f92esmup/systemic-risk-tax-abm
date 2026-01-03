
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_simulation_results(filename="simulation_results.csv"):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"No se encontró el archivo {filename}")
        return

    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resultados de Simulación: Systemic Risk Tax ABM', fontsize=16)
    
    # 1. Systemic Risk
    sns.lineplot(ax=axes[0, 0], data=df, x='step', y='systemic_risk', hue='Scenario', marker="o")
    axes[0, 0].set_title('Evolución del Riesgo Sistémico (DebtRank Total)')
    axes[0, 0].set_ylabel('Systemic Risk')
    
    # 2. Total Debt (Credit Market Activity)
    sns.lineplot(ax=axes[0, 1], data=df, x='step', y='total_debt', hue='Scenario', marker="o")
    axes[0, 1].set_title('Volumen Total de Crédito (Bancos -> Firmas)')
    axes[0, 1].set_ylabel('Total Debt Amount')
    
    # 3. Collected Tax
    sns.lineplot(ax=axes[1, 0], data=df, x='step', y='collected_tax', hue='Scenario', marker="o")
    axes[1, 0].set_title('Recaudación de Impuestos')
    axes[1, 0].set_ylabel('Tax Collected per Step')
    
    # 4. Bank Equity (Health)
    sns.lineplot(ax=axes[1, 1], data=df, x='step', y='total_bank_equity', hue='Scenario', marker="o")
    axes[1, 1].set_title('Patrimonio Total Bancario')
    axes[1, 1].set_ylabel('Total Equity')
    
    plt.tight_layout()
    plt.savefig('simulation_plots.png')
    print("Gráficos guardados en 'simulation_plots.png'")
    # plt.show() # No gui in headless

if __name__ == "__main__":
    plot_simulation_results()
