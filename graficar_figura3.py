import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import funciones as fn
from parametros import Parametros

# Configuración de gráficas
plt.style.use('default')
sns.set_theme(style="whitegrid")

def calculate_systemic_risk_metrics(L, equity, assets):
    """
    Calcula el riesgo sistémico total (EL) y las contribuciones marginales de cada préstamo.
    """
    B = L.shape[0]
    
    # 1. Reconstruir métricas financieras necesarias
    # Pasivos Interbancarios (Borrowing) = Suma de columnas de L (porque filas=Borrower en L? NO, convención de simulación era: L[d, s] += amount. d=Borrower (Fila), s=Lender (Columna).)
    # Verificamos simulacion.py: "self.matriz_interbancaria[d, s] += amount". 
    # d es deficit (necesita dinero, Borrower). s es surplus (Lender).
    # Entonces L[row, col]: Row debe a Col.
    
    interbank_liabilities = np.sum(L, axis=1) # Lo que row debe
    
    # Deposits = Assets - Equity - Interbank_Assets? 
    # En simulacion.py: Equity = Assets * Ratio. Liq = Assets. Deposits = Assets - Eq.
    # Pero durante la simulación, Liq cambia.
    # Usemos Equity y Assets del histórico para estimar Leverage aproximado.
    # Leverage = (Assets - Equity) / Equity ? O Total Liabilities / Equity.
    # Paper Apéndice A: Leverage l_i = Total Debt / Equity.
    # Total Debt = Deposits + Interbank Liabilities.
    # Deposits no se guardan explícitamente en estado_bancos? Sí, IDX_BANK_DEPOSITS = 2.
    # Pero estado_bancos[:, 2] no se actualizaba dinámicamente en accounting, solo en init?
    # Revisando accounting: "estado_bancos[:, IDX_BANK_LIQUIDITY] -= dividends".
    # Deposits no cambiaban mucho salvo por pagos de dividendos a dueños (que aumentan depositos de dueños, pero es pasivo del banco).
    # Vamos a estimar Liabilities = Assets - Equity. Es lo más robusto contablemente.
    
    total_liabilities = np.maximum(0, assets - equity)
    leverage = np.divide(total_liabilities, equity, out=np.zeros_like(equity), where=equity>0.01)
    
    # 2. Probabilidad de Default (p_i)
    # Eq: p = 0.01 * tanh(10 * leverage)
    p_default = 0.01 * np.tanh(Parametros.K_mu * leverage)
    
    # 3. Valor Económico (v)
    v = assets
    V_total = np.sum(v)
    
    # 4. Baseline Expected Loss (EL_syst)
    # EL = Sum(p_i * R_i * V_total) ??
    # Eq 1: EL_syst = V(t) * Sum(p_i * R_i)
    # R_i es DebtRank del nodo i.
    
    R_base = fn.calcular_debtrank(L, equity, v)
    EL_base = np.sum(p_default * R_base) * V_total
    
    # 5. Marginal Contributions
    # Iterar sobre cada enlace > 0
    marginals = [] # (Loan_Size, Marginal_Contribution)
    
    rows, cols = np.where(L > 1e-5)
    for i, j in zip(rows, cols):
        loan_val = L[i, j]
        
        # Escenario Contrafactual: Remover el préstamo
        # Delta = EL_base - EL(sin préstamo)
        # Esto nos dice cuánto "aporta" ese préstamo al riesgo actual.
        
        L_temp = L.copy()
        L_temp[i, j] = 0.0
        
        R_temp = fn.calcular_debtrank(L_temp, equity, v)
        EL_temp = np.sum(p_default * R_temp) * V_total
        
        delta = EL_base - EL_temp
        
        # Guardar (Loan Size relativo a Equity del Lender?, o Absoluto?)
        # Fig 3c usa "relative loan size [%]". Generalmente Loan / Lender_Equity o Loan / Total_Assets.
        # El paper dice "relative size of interbank loans". Asumiremos Loan / Equity del Lender (j).
        # Para evitar outliers, usaremos escala logarítmica en plot, así que guardamos valor absoluto o ratio.
        # Usemos Loan Absolute para scatter raw, luego normalizamos.
        
        lender_equity = equity[j]
        rel_size = loan_val / lender_equity if lender_equity > 0 else loan_val
        
        marginals.append((rel_size, delta))
        
    return R_base, marginals

def plot_figure3_corrected():
    modes = ["none", "tobin", "srt"]
    colors = {"none": "red", "tobin": "blue", "srt": "green"}
    labels = {"none": "No Tax", "tobin": "Tobin Tax", "srt": "SRT"}
    
    data_store = {}
    
    print("Iniciando análisis forense para Figura 3...")
    
    for mode in modes:
        # Buscar el último run
        files = sorted(glob.glob(f"output_data/{mode}/*.npz"))
        if not files:
            print(f"No hay datos para {mode}")
            continue
            
        f = files[0] # Usar el primer run disponible
        print(f"Procesando {f}...")
        
        try:
            d = np.load(f)
            # Último paso
            L = d["matriz_interbancaria"][-1]
            banks = d["estado_bancos"][-1]
            
            # banks indices: 1=Equity, 7=Assets (según simulacion.py nuevo)
            equity = banks[:, 1]
            assets = banks[:, 7]
            
            R, margs = calculate_systemic_risk_metrics(L, equity, assets)
            data_store[mode] = {"R": R, "marginals": margs}
            
        except Exception as e:
            print(f"Error procesando {mode}: {e}")
            
    # --- PLOT FIGURA 3 A&B (Perfiles DebtRank) ---
    plt.figure(figsize=(10, 6))
    
    # Para el eje X, ordenamos por el ranking del "No Tax"
    if "none" in data_store:
        R_none = data_store["none"]["R"]
        rank_indices = np.argsort(R_none)[::-1] # Índices de mayor a menor riesgo
    else:
        rank_indices = np.arange(Parametros.B)
        
    x_axis = np.arange(1, Parametros.B + 1)
    
    width = 0.25
    offset = 0
    
    for mode in modes:
        if mode in data_store:
            R = data_store[mode]["R"]
            # Ordenar R según el orden de "No Tax" para comparar banco a banco
            R_sorted = R[rank_indices]
            
            plt.bar(x_axis + offset, R_sorted, width=width, label=labels[mode], color=colors[mode], alpha=0.7)
            offset += width
            
    plt.xlabel("Bank Rank (Sorted by No-Tax Risk)")
    plt.ylabel("DebtRank (R)")
    plt.title("Perfiles de Riesgo Sistémico (DebtRank)")
    plt.xticks(x_axis + width, x_axis)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("output_data/graficas_finales", exist_ok=True)
    plt.savefig("output_data/graficas_finales/figura3_ab_debtrank.png", dpi=300)
    plt.close()
    
    # --- PLOT FIGURA 3 C&D (Scatter Marginal) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    
    all_loans = []
    all_deltas = []
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        if mode in data_store:
            margs = data_store[mode]["marginals"]
            if not margs:
                ax.text(0.5, 0.5, "No Loans", ha='center')
                continue
                
            loans, deltas = zip(*margs)
            loans = np.array(loans)
            deltas = np.array(deltas)
            
            # Filtrar ceros para log-log
            mask = (loans > 0) & (deltas > 0)
            
            ax.scatter(loans[mask], deltas[mask], color=colors[mode], alpha=0.5, edgecolors='none')
            ax.set_title(labels[mode])
            ax.set_xlabel("Relative Loan Size")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
            if i == 0:
                ax.set_ylabel("Marginal Contribution to Systemic Risk")
                
    plt.suptitle("Contribución Marginal al Riesgo vs Tamaño del Préstamo (Log-Log)")
    plt.tight_layout()
    plt.savefig("output_data/graficas_finales/figura3_cd_marginal.png", dpi=300)
    plt.close()
    print("Gráficas generadas en output_data/graficas_finales/")

if __name__ == "__main__":
    plot_figure3_corrected()
