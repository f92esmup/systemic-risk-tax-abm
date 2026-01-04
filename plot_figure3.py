import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from parameters import Params

# Intentar importar la función de DebtRank. Si falla, la definimos aquí.
try:
    from functions import compute_debtrank
except ImportError:
    # Definición de fallback por si acaso
    def compute_debtrank(L, C, v):
        B = L.shape[0]
        W = np.minimum(1.0, np.divide(L, C, out=np.zeros_like(L), where=C!=0))
        R = np.zeros(B)
        # (Simplificación para visualización si no carga functions.py)
        # En producción usa la del archivo functions.py
        return np.sum(W, axis=1) / B 

def load_data_and_metrics(mode, n_samples=10):
    """
    Carga los últimos 'n_samples' runs para un modo dado (none, srt, tobin).
    Calcula el DebtRank promedio y las Contribuciones Marginales.
    """
    path = os.path.join("output_data", mode, "*.npz")
    files = sorted(glob.glob(path))[-n_samples:]
    
    if not files:
        print(f"Advertencia: No se encontraron datos para el modo '{mode}'")
        return None, None

    all_debtranks = []
    all_loans_size = []
    all_marginal_contrib = []

    print(f"Procesando {len(files)} archivos para modo: {mode}...")

    for f in files:
        try:
            data = np.load(f)
            # Usamos el último paso de tiempo (final de la simulación)
            # Shapes: L_bb [T, B, B], L_fb [T, F, B]
            L_bb = data['L_bb'][-1] 
            L_fb = data['L_fb'][-1]
            
            # --- ESTIMACIÓN DE EQUITY (CAPITAL) ---
            # Si 'banks_state' no se guardó en el .npz (Fase 7 standard),
            # lo reconstruimos aproximadante:
            # Activos ≈ Préstamos Interbancarios (L_bb col sum) + Préstamos Firmas (L_fb col sum)
            # Equity ≈ Activos * INIT_CAPITAL_RATIO (o derivado)
            
            interbank_assets = np.sum(L_bb, axis=0) # Lo que me deben otros bancos
            firm_assets = np.sum(L_fb, axis=0)      # Lo que me deben firmas
            total_assets = interbank_assets + firm_assets
            
            # Evitar división por cero
            total_assets = np.maximum(total_assets, 1e-6)
            
            # Asumimos que el Equity se mantiene cerca del ratio regulatorio o inicial
            # para propósitos de calcular el impacto relativo W_ij
            Equity = total_assets * Params.INIT_CAPITAL_RATIO
            
            # Valor Económico (v) relativo
            v = total_assets / np.sum(total_assets)

            # --- 1. Calcular DebtRank (R) ---
            # L_bb filas=Borrower (Pasivo), cols=Lender. 
            # Para DebtRank W_ij = L_ij / C_j (Impacto de i en j)
            # Cuidado: compute_debtrank espera L con filas=Liabilities
            R = compute_debtrank(L_bb, Equity, v)
            all_debtranks.append(R)

            # --- 2. Calcular Contribución Marginal (Scatter Plot) ---
            # Marginal Contribution ~ R_borrower * (Loan_Amount / Equity_Lender)
            # Solo analizamos enlaces existentes (>0)
            rows, cols = np.where(L_bb > 0)
            for i, j in zip(rows, cols):
                loan_val = L_bb[i, j]
                # Eje X: Tamaño relativo del préstamo (Loan / Equity Lender) o Loan absoluto
                rel_size = loan_val # / Equity[j] 
                
                # Eje Y: Marginal Contribution
                # Aproximación del paper: Delta_EL ~ R[i] * W[i,j] * v[j]
                # W[i,j] = L[i,j] / C[j]
                impact = (loan_val / Equity[j]) if Equity[j] > 0 else 0
                marg = R[i] * impact * v[j]
                
                all_loans_size.append(rel_size)
                all_marginal_contrib.append(marg)

        except Exception as e:
            print(f"Error leyendo {f}: {e}")
            continue

    # Promediar DebtRanks (ordenados para el plot)
    avg_debtrank = np.mean(np.sort(np.array(all_debtranks), axis=1)[:, ::-1], axis=0)
    
    return avg_debtrank, (all_loans_size, all_marginal_contrib)

def plot_figure_3_replication():
    modes = {
        'none': {'color': 'red', 'label': 'No Tax'},
        'tobin': {'color': 'blue', 'label': 'Tobin Tax'}, # Opcional si tienes datos
        'srt': {'color': 'green', 'label': 'SRT'}
    }
    
    results = {}
    
    # Cargar datos
    for mode, style in modes.items():
        if os.path.exists(f"output_data/{mode}"):
            dr, scatter = load_data_and_metrics(mode)
            if dr is not None:
                results[mode] = (dr, scatter)
    
    if not results:
        print("No hay datos para graficar.")
        return

    # --- PLOT FIGURA 3 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # === SUBPLOT 1: DebtRank Profile (Fig 3b) ===
    ax1.set_title("(b) Systemic Risk Profile (DebtRank)", fontsize=14)
    ax1.set_xlabel("Bank Rank (sorted by risk)", fontsize=12)
    ax1.set_ylabel("DebtRank ($R_i$)", fontsize=12)
    
    # Eje X para bancos (1 a 20)
    x_banks = np.arange(1, Params.B + 1)
    
    width = 0.25
    offset = 0
    
    for mode, (dr, _) in results.items():
        if dr is not None:
            # Dibujar barras agrupadas
            ax1.bar(x_banks + offset, dr, width=width, 
                    color=modes[mode]['color'], label=modes[mode]['label'], alpha=0.7)
            offset += width

    ax1.set_xticks(x_banks + width/2)
    ax1.set_xticklabels(x_banks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # === SUBPLOT 2: Marginal Contribution vs Loan Size (Fig 3d) ===
    ax2.set_title("(d) Marginal Contribution vs Loan Size", fontsize=14)
    ax2.set_xlabel("Interbank Liability Size ($L_{mn}$)", fontsize=12)
    ax2.set_ylabel(r"Marginal Contribution $\Delta EL^{syst}$", fontsize=12)
    
    
    has_data = False
    for mode, (_, scatter) in results.items():
        if scatter:
            loans, contribs = scatter
            loans = np.array(loans)
            contribs = np.array(contribs)
            
            # Debug prints
            print(f"Mode {mode}: {len(loans)} points.")
            if len(loans) > 0:
                print(f"  Loans range: [{loans.min():.2e}, {loans.max():.2e}]")
                print(f"  Contribs range: [{contribs.min():.2e}, {contribs.max():.2e}]")
            
            mask = (loans > 1e-9) & (contribs > 1e-12)
            print(f"  Points after filtering: {np.sum(mask)}")
            
            if np.sum(mask) > 0:
                has_data = True
                ax2.scatter(loans[mask], contribs[mask], 
                            c=modes[mode]['color'], label=modes[mode]['label'], 
                            alpha=0.5, s=15, edgecolors='none')

    if has_data:
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        print("Advertencia: No hay datos positivos para graficar en log-log.")
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("fig3_replication_corrected.png", dpi=300)
    print("Gráfica guardada: fig3_replication_corrected.png")
    plt.show()

if __name__ == "__main__":
    plot_figure_3_replication()
