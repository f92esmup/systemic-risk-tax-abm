import numpy as np
from parametros import Param as p

def calcular_debtrank_vector(L, equity_banks, W_initial=None):
    """
    Calcula el DebtRank de todos los bancos vectorialmente.
    Ref: Battiston et al. (2012) & Poledna et al. (2016) Eq. D1-D5
    
    Args:
        L (matrix): Matriz de Pasivos Interbancarios (B, B). L[i, j] es lo que i debe a j.
        equity_banks (vector): Capital de los bancos (B,).
    
    Returns:
        R (vector): DebtRank de cada banco (impacto sistémico).
    """
    B = len(equity_banks)
    
    # 1. Matriz de Impacto W_ij (Eq. D1 Corregida según Paper)
    # Paper def: W_ij = Impacto de j sobre i (Victima i, Causante j)
    # W_ij = min(1, L_ij / E_i) donde L_ij es loan de j a i (i debe a j)
    
    C_inv = np.zeros_like(equity_banks)
    mask_c = equity_banks > 0
    C_inv[mask_c] = 1.0 / equity_banks[mask_c]
    
    # L[i, j] es i debe a j.
    # Queremos W[i, j] = Impacto de j sobre i.
    # Impacto de j sobre i ocurre porque i tiene activo (prestamo a j)? No.
    # Si j quiebra, i pierde. i prestó a j. i tiene activo "loan to j".
    # En L[borrower, lender], "j debe a i" es L[j, i].
    # Entonces si j quiebra, i pierde L[j, i].
    # Impacto en i = L[j, i] / Equity[i].
    # W[i, j] = L[j, i] / Equity[i].
    
    # Transponemos L para tener L.T[i, j] = L[j, i] (lo que j debe a i)
    L_trans = L.T 
    
    # W[i, j] = L_trans[i, j] * C_inv[i]
    W = np.minimum(1.0, L_trans * C_inv[:, np.newaxis])
    
    # 2. Valor Económico v_i (Eq. D2: Proxy = Total Liabilities Interbank)
    # v_i = sum_k(L_ki) / sum(L) -> Activos de i (lo que prestó)
    # L[k, i] es lo que k debe a i. Sum_k L[k, i] = Total Assets de i.
    total_assets = np.sum(L, axis=0) # Sum over borrowers -> Assets of Lender
    total_val = np.sum(total_assets)
    
    if total_val > 0:
        v = total_assets / total_val
    else:
        v = np.ones(B) / B

    # 3. Cálculo Recursivo
    # Si W[i,j] es impacto de j en i.
    # Distress de i: h_i = Sum_j W[i,j] * h_j + v_i (si v es distress inicial)
    # h = W @ h + v_initial.
    # h = (I - W)^-1 @ v_initial.
    
    # Para DebtRank R_j (Impacto CAUSADO por j):
    # R_j = Sum_i (Distress inducido en i por j) * Value_i
    # Distress inducido en i por j es M_ij (elemento i,j de la inversa).
    # R_j = Sum_i M_ij * v_i
    
    I = np.eye(B)
    try:
        # M = (I - W)^-1
        M = np.linalg.inv(I - W) 
        # R_j = Sum_i (v_i * M_ij) -> R = v @ M
        R = v @ M
    except np.linalg.LinAlgError:
        R = v 
        
    return np.clip(R, 0, 1)

def paso2(state, params):
    """
    PASO 2: Mercado de Crédito (Firms-Banks) e Interbancario (Banks-Banks)
    
    1. Empresas piden crédito para cubrir salarios (Eq. A1).
    2. Bancos evalúan liquidez. Si falta, van al interbancario.
    3. Mercado Interbancario con SRT (Eq. 5, A4).
    """
    
    F = params.F
    B = params.B
    
    # --- A. DESEMPAQUETAR ESTADO ---
    L_demand_F = state['firms_labor_demand'] # (F,)
    
    # Manejo robusto de Salarios
    if 'firms_wage' in state:
        W_wage = state['firms_wage']
    else:
        W_wage = np.full(F, params.W_BASE)
        
    Liq_F = state['firms_liquidity']         # (F,)
    Equity_F = state['firms_equity']         # (F,)
    
    Liq_B = state['banks_liquidity']          # (B,) corrected key
    Equity_B = state['banks_equity']          # (B,) corrected key
    
    # Redes actuales (Stocks de deuda)
    L_FB = state['net_FB'] 
    L_BB = state['net_BB']
    
    # --- B. MERCADO DE CRÉDITO (FIRMS -> BANKS) ---
    
    # 1. Calcular Necesidad de Crédito (Firms)
    payroll = L_demand_F * W_wage
    # Solo piden si no tienen liquidez suficiente
    credit_needed_F = np.maximum(0, payroll - Liq_F) # (F,)
    
    # 2. Asignación Banco-Empresa (Relación Fija o Preferente)
    bank_indices = np.arange(F) % B
    
    # 3. Calcular Tasa de Interés Firmas (Eq. A1)
    # r_ik = r_bar * (1 + chi * mu(fragilidad))
    total_debt_F = np.sum(L_FB, axis=1)
    fragility_F = np.zeros(F)
    mask_pos = (Liq_F + total_debt_F) > 0
    fragility_F[mask_pos] = total_debt_F[mask_pos] / (Liq_F[mask_pos] + total_debt_F[mask_pos] + 1e-9)
    
    chi = np.random.uniform(0, 0.1, F) # Especificidad
    mu = np.tanh(fragility_F)          # Función monótona
    
    r_firms = params.R_BAR * (1 + chi * mu)
    
    # 4. Concesión de Crédito (Tentativa)
    mask_high_rate = r_firms > params.R_MAX
    credit_taken_F = credit_needed_F.copy()
    credit_taken_F[mask_high_rate] *= params.PHI
    
    # Actualizar liquidez empresas (reciben el dinero)
    Liq_F += credit_taken_F
    
    # Registrar nueva deuda en L_FB
    # Usamos np.add.at para sumar vectorialmente a la matriz en las posiciones correctas
    np.add.at(L_FB, (np.arange(F), bank_indices), credit_taken_F)
    
    # Actualizar tasas FB (donde hubo nuevo préstamo)
    # state['rates_FB'] se actualiza fuera o aquí?
    # Mejor devolver las tasas para actualizar en main o actualizar matrix directo si se pasara.
    # Dado que state es mutable y se pasa ref, 'net_FB' ya se actualizó.
    # Necesitamos pasar tasas.
    
    # Impacto en Liquidez Bancaria (Salida de caja)
    withdrawals_per_bank = np.bincount(bank_indices, weights=credit_taken_F, minlength=B)
    Liq_B -= withdrawals_per_bank
    
    
    # --- C. MERCADO INTERBANCARIO (BANKS -> BANKS) ---
    
    # 1. Identificar Déficit y Superávit
    deficit_B_mask = Liq_B < 0
    surplus_B_mask = Liq_B > 0
    
    demand_IB = np.abs(Liq_B[deficit_B_mask])
    supply_IB = Liq_B[surplus_B_mask]
    
    idxs_deficit = np.where(deficit_B_mask)[0]
    idxs_surplus = np.where(surplus_B_mask)[0]
    
    tax_matrix = np.zeros((B, B)) # Para reporting
    delta_el_matrix = np.zeros((B, B)) # Para reporting

    if len(idxs_deficit) > 0 and len(idxs_surplus) > 0:
        
        # 2. Calcular DebtRank ACTUAL (Antes de nuevos préstamos)
        R_current = calcular_debtrank_vector(L_BB, Equity_B)
        
        # 3. Tasa Interbancaria con Impuestos
        # r_ij = r_bar * (1 + psi * mu(leverage_i))
        leverage_B = np.zeros(B)
        mask_eq = Equity_B > 0
        
        total_liab_IB = np.sum(L_BB, axis=1) # Lo que i debe a otros
        leverage_B[mask_eq] = total_liab_IB[mask_eq] / Equity_B[mask_eq]
        
        mu_B = np.tanh(leverage_B)
        
        # Tasa base para los deficitarios
        r_base_IB = params.R_BAR * (1 + params.RANGO_PSI * mu_B[idxs_deficit])
        
        # Añadir Impuesto
        # MODO_IMPUESTO debe estar en params
        modo = getattr(params, 'MODO_IMPUESTO', 'NINGUNO')
        
        # Tax matrix local (Deficit x Surplus)
        tax_matrix_local = np.zeros((len(idxs_deficit), len(idxs_surplus)))
        
        if modo == 'SRT':
            # SRT ~ Zeta * R_i
            srt_rates = params.ZETA * R_current[idxs_deficit]
            tax_matrix_local = srt_rates[:, np.newaxis] + np.zeros((len(idxs_deficit), len(idxs_surplus)))
            
            # Guardar en matriz global para output
            # Broadcast to fill (Deficit rows, Surplus cols)
            # Como todos los surplus ofrecen lo mismo, llenamos
            for idx_d_local, idx_d_global in enumerate(idxs_deficit):
                tax_matrix[idx_d_global, idxs_surplus] = srt_rates[idx_d_local]
                # Delta EL proxy
                delta_el_matrix[idx_d_global, idxs_surplus] = srt_rates[idx_d_local] / params.ZETA

        elif modo == 'TOBIN':
            tax_matrix_local[:] = params.TASA_TOBIN
            tax_matrix[np.ix_(idxs_deficit, idxs_surplus)] = params.TASA_TOBIN
            
        # 4. Matching (Clearing)
        supply_remaining = supply_IB.copy()
        
        # Iterar sobre demandantes (Deficit)
        for i_local, i_global in enumerate(idxs_deficit):
            amount_needed = demand_IB[i_local]
            if amount_needed < 1e-9: continue
            
            took_total = 0
            for j_local, j_global in enumerate(idxs_surplus):
                if supply_remaining[j_local] > 0:
                    amount = min(amount_needed - took_total, supply_remaining[j_local])
                    
                    # Ejecutar Transacción
                    supply_remaining[j_local] -= amount
                    took_total += amount
                    
                    # Actualizar Matriz Interbancaria Global
                    L_BB[i_global, j_global] += amount
                    
                    # Actualizar Liquidez
                    Liq_B[i_global] += amount
                    Liq_B[j_global] -= amount
                    
                    if took_total >= amount_needed - 1e-9:
                        break
    
    # --- D. ACTUALIZAR ESTADO ---
    # Actualizar salarios reales pagados
    # Si hubo préstamo, Liq_F aumentó. Verificamos cuánto se puede pagar.
    payroll_possible = Liq_F
    L_hired_F = np.minimum(L_demand_F, np.floor(payroll_possible / W_wage))
    
    # Pagar salarios (Salida de caja empresas)
    wages_paid = L_hired_F * W_wage
    Liq_F -= wages_paid
    
    return {
        'net_FB': L_FB,
        'net_BB': L_BB,
        'firms_liquidity': Liq_F,
        'banks_liquidity': Liq_B, # corrected key
        'firms_labor_demand': L_hired_F, 
        'wages_paid_vector': wages_paid,
        'tax_matrix': tax_matrix,
        'new_rates_FB': r_firms, # Para actualizar rates_FB
        'bank_indices': bank_indices, # Para saber quién prestó a quién
        'delta_el': delta_el_matrix
    }