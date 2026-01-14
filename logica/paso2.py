import numpy as np
from parametros import Param as p

def calcular_riesgo_sistemico_scalar(L, equity_banks):
    """
    Calcula el Riesgo Sistémico Total H(L) del sistema bancario.
    H = Sum(R_i * v_i)
    
    Args:
        L (matrix): Matriz de Pasivos Interbancarios (B, B). L[i, j] es lo que i debe a j.
        equity_banks (vector): Capital de los bancos (B,).
    
    Returns:
        H (float): Nivel de riesgo sistémico escalar (0 a 1).
    """
    B = len(equity_banks)
    
    # 1. Matriz de Impacto W_ij (Eq. D1)
    # W_ij = min(1, L_ji / E_i) -> Impacto de j sobre i
    C_inv = np.zeros_like(equity_banks)
    mask_c = equity_banks > 0
    C_inv[mask_c] = 1.0 / equity_banks[mask_c]
    
    # L[j, i] es lo que j debe a i.
    L_trans = L.T 
    
    # W[i, j] = L_trans[i, j] * C_inv[i]
    W = np.minimum(1.0, L_trans * C_inv[:, np.newaxis])
    
    # 2. Valor Económico v_i
    # v_i = Total Assets Interbank_i / Total System Assets
    # Assets de i = Sum_k L[k, i] (lo que otros deben a i)
    total_assets_per_bank = np.sum(L, axis=0) 
    total_val = np.sum(total_assets_per_bank)
    
    if total_val > 0:
        v = total_assets_per_bank / total_val
    else:
        # Si no hay deuda, pesos uniformes o cero riesgo
        v = np.ones(B) / B

    # 3. DebtRank Vectorial R
    I = np.eye(B)
    try:
        # M = (I - W)^-1
        M = np.linalg.inv(I - W) 
        # R = v @ M
        R = v @ M
    except np.linalg.LinAlgError:
        R = v 
        
    R = np.clip(R, 0, 1)
    
    # 4. Riesgo Escalar H
    H = np.sum(R * v)
    return H, R  # Devolvemos también R por si se necesita reporting

def paso2(state, params):
    """
    PASO 2: Mercado de Crédito (Firms-Banks) e Interbancario (Banks-Banks)
    
    1. Empresas piden crédito a N bancos aleatorios y eligen el mejor.
    2. Bancos evalúan liquidez.
    3. Mercado Interbancario con SRT Marginal (Eq. 5).
    """
    
    F = params.F
    B = params.B
    
    # --- A. DESEMPAQUETAR ESTADO ---
    L_demand_F = state['firms_labor_demand'] 
    
    if 'firms_wage' in state:
        W_wage = state['firms_wage']
    else:
        W_wage = np.full(F, params.W_BASE)
        
    Liq_F = state['firms_liquidity']
    Equity_F = state['firms_equity'] # No se usa directamente en tasa, sino deuda/liq
    
    Liq_B = state['banks_liquidity']
    Equity_B = state['banks_equity']
    
    # Redes actuales
    L_FB = state['net_FB'] 
    L_BB = state['net_BB']
    
    
    # --- B. MERCADO DE CRÉDITO (FIRMS -> BANKS) ---
    # [FIX] Selección Aleatoria de Bancos (Paper 1: "Contact random sample of n banks")
    
    # 1. Necesidad de Crédito
    payroll = L_demand_F * W_wage
    credit_needed_F = np.maximum(0, payroll - Liq_F) # (F,)
    
    # 2. Fragilidad Financiera de Firmas (para Tasa)
    # mu_i = tanh(Debt / (Liq + Debt))
    total_debt_F = np.sum(L_FB, axis=1)
    fragility_F = np.zeros(F)
    mask_pos = (Liq_F + total_debt_F) > 0
    fragility_F[mask_pos] = total_debt_F[mask_pos] / (Liq_F[mask_pos] + total_debt_F[mask_pos] + 1e-9)
    mu_F = np.tanh(fragility_F) # (F,)
    
    # 3. Muestreo de Bancos y Ofertas
    N_CONTACTS = params.N_BANCOS_CONTACTADOS
    
    # Generar N candidatos por firma: (F, N)
    candidate_banks = np.random.randint(0, B, (F, N_CONTACTS))
    
    # Generar especificidad chi para cada par (F, N): Uniforme(0, 0.1)
    # chi_ik ~ U[0, 0.1]
    chi_matrix = np.random.uniform(0, 0.1, (F, N_CONTACTS))
    
    # Calcular Tasas Ofertadas
    # r_ik = r_bar * (1 + chi_ik * mu_i)
    # mu_F necesita shape (F, 1) para broadcasting
    rates_offered = params.R_BAR * (1 + chi_matrix * mu_F[:, np.newaxis])
    
    # 4. Selección del Mejor Banco (Menor Tasa)
    best_idx_local = np.argmin(rates_offered, axis=1) # (F,) indices 0..N-1
    
    # Obtener el índice global del banco ganador y su tasa
    # candidate_banks[i, best_idx_local[i]]
    chosen_banks = candidate_banks[np.arange(F), best_idx_local]
    chosen_rates = rates_offered[np.arange(F), best_idx_local]
    
    # 5. Toma de Crédito
    # Aplicar restricción de demanda si tasa es muy alta (Elasticidad)
    mask_high_rate = chosen_rates > params.R_MAX
    credit_taken_F = credit_needed_F.copy()
    credit_taken_F[mask_high_rate] *= params.PHI
    
    # Ejecutar Préstamos
    Liq_F += credit_taken_F
    np.add.at(L_FB, (np.arange(F), chosen_banks), credit_taken_F)
    
    # Impacto en Liquidez Bancaria
    withdrawals_per_bank = np.bincount(chosen_banks, weights=credit_taken_F, minlength=B)
    Liq_B -= withdrawals_per_bank

    
    # --- C. MERCADO INTERBANCARIO (BANKS -> BANKS) ---
    
    # Identificar Déficit y Superávit
    deficit_B_mask = Liq_B < 0
    surplus_B_mask = Liq_B > 0
    
    demand_IB = np.abs(Liq_B[deficit_B_mask])
    supply_IB = Liq_B[surplus_B_mask]
    
    idxs_deficit = np.where(deficit_B_mask)[0]
    idxs_surplus = np.where(surplus_B_mask)[0]
    
    tax_matrix = np.zeros((B, B)) 
    delta_el_matrix = np.zeros((B, B))

    if len(idxs_deficit) > 0 and len(idxs_surplus) > 0:
        
        # Calcular Riesgo Sistémico Inicial H(L)
        H_current, _ = calcular_riesgo_sistemico_scalar(L_BB, Equity_B)
        
        # Pre-calcular factores de tasa base para interbancario
        # r_ij = r_bar * (1 + psi_j * mu_i)
        # mu_i = tanh(Leverage_i) del PRESTATARIO (Deficit)
        leverage_B = np.zeros(B)
        mask_eq = Equity_B > 0
        total_liab_IB = np.sum(L_BB, axis=1)
        leverage_B[mask_eq] = total_liab_IB[mask_eq] / Equity_B[mask_eq]
        mu_B = np.tanh(leverage_B)
        
        # Psi (especificidad) del PRESTAMISTA
        psi_B = np.random.uniform(0, params.RANGO_PSI, B)
        
        modo = getattr(params, 'MODO_IMPUESTO', 'NINGUNO')
        
        supply_remaining = supply_IB.copy()
        
        # Iterar sobre Bancos con Déficit (Borrowers)
        for i_local, i_global in enumerate(idxs_deficit):
            amount_needed = demand_IB[i_local]
            if amount_needed < 1e-9: continue
            
            offers = []
            
            # Evaluar ofertas de Bancos con Superávit (Lenders)
            for j_local, j_global in enumerate(idxs_surplus):
                if supply_remaining[j_local] <= 1e-9: continue
                
                # 1. Tasa de Interés
                r_offer = params.R_BAR * (1 + psi_B[j_global] * mu_B[i_global])
                
                # 2. Impuesto (SRT Marginal o Tobin)
                tax = 0.0
                if modo == 'SRT':
                    # [FIX] Cálculo Marginal de SRT: Zeta * (H(L_new) - H(L_old))
                    # Simular préstamo temporal
                    amount_test = min(amount_needed, supply_remaining[j_local])
                    # Usamos un monto fijo de test o el real? Paper dice "marginal loan".
                    # Usaremos el monto real estimado para exactitud o 1.0 unitario.
                    # Poledna: "Impact of a NEW transaction". Usaremos el monto real.
                    
                    # Copia ligera para simulación (solo cambia un elemento)
                    # Optimización: No copiar toda la matriz si es lenta, pero 20x20 es trivial.
                    L_sim = L_BB.copy()
                    L_sim[i_global, j_global] += amount_test
                    
                    H_new, _ = calcular_riesgo_sistemico_scalar(L_sim, Equity_B)
                    
                    delta_H = max(0, H_new - H_current) # No hay tax negativo
                    tax = params.ZETA * delta_H
                    
                elif modo == 'TOBIN':
                    tax = params.TASA_TOBIN
                
                total_cost = r_offer + tax
                
                offers.append({
                    'j_local': j_local,
                    'j_global': j_global,
                    'rate': r_offer,
                    'tax': tax,
                    'total_cost': total_cost,
                    'available': supply_remaining[j_local]
                })
            
            # Ordenar por menor costo total
            offers.sort(key=lambda x: x['total_cost'])
            
            # Tomar préstamos
            took_total = 0
            for off in offers:
                j_local = off['j_local']
                j_global = off['j_global']
                available = supply_remaining[j_local]
                
                if available <= 0: continue
                
                amount = min(amount_needed - took_total, available)
                
                # Ejecutar
                supply_remaining[j_local] -= amount
                took_total += amount
                
                L_BB[i_global, j_global] += amount
                Liq_B[i_global] += amount
                Liq_B[j_global] -= amount
                
                # Guardar Tax para reporting
                tax_matrix[i_global, j_global] = off['tax']
                
                # Actualizar H_current para la siguiente transacción?
                # Poledna: "Transactions are executed sequentially". Sí, se actualiza.
                if modo == 'SRT' and amount > 0:
                     H_current, _ = calcular_riesgo_sistemico_scalar(L_BB, Equity_B)

                if took_total >= amount_needed - 1e-9:
                    break
    
    # --- D. ACTUALIZAR SALARIOS Y RETORNO ---
    # Pago de salarios con la liquidez final (reducida si no consiguieron crédito)
    # Se limita contratación a lo que se puede pagar
    payroll_possible = Liq_F
    L_hired_F = np.minimum(L_demand_F, np.floor(payroll_possible / W_wage))
    wages_paid = L_hired_F * W_wage
    Liq_F -= wages_paid
    
    return {
        'net_FB': L_FB,
        'net_BB': L_BB,
        'firms_liquidity': Liq_F,
        'banks_liquidity': Liq_B,
        'firms_labor_demand': L_hired_F, 
        'wages_paid_vector': wages_paid,
        'tax_matrix': tax_matrix,
        'new_rates_FB': chosen_rates,
        'bank_indices': chosen_banks, # [FIX] Return chosen banks indices
        'delta_el': delta_el_matrix # Placeholder, no calculado marginalmente para todo par
    }

def calcular_debtrank_vector(L, equity_banks, v_sys=None):
    """
    Wrapper de compatibilidad para calcular el vector DebtRank.
    Usa la lógica optimizada de calcular_riesgo_sistemico_scalar.
    """
    _, R = calcular_riesgo_sistemico_scalar(L, equity_banks)
    return R
