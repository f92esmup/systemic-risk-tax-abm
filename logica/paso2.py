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
        R (vector): Vector DebtRank.
    """
    B = len(equity_banks)
    
    # 1. Matriz de Impacto W_ij (Eq. D1)
    C_inv = np.zeros_like(equity_banks)
    mask_c = equity_banks > 0
    C_inv[mask_c] = 1.0 / equity_banks[mask_c]
    
    # L[j, i] es lo que j debe a i.
    L_trans = L.T 
    
    # W[i, j] = L_trans[i, j] * C_inv[i]
    W = np.minimum(1.0, L_trans * C_inv[:, np.newaxis])
    
    # 2. Valor Económico v_i
    total_assets_per_bank = np.sum(L, axis=0) 
    total_val = np.sum(total_assets_per_bank)
    
    if total_val > 0:
        v = total_assets_per_bank / total_val
    else:
        v = np.ones(B) / B

    # 3. DebtRank Vectorial R
    I = np.eye(B)
    try:
        M = np.linalg.inv(I - W) 
        R = v @ M
    except np.linalg.LinAlgError:
        R = v 
        
    R = np.clip(R, 0, 1)
    
    # 4. Riesgo Escalar H
    H = np.sum(R * v)
    return H, R

def paso2(state, params):
    """
    PASO 2: Mercado de Crédito (Firms-Banks) e Interbancario (Banks-Banks)
    
    1. Empresas piden crédito a N bancos aleatorios y eligen el mejor.
    2. Bancos evalúan liquidez.
    3. Mercado Interbancario con SRT Marginal y muestreo aleatorio.
    """
    
    F = params.F
    B = params.B
    
    # --- A. DESEMPAQUETAR ESTADO ---
    L_demand_F = state['firms_labor_demand'] 
    W_wage = state.get('firms_wage', np.full(F, params.W_BASE))
        
    Liq_F = state['firms_liquidity']
    Liq_B = state['banks_liquidity']
    Equity_B = state['banks_equity']
    
    L_FB = state['net_FB'] 
    L_BB = state['net_BB']
    
    
    # --- B. MERCADO DE CRÉDITO (FIRMS -> BANKS) ---
    
    # 1. Necesidad de Crédito
    payroll = L_demand_F * W_wage
    credit_needed_F = np.maximum(0, payroll - Liq_F)
    
    # 2. Fragilidad Financiera de Firmas
    total_debt_F = np.sum(L_FB, axis=1)
    fragility_F = np.zeros(F)
    mask_pos = (Liq_F + total_debt_F) > 0
    fragility_F[mask_pos] = total_debt_F[mask_pos] / (Liq_F[mask_pos] + total_debt_F[mask_pos] + 1e-9)
    mu_F = np.tanh(fragility_F)
    
    # 3. Muestreo de Bancos (Sin duplicados por firma)
    N_CONTACTS = params.N_BANCOS_CONTACTADOS
    
    # Para asegurar muestreo sin reemplazo por fila de forma vectorizada:
    # Generamos permutaciones aleatorias para cada firma y tomamos los primeros N
    perms = np.array([np.random.permutation(B) for _ in range(F)])
    candidate_banks = perms[:, :N_CONTACTS] # (F, N)
    
    # Especificidad chi
    chi_matrix = np.random.uniform(0, 0.1, (F, N_CONTACTS))
    
    # Calcular Tasas Ofertadas
    rates_offered = params.R_BAR * (1 + chi_matrix * mu_F[:, np.newaxis])
    
    # 4. Selección del Mejor Banco
    best_idx_local = np.argmin(rates_offered, axis=1)
    chosen_banks = candidate_banks[np.arange(F), best_idx_local]
    chosen_rates = rates_offered[np.arange(F), best_idx_local]
    
    # 5. Toma de Crédito con Elasticidad
    mask_high_rate = chosen_rates > params.R_MAX
    credit_taken_F = credit_needed_F.copy()
    credit_taken_F[mask_high_rate] *= params.PHI
    
    # Ejecutar Préstamos
    Liq_F += credit_taken_F
    np.add.at(L_FB, (np.arange(F), chosen_banks), credit_taken_F)
    
    withdrawals_per_bank = np.bincount(chosen_banks, weights=credit_taken_F, minlength=B)
    Liq_B -= withdrawals_per_bank

    
    # --- C. MERCADO INTERBANCARIO (BANKS -> BANKS) ---
    
    deficit_B_mask = Liq_B < 0
    idxs_deficit = np.where(deficit_B_mask)[0]
    idxs_surplus = np.where(Liq_B > 0)[0]
    
    tax_matrix = np.zeros((B, B)) 

    if len(idxs_deficit) > 0 and len(idxs_surplus) > 0:
        
        H_current, _ = calcular_riesgo_sistemico_scalar(L_BB, Equity_B)
        
        # Fragilidad del Borrower (i)
        leverage_B = np.zeros(B)
        total_liab_IB = np.sum(L_BB, axis=1)
        mask_eq = Equity_B > 0
        leverage_B[mask_eq] = total_liab_IB[mask_eq] / Equity_B[mask_eq]
        mu_B = np.tanh(leverage_B)
        
        # Psi (especificidad) del Lender
        psi_B = np.random.uniform(0, params.RANGO_PSI, B)
        
        modo = getattr(params, 'MODO_IMPUESTO', 'NINGUNO')
        
        # Iterar sobre Bancos con Déficit
        for i_global in idxs_deficit:
            amount_needed = abs(Liq_B[i_global])
            if amount_needed < 1e-9: continue
            
            # [AUDIT FIX] Muestreo aleatorio en el interbancario
            # El banco con déficit solo contacta a N bancos con superávit
            num_surplus = len(idxs_surplus)
            n_contacts_ib = min(num_surplus, params.N_BANCOS_CONTACTADOS)
            
            # Seleccionar muestra aleatoria de bancos con superávit
            sampled_j_locals = np.random.choice(num_surplus, n_contacts_ib, replace=False)
            sampled_j_globals = idxs_surplus[sampled_j_locals]
            
            offers = []
            
            for j_global in sampled_j_globals:
                available = Liq_B[j_global]
                if available <= 1e-9: continue
                
                # 1. Tasa Base
                r_offer = params.R_BAR * (1 + psi_B[j_global] * mu_B[i_global])
                
                # 2. Impuesto Marginal
                tax = 0.0
                if modo == 'SRT':
                    amount_test = min(amount_needed, available)
                    L_sim = L_BB.copy()
                    L_sim[i_global, j_global] += amount_test
                    H_new, _ = calcular_riesgo_sistemico_scalar(L_sim, Equity_B)
                    tax = params.ZETA * max(0, H_new - H_current)
                elif modo == 'TOBIN':
                    tax = params.TASA_TOBIN
                
                offers.append({
                    'j_global': j_global,
                    'total_cost': r_offer + tax,
                    'tax': tax
                })
            
            # Ordenar y ejecutar
            offers.sort(key=lambda x: x['total_cost'])
            
            took_total = 0
            for off in offers:
                j_glob = off['j_global']
                available = Liq_B[j_glob]
                if available <= 0: continue
                
                amount = min(amount_needed - took_total, available)
                
                L_BB[i_global, j_glob] += amount
                Liq_B[i_global] += amount
                Liq_B[j_glob] -= amount
                tax_matrix[i_global, j_glob] = off['tax']
                
                if modo == 'SRT' and amount > 0:
                     H_current, _ = calcular_riesgo_sistemico_scalar(L_BB, Equity_B)

                took_total += amount
                if took_total >= amount_needed - 1e-9: break
    
    # --- D. ACTUALIZAR SALARIOS ---
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
        'bank_indices': chosen_banks
    }

def calcular_debtrank_vector(L, equity_banks, v_sys=None):
    _, R = calcular_riesgo_sistemico_scalar(L, equity_banks)
    return R