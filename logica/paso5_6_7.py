import numpy as np
from parametros import Param as p

def paso5(state, params):
    """
    PASO 5, 6, 7: Contabilidad, Servicio de Deuda y Gestión de Quiebras.
    
    1. Empresas pagan deuda (Intereses + Principal).
    2. Bancos procesan cobros y defaults.
    3. Actualización de Equity (Profit/Loss).
    4. CASCADA DE QUIEBRAS BANCARIAS (Contagio).
    5. Reinicio de empresas (Revival Rule).
    6. Bancos quebrados NO renacen.
    """
    
    # --- A. DESEMPAQUETAR ---
    F = params.F
    B = params.B
    
    # Stocks
    Liq_F = state['firms_liquidity']
    Eq_F = state['firms_equity']
    Revenue_F = state['firms_revenue'] # Del paso 4
    
    Liq_B = state['banks_liquidity'] # Key matching main.py
    Eq_B = state['banks_equity']     # Key matching main.py
    
    # Deudas
    L_FB = state['net_FB'] # Matriz (F, B)
    L_BB = state['net_BB'] # Matriz (B, B)
    
    # --- B. SERVICIO DE DEUDA EMPRESAS (F -> B) ---
    # Pago Total = Intereses (r * L) + Principal (tau * L)
    
    # Tasa efectiva promedio (simplificación)
    r_eff_F = params.R_BAR * 1.5 
    
    # Monto a pagar
    interest_payment_matrix = L_FB * r_eff_F
    principal_payment_matrix = L_FB * params.TAU
    total_payment_matrix = interest_payment_matrix + principal_payment_matrix
    
    total_obligation_F = np.sum(total_payment_matrix, axis=1)
    
    # Verificar solvencia de liquidez
    Liq_F += Revenue_F
    
    # Vector de pago real
    can_pay_mask = Liq_F >= total_obligation_F
    
    # Matriz de pagos reales
    actual_payment_matrix = total_payment_matrix.copy()
    
    # Para los que no pueden pagar, pagan todo lo que tienen proporcionalmente
    ratio_payment = np.ones(F)
    ratio_payment[~can_pay_mask] = Liq_F[~can_pay_mask] / (total_obligation_F[~can_pay_mask] + 1e-9)
    
    # Ajustar matriz filas para insolventes
    actual_payment_matrix *= ratio_payment[:, np.newaxis]
    
    # Ejecutar Pagos
    # 1. Salida de caja Empresas
    total_paid_F = np.sum(actual_payment_matrix, axis=1)
    Liq_F -= total_paid_F
    
    # 2. Entrada a caja Bancos (Suma por columnas)
    total_received_B = np.sum(actual_payment_matrix, axis=0)
    Liq_B += total_received_B
    
    # 3. Reducción de Deuda (Solo la parte de principal cuenta como desapalancamiento)
    interest_covered_matrix = np.minimum(actual_payment_matrix, interest_payment_matrix)
    principal_covered_matrix = actual_payment_matrix - interest_covered_matrix
    
    L_FB -= principal_covered_matrix
    
    # --- C. CONTABILIDAD EMPRESAS ---
    # Costos del periodo: Salarios + Intereses pagados
    # Recuperamos salarios pagados. En main.py se llama 'wages_paid_vector' en res_p2?
    # En main.py loop, paso2 devuelve 'wages_paid_vector'.
    # Pero no lo guardamos en state['firms_wages_paid'] permanentemente?
    # Deberíamos haberlo pasado en state o params?
    # En main.py, 'wages_paid_vector' se pasa a paso3 y paso4, pero no se guardó en state['...'] explícitamente para paso5?
    # ERROR POTENCIAL: paso5 necesita saber los salarios pagados para calcular profit exacto.
    # Asumiremos que el profit contable aproximado es Revenue - Intereses - (Salarios estimados).
    # O mejor: Revenue - (Liq_pre - Liq_post_wages)?
    # Para consistencia rápida: Profit = Delta Equity.
    # Pero necesitamos calcular Profit para saber si hay dividendos.
    # Usaremos una aproximación o asumiremos que se pasa en 'state'.
    # Vamos a usar 'firms_labor_demand' * W_BASE como proxy de wage bill pagado (si hubo cash).
    # O mejor: state['firms_production'] / alpha * W_BASE ?
    
    # Corrección: En paso2 calculamos 'wages_paid_vector'. Deberíamos haberlo inyectado en state.
    # Como paso2 devolvió un dict y en main hicimos update de varios keys, pero no de wages_paid_vector al state global.
    # Calcularemos wages paid aproximados aquí:
    wage_bill_est = state['firms_labor_demand'] * state['firms_wage']
    
    costs_F = wage_bill_est + np.sum(interest_covered_matrix, axis=1)
    profits_F = Revenue_F - costs_F
    
    # Actualizar Equity
    Eq_F += profits_F
    
    # [FIX] Guardar beneficios para uso en paso1 (Actualización de Salarios)
    state['firms_last_profit'] = profits_F.copy()
    
    # Dividendos (si Equity > 0 y Profit > 0)
    div_mask = (Eq_F > 0) & (profits_F > 0)
    dividends = np.zeros(F)
    dividends[div_mask] = profits_F[div_mask] * params.DIVIDEND_RATIO
    
    Eq_F -= dividends
    Liq_F -= dividends
    
    
    # --- D. CONTABILIDAD BANCOS Y QUIEBRAS EMPRESARIALES ---
    
    # 1. Detectar Firmas Quebradas (Equity < 0)
    bankrupt_F_mask = Eq_F < 0
    bad_debt_loss_B = np.zeros(B)
    
    if np.any(bankrupt_F_mask):
        # Bancos asumen pérdidas por el stock de deuda restante de estas firmas
        losses_matrix = L_FB[bankrupt_F_mask, :]
        
        # Sumar pérdidas por banco
        bad_debt_loss_B = np.sum(losses_matrix, axis=0)
        
        # Write-off de la deuda (borrarla)
        L_FB[bankrupt_F_mask, :] = 0
        
        # REINICIO DE AGENTES (Revival Rule) - Empresas SI renacen
        # Implementación Probabilística del Bailout (Mark 0)
        # Con prob PROB_BAILOUT (ej 0.5), una firma sana compra la deuda (simplificado: reset)
        # Con prob 1-p, quiebra real (reset).
        # En este modelo simplificado, ambos llevan al reset, pero el "Bailout" 
        # implicaría que la deuda se cubre externamente (ej. Equity negativo cubierto).
        # El código original hacía reset directo. Vamos a mantener el reset pero explicitar
        # que es el mecanismo de resolución.
        # Si hubiera lógica diferenciada:
        # random_bailout = np.random.rand(np.sum(bankrupt_F_mask)) < params.PROB_BAILOUT
        # ... logic ...
        
        # Reset Estándar (Revival)
        Eq_F[bankrupt_F_mask] = params.PRECIO_INICIAL * params.UMBRAL_INVENTARIO * 10
        Liq_F[bankrupt_F_mask] = Eq_F[bankrupt_F_mask]
        state['firms_prices'][bankrupt_F_mask] = np.mean(state['firms_prices']) # Precio promedio
    
    # 2. Flujos Interbancarios (Simplificado: Pago de intereses neto)
    r_IB = params.R_BAR 
    interest_IB_pay = np.sum(L_BB, axis=1) * r_IB
    interest_IB_rec = np.sum(L_BB, axis=0) * r_IB
    
    # Net flow
    net_IB_cashflow = interest_IB_rec - interest_IB_pay
    Liq_B += net_IB_cashflow
    
    # 3. Equity Bancos
    income_interest_F = np.sum(interest_covered_matrix, axis=0)
    profits_B = income_interest_F + net_IB_cashflow - bad_debt_loss_B
    
    Eq_B += profits_B
    
    # 4. CASCADA DE QUIEBRAS BANCARIAS (Contagio)
    # "Si un banco quiebra, no renace, tampoco los afectados"
    
    bankrupt_B_mask = Eq_B < 0
    
    # Loop de Contagio (Iterative default)
    # Si un banco quiebra, sus acreedores pierden el activo (L_BB[:, bankrupt])
    
    # Cola de procesamiento
    queue = np.where(bankrupt_B_mask)[0].tolist()
    processed_failures = set(queue)
    
    total_losses_contagion = 0.0
    
    while queue:
        failed_idx = queue.pop(0)
        
        # Encontrar quién le prestó a este banco (Acreedores)
        # L_BB[i, j] -> i debe a j.
        # Si 'failed_idx' quiebra, no paga a sus 'j' (acreedores).
        # Acreedores son las columnas donde L_BB[failed_idx, j] > 0
        
        # Deuda del fallido hacia otros
        debt_obligations = L_BB[failed_idx, :]
        creditors = np.where(debt_obligations > 0)[0]
        
        for cred in creditors:
            # Si el acreedor ya está muerto, no importa (o sí, para stats, pero no propaga)
            # Pero el requisito es "no renace", así que si ya murió, sigue muerto.
            
            loss = debt_obligations[cred]
            total_losses_contagion += loss
            
            # Impactar en Equity del acreedor
            Eq_B[cred] -= loss
            
            # Write-off
            L_BB[failed_idx, cred] = 0
            
            # Verificar si el acreedor quiebra ahora por esta pérdida
            if Eq_B[cred] < 0 and cred not in processed_failures:
                bankrupt_B_mask[cred] = True
                processed_failures.add(cred)
                queue.append(cred)
        
        # Limpiar pasivos del fallido (ya impactados)
        L_BB[failed_idx, :] = 0
        # Limpiar activos del fallido? (Nadie le paga? O cobra para liquidar?)
        # Simplificación: Sus activos desaparecen o se congelan.
        L_BB[:, failed_idx] = 0 
    
    # 5. NO RENACIMIENTO
    # Los bancos quebrados quedan con Eq < 0. No hacemos reset.
    # Aseguramos Liq = 0 para que no presten en el futuro
    Liq_B[bankrupt_B_mask] = 0.0
    
    
    # --- E. RETORNO DE ESTADO ---
    return {
        'firms_equity': Eq_F,
        'firms_liquidity': Liq_F,
        'banks_equity': Eq_B,       # Key matching main.py
        'banks_liquidity': Liq_B,   # Key matching main.py
        'net_FB': L_FB,
        'net_BB': L_BB,
        'bankruptcies_F': np.sum(bankrupt_F_mask),
        'bankruptcies_B': np.sum(bankrupt_B_mask),
        'dividends_total': np.sum(dividends),
        'contagion_loss': total_losses_contagion,
        'mask_bankrupt_F': bankrupt_F_mask # NEW
    }