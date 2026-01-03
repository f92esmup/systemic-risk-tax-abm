import numpy as np
from parameters import *

def step1_firms_planning(state):
    """
    Paso 1: Las firmas definen su demanda de trabajo y capital (crédito).
    
    Lógica Tensorial:
    1. Estimación de Demanda (D_i): Por simplicidad inicial (y falta de historia), 
       asumimos una demanda aleatoria alrededor de la productividad base o constante.
       TODO: Implementar regla adaptativa real cuando haya historia (Paso 7 -> 1).
    2. Cálculo de Producción Planeada (Y_i): Cubrir demanda esperada.
    3. Cálculo de Trabajo Requerido (N_i): Y_i / alpha.
    4. Cálculo de Masa Salarial (W_i): N_i * salario.
    5. Cálculo de Necesidad de Crédito: max(0, W_i - Cash_i).
    """
    
    # 1. Estimación de Demanda
    # Nota: En una simulación completa, esto depende de las ventas del paso anterior.
    # Inicialización: Demanda aleatoria uniforme para ver heterogeneidad.
    # D ~ U(5, 15) unidades de producto
    state.firm_expected_demand = np.random.uniform(5.0, 15.0, size=N_FIRMS)
    
    # 2. Producción Planeada 
    # Las firmas intentan producir lo que esperan vender.
    state.firm_planned_production = state.firm_expected_demand.copy()
    
    # 3. Demanda de Trabajo (Invertir función de producción Cobb-Douglas/Lineal)
    # Y = alpha * N  =>  N = Y / alpha
    state.firm_labor_demand = state.firm_planned_production / LABOR_PRODUCTIVITY
    
    # 4. Costo Salarial (Wage Bill)
    state.firm_wage_bill = state.firm_labor_demand * WAGE_RATE
    
    # 5. Demanda de Crédito
    # Si la firma no tiene suficiente efectivo para pagar salarios, pide prestado.
    # Vectorización: max(0, wage_bill - cash)
    liquidity_gap = state.firm_wage_bill - state.firm_cash
    state.firm_credit_demand = np.maximum(0.0, liquidity_gap)
    
    return state

def step2_banks_lending(state):
    """
    Paso 2: Mercado de Crédito y Asignación de Liquidez.
    
    Lógica Tensorial:
    1. Firmas con credit_demand > 0 seleccionan N_CREDIT_APPS bancos al azar.
    2. Bancos calculan tasa de oferta r_ij basada en fragilidad financiera de la firma A.1.
       r_firm = r_bar * (1 + chi * tanh(leverage)) 
       (Simplificación: chi uniforme, leverage de firma).
    3. Firmas eligen el banco con menor tasa.
    4. Si tasa > r_max, reducen demanda por phi.
    5. Bancos verifican liquidez. (Si falta, interbancario - NO IMPLEMENTADO FULL AÚN, asumimos 'credit crunch' si falla).
       NOTA: En este paso implementamos la ASIGNACIÓN. La gestión de liquidez interbancaria completa se hará
       si hay déficit. Por ahora, si el banco no tiene liquidez, deniega (o se anota el déficit).
    """
    
    # Identificar firmas que necesitan crédito
    needy_firms_indices = np.where(state.firm_credit_demand > 1e-5)[0] # Threshold numérico
    if len(needy_firms_indices) == 0:
        return state
        
    # --- 1. Selección de Bancos Candidatos ---
    # Creamos una máscara (F x B) donde 1 significa "Firma i pide cotización a Banco j"
    # Vectorización: Generamos índices aleatorios para cada firma necesitada.
    quotes_mask = np.zeros((N_FIRMS, N_BANKS), dtype=bool)
    
    # Truco de vectorización para "random choice without replacement" por fila es complejo en NP puro eficiente.
    # Usaremos loop rápido solo sobre firmas necesitadas o argsort de randoms.
    rand_matrix = np.random.rand(len(needy_firms_indices), N_BANKS)
    # Los indices de los top N bancos para cada firma
    top_n_banks = np.argsort(rand_matrix, axis=1)[:, :N_CREDIT_APPS]
    
    # Rellenar máscara
    # Usamos fancy indexing
    row_indices = needy_firms_indices[:, np.newaxis] # Shape (N_needy, 1)
    quotes_mask[row_indices, top_n_banks] = True
    
    # --- 2. Cálculo de Tasas de Interés (Eq A1) ---
    # Necesitamos el leverage de las firmas: L_i / (L_i + Equity_i) ???
    # Paper def: "debt to liquidity ratio l_k(t)" (Page 13, Sec A.1)
    # l_k = Outstanding Debt / Liquid Financial Resources
    # Liquid resources ~ Cash + Equity?. Paper dice "liquid financial resources".
    # Asumiremos Debt / (Existing_Cash + Expected_Income) o algo similar.
    # En t=0, Debt es 0 probablemente.
    # Vamos a usar Debt / Equity como proxy estándar si Cash es volátil.
    # O mejor: Debt / (Cash + Equity).
    
    firm_debt = np.sum(state.bank_firm_loans, axis=0) # Suma por columnas (Firmas) -> (F,)
    # Evitar división por cero
    firm_resources = state.firm_cash + firm_stock_value(state) # Simplificación
    # Si resources es 0, leverage es alto.
    firm_leverage = np.divide(firm_debt, np.maximum(firm_resources, 1.0))
    
    # Chi_i: Especificidad del banco (random uniforme 0-1)
    bank_specificities = np.random.uniform(0, 1, size=N_BANKS)
    
    # Construcción de matriz de tasas (broadcast)
    # r_ij = r_bar * (1 + chi_i * tanh(leverage_j))
    # Shape: (B, F) para facilitar operaciones de banco
    
    # Termino de firma: tanh(leverage) -> (F,)
    firm_risk_factor = np.tanh(firm_leverage)
    
    # Tasa base matrix (B, F)
    # R[i, j] = r_bar * (1 + bank_spec[i] * firm_risk[j])
    # Outer product adaptado
    # bank_spec[i] * firm_risk[j] -> (B, F)
    risk_matrix = np.outer(bank_specificities, firm_risk_factor)
    offered_rates = REFINANCING_RATE * (1.0 + risk_matrix)
    
    # Aplicar máscara: Las tasas de bancos NO consultados deben ser infinitas para no ser elegidas
    # Transponemos quotes_mask (F, B) -> (B, F) para coincidir con offered_rates
    final_rates = np.where(quotes_mask.T, offered_rates, np.inf)
    
    # --- 3. Selección de Mejor Oferta ---
    # Argmin por columna (Bancos) -> Para cada firma (columna), cual es el mejor banco (fila)
    best_bank_indices = np.argmin(final_rates, axis=0) # (F,)
    best_bank_rates = np.min(final_rates, axis=0)      # (F,)
    
    # --- 4. Asignación y Chequeo de Liquidez ---
    # Solo procesamos firmas que tenían demanda
    for f_idx in needy_firms_indices:
        chosen_bank = best_bank_indices[f_idx]
        rate = best_bank_rates[f_idx]
        
        # Check r_max
        demand = state.firm_credit_demand[f_idx]
        if rate > MAX_INTEREST_RATE:
            demand *= CREDIT_DEMAND_CONTRACTION
            
        # Verificar Liquidez del Banco
        # (Simplificación Paso 2: Si tiene cash, presta. Si no, RECHAZA por ahora)
        # TODO: Implementar paso interbancario aquí si queremos fidelidad total
        if state.bank_cash[chosen_bank] >= demand:
            # Aprobar préstamo
            state.bank_cash[chosen_bank] -= demand
            state.firm_cash[f_idx] += demand
            state.bank_firm_loans[chosen_bank, f_idx] += demand
            state.loan_interest_rates[chosen_bank, f_idx] = rate # Guardamos tasa (simplificado: sobreescribe media)
            
            # Registro
            state.new_loans_granted[f_idx] = demand
            state.firm_lender_choice[f_idx] = chosen_bank
        else:
            # Credit Crunch (No hay préstamo)
            # En el modelo full, aquí el banco iría al Interbancario.
            pass

    return state

def firm_stock_value(state):
    # Helper para valorar inventario (precio * cantidad)
    return state.firm_stock * state.firm_prices
