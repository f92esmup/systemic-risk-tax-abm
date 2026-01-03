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

def step3_production(state):
    """
    Paso 3: Mercado Laboral y Producción.
    
    Lógica Tensorial:
    1. Determinar presupuesto laboral máximo por firma (limited by Cash).
    2. Determinar demanda laboral efectiva N_desired = min(Planned, Budget/Wage).
    3. Mercado Laboral (Matching):
       - Oferta Total = N_HOUSEHOLDS (Asumimos todos quieren trabajar por ahora).
       - Demanda Total = sum(N_desired).
       - Si Demanda > Oferta -> Racionamiento.
       - Asignamos trabajadores a firmas (actualizamos household_employer).
    4. Pago de Salarios:
       - Firmas pagan w * Hired.
       - Hogares reciben w.
    5. Producción:
       - Y_real = alpha * Hired.
       - Stock += Y_real.
    """
    
    # 1. Restricción Presupuestaria
    # Cuántos trabajadores puede pagar la firma con su caja actual?
    # affordable_workers = floor(cash / wage)
    max_affordable_labor = np.floor(state.firm_cash / WAGE_RATE)
    
    # 2. Demanda Efectiva (limitada por plan y por dinero)
    # state.firm_labor_demand viene del Paso 1
    desired_labor = np.minimum(state.firm_labor_demand, max_affordable_labor)
    
    total_labor_demand = np.sum(desired_labor)
    total_labor_supply = N_HOUSEHOLDS
    
    # 3. Matching / Racionamiento
    # Vector de trabajadores contratados por firma
    firm_hired_workers = np.zeros(N_FIRMS, dtype=np.int32)
    
    if total_labor_demand <= total_labor_supply:
        # Hay suficientes trabajadores
        firm_hired_workers = np.floor(desired_labor).astype(np.int32)
        # Asignación de empleadores a hogares (Simplificado: Llenado buckets)
        # En una simulación real tensorial, mantener el link exacto es costoso si barajamos cada turno.
        # Aquí regeneramos el mapa de empleo por simplicidad del paso.
        pass # Se hará en el bloque de asignación abajo
    else:
        # Escasez de trabajadores (Caso Típico con params actuales: 10k demand vs 1.3k supply)
        # Racionamiento proporcional o aleatorio?
        # Proporcional: Cada firma recibe (Supply/Demand) * Desired
        rationing_ratio = total_labor_supply / total_labor_demand
        firm_hired_workers = np.floor(desired_labor * rationing_ratio).astype(np.int32)
        
        # Ajuste de redondeo: Si sobran trabajadores por el floor, asignarlos al azar
        current_hired = np.sum(firm_hired_workers)
        remainder = total_labor_supply - current_hired
        if remainder > 0:
            lucky_firms = np.random.choice(N_FIRMS, int(remainder), replace=True) # O replace=False
            np.add.at(firm_hired_workers, lucky_firms, 1)
            
    # Actualizar estado de empleadores de hogares
    # Reseteamos empleos actuales
    state.household_employer[:] = -1
    
    # Asignación vectorial 'flat'
    # Creamos un vector de IDs de firmas repetidos tantas veces como trabajadores contrataron
    # Ej: Firma 0 contrata 2 -> [0, 0, Firma 1 contrata 1 -> 1...]
    employer_ids = np.repeat(np.arange(N_FIRMS), firm_hired_workers)
    
    # Limitamos a N_HOUSEHOLDS por seguridad (si hubo error de redondeo arriba)
    assigned_count = min(len(employer_ids), N_HOUSEHOLDS)
    employer_ids = employer_ids[:assigned_count]
    
    # Asignamos a los primeros H hogares (podríamos barajar households para realismo)
    # random_households = np.random.permutation(N_HOUSEHOLDS)
    # state.household_employer[random_households[:assigned_count]] = employer_ids
    # Por ahora directo 0..N
    state.household_employer[:assigned_count] = employer_ids
    
    # 4. Pago de Salarios y Flujos
    wages_paid = firm_hired_workers * WAGE_RATE
    
    state.firm_cash -= wages_paid
    
    # Hogares reciben salario
    # Usamos np.add.at para sumar salarios a los hogares empleados (aunque w es cte, es mas general)
    # Aquí: household_cash += WAGE_RATE donde employer != -1
    employed_mask = state.household_employer != -1
    state.household_cash[employed_mask] += WAGE_RATE
    
    # 5. Producción Real y Stock
    actual_production = firm_hired_workers * LABOR_PRODUCTIVITY
    state.firm_stock += actual_production
    
    # Guardamos producción real para métricas si hiciera falta
    # state.firm_planned_production se sobreescribe o mantenemos diff?
    state.firm_stock = np.maximum(state.firm_stock, 0.0) # Corrección error flotante
    
    return state

def step4_consumption(state):
    """
    Paso 4: Consumo de los Hogares.
    
    Lógica Tensorial:
    1. Hogares definen presupuesto C_h = c * Cash_h.
    2. Seleccionan Z firmas al azar y eligen la de menor precio (P_i).
    3. Agregan demanda a esas firmas.
    4. Firmas verifican stock. Si Demanda > Stock -> Racionamiento.
    5. Ejecución de compras.
    """
    
    # 1. Presupuesto
    household_budgets = state.household_cash * PROPENSITY_TO_CONSUME
    
    # 2. Selección de Proveedores (Z firmas al azar)
    # Matriz (H, Z) de índices de firmas
    # N_CONSUMPTION_APPS = z
    firm_choices = np.random.randint(0, N_FIRMS, size=(N_HOUSEHOLDS, N_CONSUMPTION_APPS))
    
    # Obtener precios de las firmas elegidas -> (H, Z)
    # state.firm_prices shape (F,)
    chosen_prices = state.firm_prices[firm_choices]
    
    # Encontrar índice del mínimo precio por fila (axis=1)
    best_idx_local = np.argmin(chosen_prices, axis=1) # (H,) 0..Z-1
    
    # Obtener el ID de la firma ganadora para cada hogar
    # Fancy indexing: range(H) para filas, best_idx_local para columnas
    winner_firms = firm_choices[np.arange(N_HOUSEHOLDS), best_idx_local] # (H,)
    winner_prices = chosen_prices[np.arange(N_HOUSEHOLDS), best_idx_local] # (H,)
    
    # 3. Calcular Demanda Deseada (Q = Budget / Price)
    # Evitar div por cero si price=0 (aunque init es 1.0)
    desired_quantities = np.zeros_like(household_budgets)
    valid_prices = winner_prices > 1e-9
    desired_quantities[valid_prices] = household_budgets[valid_prices] / winner_prices[valid_prices]
    
    # 4. Agregación de Demanda por Firma
    # Sumar desired_quantities agrupado por winner_firms
    total_demand_per_firm = np.bincount(winner_firms, weights=desired_quantities, minlength=N_FIRMS)
    
    # 5. Racionamiento (Supply Constraint)
    available_stock = state.firm_stock
    # Calcular factor de racionamiento (1.0 si sobra stock, <1.0 si falta)
    # ration = stock / demand
    rationing_factor = np.ones(N_FIRMS, dtype=np.float64)
    shortage_mask = total_demand_per_firm > available_stock
    
    # Para evitar div/0 si demand es 0 (mask lo cubre, pero por seguridad numérica)
    denominator = np.maximum(total_demand_per_firm, 1e-9)
    rationing_factor[shortage_mask] = available_stock[shortage_mask] / denominator[shortage_mask]
    
    # 6. Ejecución (Broadcasting del racionamiento a los hogares)
    # Cada hogar 'h' compró a 'winner_firms[h]', le aplica 'rationing_factor[winner_firms[h]]'
    household_rationing = rationing_factor[winner_firms]
    
    actual_quantities = desired_quantities * household_rationing
    actual_spending = actual_quantities * winner_prices
    
    # Actualizar saldos
    state.household_cash -= actual_spending
    
    # Ingresos de Firmas: Sumar spending agrupado por winner_firms
    firm_revenue = np.bincount(winner_firms, weights=actual_spending, minlength=N_FIRMS)
    state.firm_cash += firm_revenue
    
    # Decremento de Stock
    firm_sales_vol = np.bincount(winner_firms, weights=actual_quantities, minlength=N_FIRMS)
    state.firm_stock -= firm_sales_vol
    state.firm_stock = np.maximum(state.firm_stock, 0.0) # Corrección error flotante
    
    return state


def step5_firm_repayment(state):
    """
    Paso 5: Servicio de Deuda, Repago y Quiebras.
    
    Lógica Tensorial:
    1. Calcular obligaciones de pago por préstamo (Interés + Amortización).
    2. Comparar con caja de la firma.
    3. Solventes: Pagan todo, reducen deuda.
    4. Insolventes (Default):
       - Pagan lo que tienen (proporcionalmente a los acreedores).
       - El resto es pérdida para el banco (Bad Debt).
       - La firma quiebra y es reemplazada (reset).
    5. Actualizar Equity de Bancos (Equity += Intereses - Bad Debt).
    """
    
    # 1. Calcular Obligaciones (B x F)
    # Principal Repayment = tau * Loan
    # Interest Payment = rate * Loan
    principal_due = state.bank_firm_loans * DEBT_REIMBURSEMENT_RATE
    interest_due = state.bank_firm_loans * state.loan_interest_rates
    total_due_matrix = principal_due + interest_due
    
    # Total a pagar por firma (sumar columnas) -> (F,)
    total_due_per_firm = np.sum(total_due_matrix, axis=0)
    
    # 2. Identificar Solventes e Insolventes
    # Evitamos división por cero más adelante
    # Si total_due es 0, no hay problema (solvente)
    
    # Mask de firmas con deuda real para evitar ruido
    has_debt_mask = total_due_per_firm > 1e-9
    
    default_mask = np.zeros(N_FIRMS, dtype=bool)
    default_mask[has_debt_mask] = state.firm_cash[has_debt_mask] < total_due_per_firm[has_debt_mask]
    
    solvent_mask = ~default_mask & has_debt_mask
    
    # --- 3. Procesar Solventes ---
    if np.any(solvent_mask):
        # Disminuir Cash Firma
        state.firm_cash[solvent_mask] -= total_due_per_firm[solvent_mask]
        
        # Aumentar Cash Banco
        # Sumar los pagos agrupados por banco (filas de total_due_matrix)
        # Slicing: columnas que son solvent_mask
        payments_matrix = total_due_matrix[:, solvent_mask] # (B, N_solvent)
        payments_per_bank = np.sum(payments_matrix, axis=1) # (B,)
        state.bank_cash += payments_per_bank
        
        # Reducir Deuda Principal
        # OJO: Solo reducimos el PRINCIPAL pagado (tau * Loan), no los intereses.
        amortization_matrix = principal_due[:, solvent_mask]
        state.bank_firm_loans[:, solvent_mask] -= amortization_matrix
        
        # Actualizar Equity Banco (Ganancia = Intereses)
        # El banco recibe Principal + Interés. Principal es neutral para Equity (Asset swap: Loan -> Cash).
        # Interés es Ganancia (Aumenta Equity).
        interest_income_matrix = interest_due[:, solvent_mask]
        interest_income_per_bank = np.sum(interest_income_matrix, axis=1)
        state.bank_equity += interest_income_per_bank
        
    # --- 4. Procesar Defaults ---
    # Reseteo de registros de bad debt de este paso
    state.bank_bad_debt[:] = 0.0
    
    if np.any(default_mask):
        # Firmas en default pagan todo lo que tienen
        cash_available = state.firm_cash[default_mask]
        total_owed = total_due_per_firm[default_mask]
        
        # Ratio de recuperación: Cash / Owed
        recovery_rate = cash_available / total_owed # (N_def,)
        
        # Pagos a cada banco
        # Matrix de deuda subset (B, N_def)
        due_sub = total_due_matrix[:, default_mask]
        
        # Lo que se paga efectivamente (B, N_def) = Due_ij * Recovery_j
        recovered_matrix = due_sub * recovery_rate[np.newaxis, :]
        
        # Lo que se PIERDE (Bad Debt) (B, N_def) = Loan_Principal - Recovered_Principal?
        # Cuidado con la contabilidad.
        # El banco tiene un Activo (Loan).
        # Recibe Cash. La diferencia entre Valor Libro del Activo y Cash Recibido es Pérdida.
        # Valor Libro para cancelar = Loan Principal Total (porque quiebra, se borra todo el préstamo).
        # Cash Recibido = recovered_matrix (que incluye parte de intereses teóricos, pero es todo cash).
        # Pérdida = Loan_Principal_Total - Cash_Recibido.
        # Nota: Si Cash > Principal (raro en default), sería ganancia, pero asumimos default = no cubre 'due'.
        # 'due' incluye intereses.
        # Bad Debt = Loan_Principal (Outstanding) - Cash_Paid.
        
        loans_sub = state.bank_firm_loans[:, default_mask]
        
        bank_losses_matrix = loans_sub - recovered_matrix
        # Si recovered > loans (pago intereses parciales), la pérdida de PRINCIPAL es menor, o incluso ganancia.
        # Pero contablemente:
        # Asset change: -Loan
        # Cash change: +Recovered
        # Equity change: Cash - Loan
        bank_equity_delta_matrix = recovered_matrix - loans_sub
        
        # Sumar cambios por banco
        total_recovered_per_bank = np.sum(recovered_matrix, axis=1)
        total_equity_delta_per_bank = np.sum(bank_equity_delta_matrix, axis=1)
        
        # Aplicar cambios
        state.bank_cash += total_recovered_per_bank
        state.bank_equity += total_equity_delta_per_bank
        
        # Registro de Bad Debt (solo la parte negativa del equity delta, para estadísticas)
        # Bad Debt suele ser positivo
        state.bank_bad_debt -= np.minimum(0, total_equity_delta_per_bank)
        
        # --- REEMPLAZO DE FIRMAS ---
        # Resetear variables de firmas quebradas
        # Borrar préstamos (ya fueron "dados de baja" del activo del banco implícitamente al ajustar Equity, 
        # pero hay que poner ceros en la matriz para futuros pasos)
        state.bank_firm_loans[:, default_mask] = 0.0
        state.loan_interest_rates[:, default_mask] = 0.0
        
        # Resetear estado firma
        state.firm_cash[default_mask] = INITIAL_FIRM_CASH
        state.firm_stock[default_mask] = 0.0 # O stock inicial? 0 asumimos start fresh
        state.firm_cumulative_default[default_mask] += 1
        
        # Re-asignar ownership aleatorio
        state.firm_owners[default_mask] = np.random.randint(0, N_HOUSEHOLDS, size=np.sum(default_mask))
        
        # Nota: ¿De dónde sale el INITIAL_FIRM_CASH del "bailaout" o reinicio?
        # En modelos simples, "manna from heaven" o se resta de hogares.
        # Por ahora manna from heaven para mantener dinero en sistema estable.
        
    return state

def calculate_debtrank(interbank_matrix, bank_equity):
    """
    Calcula el Riesgo Sistémico Total usando una aproximación de DebtRank.
    
    Metodología:
    1. Matriz de Impacto W (B x B): W_ij = Cuánto pierde el banco i si j cae.
       W_ij = Loan_ij / Equity_i.
       (Loan_ij es Activo de i, Pasivo de j).
    2. Para cada banco k, simulamos su quiebra inicial (h_k = 1) y propagamos.
    3. DR_k = Suma de equity perdido en el sistema debido a k (excluyendo k).
    4. SR Total = Suma(DR_k).
    
    Retorna: float (Total Systemic Risk)
    """
    # Evitar división por cero en equity
    safe_equity = np.maximum(bank_equity, 1e-2)
    
    # 1. Matriz de Impacto Directo (Impact Matrix)
    # interbank_matrix[i, j] es lo que i le prestó a j.
    # Si j quiebra, i pierde interbank_matrix[i, j].
    # Impacto relativo en i: Loan[i, j] / Equity[i]
    # Broadcasting: (B, B) / (B, 1)
    impact_matrix = np.minimum(1.0, interbank_matrix / safe_equity[:, np.newaxis])
    
    total_sr = 0.0
    
    # Podríamos hacerlo todo matricial (B x B x B) pero B=20 es pequeño, loop es claro.
    # Iteramos cada banco como "paciente cero"
    for k in range(N_BANKS):
        # Vector de estado de stress (h)
        # h[i] = 1.0 significa quiebra total de i
        h = np.zeros(N_BANKS)
        h[k] = 1.0
        
        # Propagación iterativa (Power iteration style)
        # En DebtRank original, los nodos que quiebran se vuelven inactivos y no re-transmiten en ciclos infinitos,
        # pero aquí usamos la versión de stress continuo acumulativo simple.
        # Hacemos unos pocos pasos fijos de propagación (ej. 5 o 10 suficiente para red pequeña)
        
        for _ in range(5):
            # Nuevo stress que llega a i desde j: W_ij * h_j
            # new_stress = impact_matrix @ h
            # h(t+1) = min(1, h(t) + new_stress)
            # Pero cuidado de no contar doble lo que ya estaba.
            # Versión simple: h se actualiza acumulando.
            
            # Solo propagamos desde nodos activos recién estresados? No, DebtRank estándar propaga niveles.
            incoming_stress = impact_matrix @ h
            h_next = np.minimum(1.0, h + incoming_stress)
            
            # Si no hay cambio, break
            if np.allclose(h, h_next):
                break
            h = h_next
            
        # Impacto sistémico de k: Pérdida total de equity en el sistema (menos k)
        # Loss = sum_(i!=k) (h[i] * Equity[i])
        # Pero usamos h como proxy de % equity perdido.
        losses = h * safe_equity
        losses[k] = 0.0 # No contamos la pérdida "propia" inicial del shock
        
        dr_k = np.sum(losses)
        total_sr += dr_k
        
    return total_sr

def step6_interbank_market(state):
    """
    Paso 6: Mercado Interbancario con Impuesto de Riesgo Sistémico (SRT).
    
    Lógica:
    1. Identificar déficit/superávit de liquidez.
       Target = BUFFER_RATIO * Total_Assets.
    2. Bancos deficitarios intentan pedir prestado.
    3. Antes de aceptar, calculamos SR_pre y SR_post.
    4. Tax = SRT_SENSITIVITY * max(0, SR_post - SR_pre).
    5. Si (Tasa + Tax) aceptable, se hace el préstamo.
       El Tax se deduce del cash del prestatario (o prestamista? Paper: "Transaction Tax").
       Asumimos que lo paga quien inicia la transacción (borrower) o split. 
       Lo paga el Borrower para desincentivarlo a conectarse con nodos sistémicos.
    """
    
    # 1. Calcular Liquidez Objetivo
    # Assets = Loans to Firms (Activos Reales) + Interbank Loans (Activos Fin)
    # (Ignoramos cash como activo para el cálculo del buffer recursivo)
    real_assets = np.sum(state.bank_firm_loans, axis=1) # (B,)
    interbank_assets = np.sum(state.interbank_matrix, axis=1) # (B,)
    total_assets = real_assets + interbank_assets
    
    target_liquidity = total_assets * LIQUIDITY_BUFFER_RATIO
    liquidity_gap = target_liquidity - state.bank_cash
    
    # Identificar jugadores
    borrowers = np.where(liquidity_gap > 1.0)[0] # Necesitan cash
    lenders = np.where(liquidity_gap < -1.0)[0]  # Sobra cash (Gap negativo -> Surplus)
    
    # Barajar para orden aleatorio
    np.random.shuffle(borrowers)
    
    # SR Inicial
    current_sr = calculate_debtrank(state.interbank_matrix, state.bank_equity)
    state.total_systemic_risk = current_sr # Guardar métrica
    
    taxes_collected = 0.0
    
    for borrower_idx in borrowers:
        needed = liquidity_gap[borrower_idx]
        if needed < 1.0: continue
        
        # Buscar prestamista
        # Estrategia simple: Probar Lenders al azar
        potential_lenders = lenders.copy()
        np.random.shuffle(potential_lenders)
        
        for lender_idx in potential_lenders:
            if needed < 1e-2: break 
            
            # Cuánto puede prestar?
            surplus = -liquidity_gap[lender_idx] # Gap es negativo -> Surplus positivo
            if surplus < 1.0: continue
            
            amount = min(needed, surplus)
            
            # --- SIMULACIÓN DE RIESGO ---
            # Simular préstamo: Lender presta a Borrower
            # Activo Lender aumenta (Loan_LB), Activo Borrower (Cash) aumenta, Pasivo Borrower (Debt) aumenta.
            # Matriz[L, B] += Amount.
            
            # Crear copia de matriz para simular
            sim_matrix = state.interbank_matrix.copy()
            sim_matrix[lender_idx, borrower_idx] += amount
            
            # Calcular Nuevo SR
            new_sr = calculate_debtrank(sim_matrix, state.bank_equity)
            delta_sr = max(0.0, new_sr - current_sr)
            
            # --- CÁLCULO DE IMPUESTO ---
            tax_amount = amount * SRT_SENSITIVITY * delta_sr
            # Convertimos delta_sr que es absoluto (Equity Value) a relativo? 
            # No, la fórmula del paper (Ec 3) Tax = zeta * Delta_SR. 
            # Si Zeta es alto y Delta_SR es en unidades monetarias, el tax puede ser enorme.
            # Ajuste: Delta_SR suele normalizarse o zeta ser muy pequeño.
            # En parámetros pusimos ZETA=50, pero DR es en Dinero.
            # Revisemos paper: "expected systemic loss". Es dinero.
            # Si Delta SR = 1000 (un banco entero extra en riesgo), Tax = 50 * 1000 = 50k. ¡Enorme!
            # El paper dice zeta = 1 (Page 5). "Tax rate is proportional to marginal increase".
            # Probablemente Delta_SR se mide como fracción del total system size o similar.
            # O el tax rate es per dollar transaction?
            # Ec 4: Transaction Tax = Amount * Tax_Rate(Amount)? No.
            # Dice "Payment of tax... proportional to increase in SR".
            # Asumiremos ZETA pequeño en params o que SR está normalizado.
            # Miremos calculate_debtrank output: Suma de Equity losses. Es Dinero.
            # Ajustaremos tax a algo razonable: Zeta * (Delta_SR / Total_Equity_Sistema).
            
            total_system_equity = np.sum(state.bank_equity)
            normalized_delta_sr = delta_sr / max(total_system_equity, 1.0)
            
            # Ajuste de sensibilidad del tax (heurístico para que tenga impacto pero no mate el mercado)
            real_tax = amount * SRT_SENSITIVITY * normalized_delta_sr
            
            # --- DECISIÓN ---
            # El borrower acepta si (Interbank_Rate * Amount + Tax) < Costo Alternativo?
            # Por ahora asumimos que acepta si tiene cash para pager el tax (o se deduce del prestamo).
            # Si el tax > amount, absurdo.
            
            total_cost_upfront = real_tax
            
            # Verificamos si vale la pena (Tax < 10% del préstamo? criterio heurístico)
            # O si el borrower puede pagarlo de su equity?
            # Asumamos que el tax se paga CASH from Borrower.
            
            # Si el tax es muy alto, abortamos esta transacción (buscamos otro Partner menos sistémico).
            if real_tax > amount * 0.05: # Si el impuesto es > 5% del principal, es muy caro.
                continue
            
            # Ejecutar Transacción
            state.interbank_matrix[lender_idx, borrower_idx] += amount
            state.bank_cash[lender_idx] -= amount
            state.bank_cash[borrower_idx] += amount # Recibe préstamo
            
            # Pagar Tax (Sale del sistema)
            if state.bank_cash[borrower_idx] >= real_tax:
                state.bank_cash[borrower_idx] -= real_tax
                taxes_collected += real_tax
            else:
                # No puede pagar el tax, revertimos transacción (no ocurre)
                state.interbank_matrix[lender_idx, borrower_idx] -= amount
                state.bank_cash[lender_idx] += amount
                state.bank_cash[borrower_idx] -= amount
                continue
                
            # Actualizar Gaps locales para loop
            needed -= amount
            liquidity_gap[borrower_idx] -= amount
            liquidity_gap[lender_idx] += amount
            
            # Actualizar SR base para siguiente transacción implícitamente?
            # Sí, el sistema cambió. El próximo préstamo se evalúa sobre el nuevo estado.
            current_sr = new_sr
            
            if needed < 1e-2: break 
            
    state.collected_tax = taxes_collected
    state.total_systemic_risk = current_sr # Final SR
    return state
