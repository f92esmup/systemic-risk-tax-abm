import numpy as np
from parametros import Param as p

def paso4(state, params):
    """
    PASO 4: Mercado de Bienes (Hogares -> Empresas)
    
    1. Hogares determinan presupuesto de consumo.
    2. Selección de vendedor: Cada hogar compara precios de una muestra de empresas.
    3. Transacción: Compra al menor precio, sujeta a disponibilidad (Racionamiento).
    
    Args:
        state (dict): Estado del sistema.
        params (class): Parámetros globales.
        
    Returns:
        dict: Actualizaciones de inventarios, liquidez y demanda registrada.
    """
    
    # --- A. PREPARACIÓN ---
    H = params.H
    F = params.F
    
    # Datos de entrada
    P_firms = state['firms_prices']            # (F,)
    S_firms = state['firms_inventory']         # (F,) Oferta disponible (Inv + Prod)
    # Check key consistency: main.py uses 'households_deposits', logic uses 'households_liquidity' ??
    # main.py initialization: 'households_deposits'
    # paso4 prompt code: 'households_liquidity'
    # I should use the key from main.py or update main.py. 
    # Let's check main.py state keys again.
    # main.py has 'households_deposits'.
    # I will stick to 'households_deposits' to match main.py.
    M_households = state['households_deposits'] # (H,)
    
    # 1. Presupuesto de Consumo (Propensión marginal al consumo)
    # Gualdi et al: Budget = c * Savings
    budget_H = M_households * params.PROPENSION_CONSUMO
    
    # --- B. MATCHING VECTORIAL (BÚSQUEDA DE PRECIOS) ---
    # Simulamos que cada hogar visita 'VISITS' empresas aleatorias y elige la barata.
    VISITS = params.Z_CONSUMO # Número de empresas que "mira" cada consumidor
    
    # Matriz de índices aleatorios (H, VISITS) -> Qué empresas visita cada hogar
    # [AUDIT FIX] Strict random sampling with replacement (households visit Z random firms)
    rng = np.random.default_rng()
    visited_indices = rng.integers(0, F, size=(H, VISITS))
    
    # Obtener precios de esas empresas: (H, VISITS)
    prices_seen = P_firms[visited_indices]
    
    # Encontrar índice (0..VISITS-1) del precio mínimo por fila
    best_local_idx = np.argmin(prices_seen, axis=1)
    
    # Obtener el índice global de la empresa elegida (H,)
    # Select from visited_indices using row_range and best_local_idx
    chosen_firms = visited_indices[np.arange(H), best_local_idx]
    
    # --- C. AGREGACIÓN DE DEMANDA (INTENCIÓN DE COMPRA) ---
    # Calcular cuánto quiere gastar cada hogar en su empresa elegida.
    # Demanda monetaria: Budget. Demanda física: Budget / Precio
    chosen_prices = P_firms[chosen_firms]
    
    # Cantidad demandada por hogar (física)
    demand_H_qty = np.zeros(H)
    mask_price = chosen_prices > 0
    demand_H_qty[mask_price] = budget_H[mask_price] / chosen_prices[mask_price]
    
    # Agregar demanda total por empresa (Vectorización de la suma)
    # total_demand_F[f] = sum(demand_H_qty where chosen_firms == f)
    total_demand_qty_F = np.bincount(chosen_firms, weights=demand_H_qty, minlength=F)
    # total_demand_monetary_F = np.bincount(chosen_firms, weights=budget_H, minlength=F) # Unused
    
    # --- D. RACIONAMIENTO (MERCADO) ---
    # Ventas reales = min(Demanda, Inventario)
    sales_qty_F = np.minimum(total_demand_qty_F, S_firms)
    
    # Calcular fracción de satisfacción para ajustar el gasto de los hogares
    # Si una empresa vendió todo (sales < demand), los hogares solo gastaron una fracción.
    # Rationing ratio alpha = sales / demand (0..1)
    rationing_ratio_F = np.ones(F)
    mask_demand = total_demand_qty_F > 1e-9
    rationing_ratio_F[mask_demand] = sales_qty_F[mask_demand] / total_demand_qty_F[mask_demand]
    
    # --- E. ACTUALIZACIÓN DE HOGARES (Gasto Real) ---
    # Gasto real del hogar i = Presupuesto_i * rationing_ratio_(empresa_elegida)
    # Mapear el ratio de la empresa de vuelta al hogar
    household_rationing = rationing_ratio_F[chosen_firms]
    
    actual_spending_H = budget_H * household_rationing
    
    # Actualizar liquidez hogares (Salida de dinero)
    M_households_new = M_households - actual_spending_H
    
    # --- F. ACTUALIZACIÓN EMPRESAS (Ingresos y Stock) ---
    # Ingresos = Ventas * Precio
    revenue_F = sales_qty_F * P_firms
    
    # Nuevo Inventario = Inventario Previo - Ventas
    inventory_new_F = S_firms - sales_qty_F
    
    # Generar matriz de flujo de consumo (H -> F) para visualización (Opcional/Sparse)
    # Por eficiencia, devolvemos solo agregados, salvo que se pida explícitamente la red.
    # consumption_matrix_HF = ... (Omitido por eficiencia si no se grafica cada link)
    
    return {
        'firms_sales_qty': sales_qty_F,
        'firms_revenue': revenue_F,
        'firms_inventory': inventory_new_F,
        'firms_demand_received': total_demand_qty_F, # Para el Paso 1 del siguiente turno (Expectativas)
        'households_deposits': M_households_new, # Key matching main.py
        'consumption_flows': (chosen_firms, actual_spending_H) # Tupla comprimida para logging si es necesario
    }