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
