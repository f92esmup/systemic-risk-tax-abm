import numpy as np
from parametros import Parametros


def ejecutar_paso1(modelo):
    """
    Paso 1: Planificación de Firmas.
    - Actualización de Precios.
    - Actualización de Expectativas de Demanda.
    - Definición de Demanda de Trabajo y Capital (Target).
    - Calculo de Demanda de Crédito (Estimada basada en Target).
    """

    # --- 1. Update Prices ---
    prices = modelo.estado_firmas[:, Parametros.IDX_FIRM_PRICE]

    # Save to PREV
    modelo.estado_firmas[:, Parametros.IDX_FIRM_PRICE_PREV] = prices.copy()

    # Calculate market average price
    p_avg = np.mean(prices)

    # Noise component
    noise = modelo.rng.normal(0, Parametros.PRICE_DRIFT_STD, size=Parametros.F)

    # Adjustment Rule: p_new = p * (1 + speed * (p_avg - p)/p_avg + noise)
    if p_avg > 1e-9:
        adjustment = Parametros.PRICE_ADJUSTMENT_SPEED * (p_avg - prices) / p_avg
        new_prices = prices + prices * (adjustment + noise)
    else:
        new_prices = prices * (1 + noise)

    new_prices = np.maximum(new_prices, 0.01)
    modelo.estado_firmas[:, Parametros.IDX_FIRM_PRICE] = new_prices

    # --- 2. Update Demand Expectations ---
    modelo.estado_firmas[:, Parametros.IDX_FIRM_DEMAND_PREV] = modelo.estado_firmas[
        :, Parametros.IDX_FIRM_DEMAND
    ].copy()

    current_demand = modelo.estado_firmas[:, Parametros.IDX_FIRM_DEMAND]
    if np.all(current_demand == 0):
        current_demand = modelo.rng.uniform(10, 50, size=Parametros.F)

    # Random Walk
    demand_shock = modelo.rng.normal(0, 0.05, size=Parametros.F)
    new_demand = current_demand * (1 + demand_shock)
    new_demand = np.maximum(new_demand, 0.0)

    modelo.estado_firmas[:, Parametros.IDX_FIRM_DEMAND] = new_demand

    # --- 3. Production Planning (Defining Labour Demand) ---
    labor_needed = np.ceil(new_demand / Parametros.alpha)

    modelo.estado_firmas[:, Parametros.IDX_FIRM_WORKERS] = labor_needed
    # Initial estimate of production/wages (Target)
    modelo.estado_firmas[:, Parametros.IDX_FIRM_PROD] = labor_needed * Parametros.alpha
    wage_bill = labor_needed * Parametros.WAGE
    modelo.estado_firmas[:, Parametros.IDX_FIRM_WAGES] = wage_bill

    # --- 4. Financial Health Update (Leverage) ---
    current_debt = np.sum(modelo.matriz_credito_firmas, axis=1)  # (F,)
    liquidity = modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]

    leverage = current_debt / (liquidity + 1e-9)
    modelo.estado_firmas[:, Parametros.IDX_FIRM_LEVERAGE] = leverage

    # --- 5. Credit Demand Calculation (Based on Plan) ---
    # We estimate needs based on TARGET wages
    gap = wage_bill - liquidity
    credit_demand = np.maximum(gap, 0.0)

    modelo.current_firm_credit_demand = credit_demand
