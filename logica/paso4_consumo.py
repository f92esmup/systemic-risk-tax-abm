import numpy as np
from parametros import Parametros

def ejecutar_paso4(modelo):
    """
    Paso 4: Mercado de Consumo y Salarios.
    - Hogares reciben salarios.
    - Decisión de consumo y ahorro.
    - Compra de bienes a firmas.
    """
    
    # Reset Profit Trackers
    modelo.current_step_profit_firms.fill(0.0)
    modelo.current_step_profit_bancos.fill(0.0)

    # --- A. WAGE PAYMENT ---
    wage_bills = modelo.estado_firmas[:, Parametros.IDX_FIRM_WAGES]
    
    # Firms pay their workers (even if it takes liquidity negative)
    # Note: In Step 2 they raised liquidity specifically for this.
    modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= wage_bills
    # Record Expense
    modelo.current_step_profit_firms -= wage_bills

    # Distribute to Households (Workers) using MATRIZ_LABORAL
    # Count employees per firm: Sum columns of Matrix (H x F) -> (F,)
    employee_counts = np.sum(modelo.matriz_laboral, axis=0)

    # Wage per worker
    wage_per_worker = np.zeros(Parametros.F)
    mask_c = employee_counts > 0
    np.divide(wage_bills, employee_counts, out=wage_per_worker, where=mask_c)

    # HH Income = Matriz_Laboral @ Wage_Per_Worker
    # (H x F) @ (F,) -> (H,)
    hh_income = modelo.matriz_laboral @ wage_per_worker
    modelo.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_income

    # --- B. CONSUMPTION MARKET ---

    # 1. Budget
    hh_deposits = modelo.estado_hogares[:, Parametros.IDX_HH_DEPOSITS]
    budgets = hh_deposits * Parametros.c

    # 2. Firm Selection (Z-Search)
    z_indices = modelo.rng.integers(
        0, Parametros.F, size=(Parametros.H, Parametros.Z_CONSUMPTION)
    )
    prices_options = modelo.estado_firmas[z_indices, Parametros.IDX_FIRM_PRICE]
    winner_local_indices = np.argmin(prices_options, axis=1)
    winner_global_indices = z_indices[np.arange(Parametros.H), winner_local_indices]

    # 3. Aggregate Demand
    demand_monetary = np.bincount(
        winner_global_indices, weights=budgets, minlength=Parametros.F
    )

    # 4. Sales & Rationing
    firm_prices = modelo.estado_firmas[:, Parametros.IDX_FIRM_PRICE]
    firm_inventory = modelo.estado_firmas[:, Parametros.IDX_FIRM_PROD]
    max_revenue = firm_inventory * firm_prices
    actual_revenue = np.minimum(demand_monetary, max_revenue)

    # Sales Quantity
    sales_qty = np.zeros(Parametros.F)
    price_mask = firm_prices > 1e-9
    np.divide(actual_revenue, firm_prices, out=sales_qty, where=price_mask)

    # Update Firms
    modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] += actual_revenue
    modelo.estado_firmas[:, Parametros.IDX_FIRM_PROD] -= sales_qty
    
    # Record Revenue
    modelo.current_step_profit_firms += actual_revenue

    # 5. Households Expenditure & RECORDING (Matriz Consumo)
    # Clear previous step consumption
    modelo.matriz_consumo.fill(0.0)

    scale_factors = np.ones(Parametros.F)
    demand_mask = demand_monetary > 1e-9
    np.divide(actual_revenue, demand_monetary, out=scale_factors, where=demand_mask)
    scale_factors = np.minimum(1.0, scale_factors)

    hh_scale = scale_factors[winner_global_indices]
    hh_expenditure = budgets * hh_scale

    modelo.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] -= hh_expenditure

    # Record in Matrix: Rows=Households, Cols=Firms
    np.add.at(
        modelo.matriz_consumo,
        (np.arange(Parametros.H), winner_global_indices),
        hh_expenditure,
    )
