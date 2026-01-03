import numpy as np
from state import EconomyState
from logic import step4_consumption
from parameters import *

def verify_step4():
    print("--- Verificando Paso 4: Consumo ---")
    
    state = EconomyState()
    
    # Setup: Hogares con dinero, Firmas con stock y precios variados
    state.household_cash[:] = 100.0 # Budget = 100 * 0.8 = 80
    state.firm_stock[:] = 1000.0    # Stock abundante por defecto
    state.firm_cash[:] = 0.0
    
    # Precios heterogéneos para probar selección
    state.firm_prices[:] = 2.0
    state.firm_prices[0] = 1.0 # Firma 0 barata -> Debería atraer más demanda
    
    # Caso especial: Firma 1 sin stock para probar racionamiento
    state.firm_stock[1] = 0.0
    state.firm_prices[1] = 0.5 # Muy barata pero sin stock
    
    # Ejecutar Paso 4
    initial_hh_cash_sum = np.sum(state.household_cash)
    initial_firm_stock_sum = np.sum(state.firm_stock)
    
    state = step4_consumption(state)
    
    # Verificaciones
    
    # 1. Conservación de Dinero
    # Dinero gastado = Dinero recibido
    final_hh_cash_sum = np.sum(state.household_cash)
    total_revenue = np.sum(state.firm_cash)
    spending = initial_hh_cash_sum - final_hh_cash_sum
    
    print(f"Gasto Hogares: {spending:.2f}")
    print(f"Ingreso Firmas: {total_revenue:.2f}")
    assert np.isclose(spending, total_revenue), "Dinero desaparecido en la transacción"
    
    # 2. Selección de Precios (Firma 0 vs Resto)
    # Dado que z=2, la probabilidad de que un hogar elija la firma 0 (si es elegida entre las 2) es alta.
    # Ingreso Firma 0 debería ser significativamente mayor que el promedio de las otras (si fueron elegidas).
    # Bueno, como z=2 es bajo y N=100, muchas no verán a la firma 0.
    # Pero las que vean a la 1 (stock 0) no deberían haber comprado nada.
    
    print(f"Ingreso Firma 0 (Barata, Stock OK): {state.firm_cash[0]:.2f}")
    print(f"Ingreso Firma 1 (Muy Barata, Sin Stock): {state.firm_cash[1]:.2f}")
    
    assert state.firm_cash[1] < 1e-5, "Firma 1 vendió productos sin tener stock!"
    assert state.firm_stock[1] >= 0, "Stock negativo en firma 1"
    
    # 3. Reducción de Stock General
    # Stock final < Stock Inicial
    final_firm_stock_sum = np.sum(state.firm_stock)
    stock_delta = initial_firm_stock_sum - final_firm_stock_sum
    print(f"Stock Vendido Total: {stock_delta:.2f}")
    assert stock_delta > 0, "No se vendió nada"
    
    print("✅ Verificación del Paso 4 completada.")

if __name__ == "__main__":
    verify_step4()
