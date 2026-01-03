import numpy as np
from state import EconomyState
from logic import step7_evolution
from parameters import *

def verify_step7():
    print("--- Verificando Paso 7: Evolución y Aprendizaje ---")
    
    state = EconomyState()
    
    # Setup
    initial_demand = 10.0
    initial_price = 2.0
    
    state.firm_expected_demand[:] = initial_demand
    state.firm_prices[:] = initial_price
    
    # Caso 1: Firma 0 - Stock Sobrante (Overproduction)
    # Stock = 5.0. No vendió todo.
    # Esperamos: Expectativa Demanda Baje, Precio Baje.
    state.firm_stock[0] = 5.0
    
    # Caso 2: Firma 1 - Stock Agotado (Underproduction)
    # Stock = 0.0. Vendió todo.
    # Esperamos: Expectativa Demanda Suba, Precio Suba.
    state.firm_stock[1] = 0.0
    
    # Ejecutar Step 7
    state = step7_evolution(state)
    
    # Verificaciones
    
    # Firma 0
    new_demand_0 = state.firm_expected_demand[0]
    new_price_0 = state.firm_prices[0]
    print(f"\n[Firma 0 - Stock Sobrante (5.0)]")
    print(f"Demanda: {initial_demand} -> {new_demand_0:.2f} (Esperado < {initial_demand})")
    print(f"Precio:  {initial_price} -> {new_price_0:.2f} (Esperado < {initial_price})")
    
    assert new_demand_0 < initial_demand, "La demanda debería bajar por stock sobrante"
    assert new_price_0 < initial_price, "El precio debería bajar por stock sobrante"
    
    # Firma 1
    new_demand_1 = state.firm_expected_demand[1]
    new_price_1 = state.firm_prices[1]
    print(f"\n[Firma 1 - Stock Agotado (0.0)]")
    print(f"Demanda: {initial_demand} -> {new_demand_1:.2f} (Esperado > {initial_demand})")
    print(f"Precio:  {initial_price} -> {new_price_1:.2f} (Esperado > {initial_price})")
    
    assert new_demand_1 > initial_demand, "La demanda debería subir por escasez"
    assert new_price_1 > initial_price, "El precio debería subir por escasez"
    
    print("\n✅ Verificación del Paso 7 completada.")

if __name__ == "__main__":
    verify_step7()
