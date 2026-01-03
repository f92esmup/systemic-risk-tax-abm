import numpy as np
from state import EconomyState
from logic import step1_firms_planning, step2_banks_lending, step3_production
from parameters import *

def verify_step3():
    print("--- Verificando Paso 3: Producción y Labor ---")
    
    state = EconomyState()
    
    # Configuración para provocar escasez masiva de trabajadores
    # Cash suficiente para que la restricción sea N_HOUSEHOLDS, no dinero.
    state.firm_cash[:] = 1e6 
    state.bank_cash[:] = 1e9
    
    # Pre-pasos
    state = step1_firms_planning(state) # Genera demanda total ~10,000
    state = step2_banks_lending(state)
    
    # Capturar estado previo
    initial_firm_cash = np.copy(state.firm_cash)
    initial_hh_cash = np.copy(state.household_cash)
    initial_stock = np.copy(state.firm_stock)
    
    # Ejecutar Paso 3
    state = step3_production(state)
    
    # Verificaciones
    
    # 1. Restricción de Población
    # El número total de empleados no puede superar N_HOUSEHOLDS
    total_employed = np.sum(state.household_employer != -1)
    print(f"Total Empleados: {total_employed} / {N_HOUSEHOLDS}")
    assert total_employed <= N_HOUSEHOLDS, "Se contrató más gente de la que existe"
    
    # Debería estar cerca del 100% de ocupación dadas las demandas altas
    assert total_employed > N_HOUSEHOLDS * 0.95, "El desempleo es sospechosamente alto dado el exceso de demanda"
    
    # 2. Conservación de Dinero (Salarios)
    # Lo que pagaron las firmas == Lo que recibieron los hogares
    firm_cash_delta = np.sum(initial_firm_cash) - np.sum(state.firm_cash)
    hh_cash_delta = np.sum(state.household_cash) - np.sum(initial_hh_cash)
    
    print(f"Firmas pagaron: {firm_cash_delta:.2f}")
    print(f"Hogares recibieron: {hh_cash_delta:.2f}")
    
    assert np.isclose(firm_cash_delta, hh_cash_delta), "El dinero de salarios se perdió o creó en el camino"
    assert firm_cash_delta > 0, "No se pagaron salarios"
    
    # 3. Producción Consistente
    # Stock increase = Total Employees * alpha
    stock_increase = np.sum(state.firm_stock) - np.sum(initial_stock)
    expected_production = total_employed * LABOR_PRODUCTIVITY
    
    print(f"Producción Real: {stock_increase:.2f} (Esperada: {expected_production:.2f})")
    assert np.isclose(stock_increase, expected_production), "La producción no coincide con la mano de obra contratada"
    
    print("✅ Verificación del Paso 3 completada.")

if __name__ == "__main__":
    verify_step3()
