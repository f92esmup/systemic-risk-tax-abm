import numpy as np
from state import EconomyState
from logic import step1_firms_planning
from parameters import *

def verify_step1():
    print("--- Verificando Paso 1: Demanda de Firmas ---")
    
    # Inicializar estado
    state = EconomyState()
    # Modificamos el cash de algunas firmas para forzar situaciones de crédito
    state.firm_cash[0] = 0.0 # Esta firma necesitará crédito seguro
    state.firm_cash[1] = 1e6 # Esta firma NO debería necesitar crédito
    
    # Ejecutar Paso 1
    state = step1_firms_planning(state)
    
    # Verificaciones Tensoriales
    
    # 1. No debe haber demandas negativas
    assert np.all(state.firm_expected_demand >= 0), "Error: Demanda negativa detectada"
    assert np.all(state.firm_labor_demand >= 0), "Error: Demanda laboral negativa"
    
    # 2. Relación de Producción
    # N * alpha debe ser igual a Y (con margen de error flotante)
    production_check = state.firm_labor_demand * LABOR_PRODUCTIVITY
    assert np.allclose(production_check, state.firm_planned_production), "Error: Función de producción inconsistente"
    
    # 3. Lógica de Crédito
    # Firma 0 (sin cash) debe pedir crédito igual a su wage bill
    assert state.firm_credit_demand[0] == state.firm_wage_bill[0], \
        f"Firma 0 falló: Credit gap ({state.firm_credit_demand[0]}) != Wage Bill ({state.firm_wage_bill[0]})"
    
    # Firma 1 (mucha cash) no debe pedir crédito
    assert state.firm_credit_demand[1] == 0.0, \
        f"Firma 1 falló: Pidió crédito ({state.firm_credit_demand[1]}) teniendo cash de sobra"
        
    print("✅ Verificación del Paso 1 completada con éxito.")
    print(f"Ejemplo - Firma 0: Demanda {state.firm_expected_demand[0]:.2f}, Crédito: {state.firm_credit_demand[0]:.2f}")
    print(f"Total Crédito Solicitado en el sistema: {np.sum(state.firm_credit_demand):.2f}")

if __name__ == "__main__":
    verify_step1()
