import numpy as np
from state import EconomyState
from logic import step1_firms_planning, step2_banks_lending
from parameters import *

def verify_step2():
    print("--- Verificando Paso 2: Mercado de Crédito ---")
    
    # Inicializar estado y forzar necesidades
    state = EconomyState()
    
    # Configurar: 
    # Banco 0: Muy específico (chi cercano a 1) -> Tasas altas
    # Banco 1: Poco específico (chi cercano a 0) -> Tasas bajas (competitivo)
    # Pero las 'chi' son random en logic.py. Para test determinista deberíamos mockear o chequear propiedades estadísticas.
    # Vamos a confiar en la lógica de selección: "La firma siempre debe elegir la tasa mínima disponible de su subset".
    
    # Forzamos demanda
    state.firm_cash[:] = 0.0 # Todas necesitan crédito
    state.bank_cash[:] = 1e9 # Bancos infinitamente ricos para evitar credit crunch por ahora
    
    # Ejecutar pasos
    state = step1_firms_planning(state) # Genera demandas
    
    # Pre-check
    total_demand = np.sum(state.firm_credit_demand)
    print(f"Demanda Total Inicial: {total_demand:.2f}")
    assert total_demand > 0, "No hay demanda de crédito generada"
    
    # Ejecutar Paso 2
    state = step2_banks_lending(state)
    
    # Verificaciones
    
    # 1. Tasa de Asignación 
    # Dado que los bancos tienen cash infinito, todas las firmas con rate <= r_max deberían haber recibido crédito.
    # Chequeamos cuántas recibieron.
    granted = np.sum(state.new_loans_granted)
    print(f"Crédito Total Otorgado: {granted:.2f}")
    
    # Si granted < demand, puede ser por r_max contracting
    assert granted > 0, "No se otorgaron préstamos a pesar de tener liquidez infinita"
    
    # 2. Verificación de Selección Racional
    # Comprobar para una firma aleatoria si el banco elegido fue válido
    for f in range(5): # Check primeras 5 firmas
        if state.new_loans_granted[f] > 0:
            bank = state.firm_lender_choice[f]
            rate = state.loan_interest_rates[bank, f]
            print(f"Firma {f} eligió Banco {bank} con tasa {rate:.4f}")
            assert rate >= REFINANCING_RATE, "La tasa no puede ser menor a la de refinanciación"
            
    # 3. Consistencia de Balances
    # El cash TOTAL de los bancos debió bajar
    total_bank_cash_initial = 1e9 * N_BANKS
    assert np.sum(state.bank_cash) < total_bank_cash_initial, "El cash TOTAL de los bancos no disminuyó"
    
    # Conservación de flujos: Lo que bajó en bancos debió subir en firmas
    cash_decrease_banks = total_bank_cash_initial - np.sum(state.bank_cash)
    cash_increase_firms = np.sum(state.firm_cash) # Iniciaron en 0
    assert np.isclose(cash_decrease_banks, cash_increase_firms), \
        f"Fallo de conservación: Bancos bajaron {cash_decrease_banks}, Firmas subieron {cash_increase_firms}"
    
    print("✅ Verificación del Paso 2 completada.")

if __name__ == "__main__":
    verify_step2()
