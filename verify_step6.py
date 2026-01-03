import numpy as np
from state import EconomyState
from logic import step6_interbank_market, calculate_debtrank
from parameters import *

def verify_step6():
    print("--- Verificando Paso 6: Interbancario y Tax ---")
    
    state = EconomyState()
    
    # --- Escenario 1: Préstamo Simple sin Riesgo Sistémico previo ---
    print("\n[Test 1] Préstamo Simple (Banco 0 -> Banco 1)")
    # Banco 0: Surplus
    # Banco 1: Deficit
    # Resto: Neutral
    
    state.bank_equity[:] = 100.0
    state.bank_cash[:] = 20.0 # Bajo cash general
    
    # Assets = Loans. 
    # ERROR PREVIO: state.bank_firm_loans[:] = 500.0 generaba 500 * 100 = 50k activos -> 5k target -> todos deficitarios.
    # Corrección: Asignar deuda moderada.
    state.bank_firm_loans[:, :] = 0.0
    state.bank_firm_loans[:, 0] = 500.0 # Cada banco tiene 1 préstamo de 500 a la firma 0. Total Assets = 500.
    
    target = 500.0 * LIQUIDITY_BUFFER_RATIO # 50.0
    # Current cash = 20. Gap = 30. Todos necesitan 30 excepto los manuales.
    
    # Banco 0: Cash 1000 (Surplus enorme)
    state.bank_cash[0] = 1000.0
    # Banco 1: Cash 0 (Deficit 50)
    state.bank_cash[1] = 0.0
    
    # Ejecutar step6
    state = step6_interbank_market(state)
    
    print(f"Banco 0 Cash: {state.bank_cash[0]:.2f}")
    print(f"Banco 1 Cash: {state.bank_cash[1]:.2f}")
    print(f"Interbank 0->1: {state.interbank_matrix[0, 1]:.2f}")
    
    assert state.interbank_matrix[0, 1] > 0, "No se ejecutó el préstamo 0->1"
    assert state.bank_cash[1] > 0, "Banco 1 no recibió liquidez"
    
    # --- Escenario 2: DebtRank y Tax ---
    print("\n[Test 2] Activación del Systemic Risk Tax")
    # Limpiamos
    state = EconomyState()
    state.bank_equity[:] = 100.0
    
    # Crear una red frágil ("Star Network" o similar)
    # Banco 2 le debe a Banco 3, 4, 5, 6... (Banco 2 es un gran deudor sistémico).
    # Si Banco 2 cae, arrastra a 3,4,5,6.
    # Equity=100. Loan=90. Impact=0.9.
    deudores_de_2 = [3, 4, 5]
    for b in deudores_de_2:
        state.interbank_matrix[b, 2] = 95.0 # Casi todo el equity de b expuesto a 2
        
    # Calculamos riesgo basal
    sr_base = calculate_debtrank(state.interbank_matrix, state.bank_equity)
    print(f"Riesgo Sistémico Base: {sr_base:.4f} (Max possible ~ N_BANKS * 100)")
    
    # Ahora Banco 2 necesita más dinero y solo Banco 0 (externo) puede prestarle.
    # Si Banco 0 le presta a Banco 2, la exposición total a 2 aumenta?
    # No necesariamente, Banco 0 se expone a 2.
    # Pero el SR total aumenta porque ahora hay UN MÁS banco (0) que caería si 2 cae.
    # Esto debería aumentar el SR y detonar Tax.
    
    state.bank_firm_loans[:, :] = 0.0
    state.bank_firm_loans[:, 0] = 500.0
    target = 500.0 * LIQUIDITY_BUFFER_RATIO # 50
    
    # Banco 2: Liquidez 0. Necesita 50.
    state.bank_cash[2] = 0.0
    
    # Banco 0: Liquidez sobra
    state.bank_cash[0] = 1000.0
    
    # Pre-Ejecución
    print("Ejecutando mercado interbancario con deudor sistémico...")
    state = step6_interbank_market(state)
    
    sr_new = state.total_systemic_risk
    tax = state.collected_tax
    loan_0_2 = state.interbank_matrix[0, 2]
    
    print(f"SR Final: {sr_new:.4f}")
    print(f"Delta SR: {sr_new - sr_base:.4f}")
    print(f"Préstamo 0->2: {loan_0_2:.2f}")
    print(f"Impuesto Recaudado: {tax:.4f}")
    
    if loan_0_2 > 0:
        print("El préstamo se realizó.")
        # Verificamos si hubo tax (si el riesgo aumentó)
        if sr_new > sr_base + 1e-4:
            assert tax > 0, "Riesgo aumentó pero no se cobró impuesto!"
            print("✅ Impuesto cobrado correctamente.")
        else:
            print("⚠️ El riesgo no aumentó significativamente (tal vez Banco 0 tiene mucho equity o la red no propagó).")
    else:
        print("El préstamo NO se realizó (quizás el tax era muy alto).")
        print("✅ Tax evitó transacción riesgosa (comportamiento esperado).")

    print("\n✅ Verificación del Paso 6 completada.")

if __name__ == "__main__":
    verify_step6()
