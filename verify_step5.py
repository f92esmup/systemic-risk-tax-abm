import numpy as np
from state import EconomyState
from logic import step5_firm_repayment
from parameters import *

def verify_step5():
    print("--- Verificando Paso 5: Repago y Quiebras ---")
    
    state = EconomyState()
    
    # --- Configuración del Escenario ---
    
    # 1. Firma Solvente (Firma 0)
    # Debe 100, Tasa 5%. Amortización 5% (tau=0.05).
    # Due = 100*0.05 + 100*0.05 = 5 + 5 = 10.
    # Cash = 20 (Suficiente)
    state.bank_firm_loans[0, 0] = 100.0
    state.loan_interest_rates[0, 0] = 0.05
    state.firm_cash[0] = 20.0
    
    # 2. Firma Insolvente / Default (Firma 1)
    # Debe 100. Due = 10.
    # Cash = 5 (Insuficiente)
    # Resultado esperado: Paga 5. Banco pierde (100 - 5) = 95. Firma reseteada.
    state.bank_firm_loans[0, 1] = 100.0
    state.loan_interest_rates[0, 1] = 0.05
    state.firm_cash[1] = 5.0
    
    # Estado inicial de bancos
    initial_bank_equity = state.bank_equity[0]
    initial_bank_cash = state.bank_cash[0]
    
    # Ejecutar Paso 5
    state = step5_firm_repayment(state)
    
    # --- Verificaciones ---
    
    # 1. Caso Solvente (Firma 0)
    print("\n--- Analizando Firma 0 (Solvente) ---")
    # Deuda nueva = 100 - 5 (amortizado) = 95
    loan_0 = state.bank_firm_loans[0, 0]
    cash_0 = state.firm_cash[0]
    print(f"Deuda restante F0: {loan_0:.2f} (Esperado 95.00)")
    print(f"Cash restante F0: {cash_0:.2f} (Esperado 10.00)")
    
    assert np.isclose(loan_0, 95.0), f"Deuda F0 incorrecta: {loan_0}"
    assert np.isclose(cash_0, 10.0), f"Cash F0 incorrecto: {cash_0}"
    
    # 2. Caso Insolvente (Firma 1)
    print("\n--- Analizando Firma 1 (Default) ---")
    # Firma debe ser reseteada
    loan_1 = state.bank_firm_loans[0, 1]
    cash_1 = state.firm_cash[1]
    cum_def = state.firm_cumulative_default[1]
    
    print(f"Deuda restante F1: {loan_1:.2f} (Esperado 0.00 - Reset)")
    print(f"Cash actual F1: {cash_1:.2f} (Esperado {INITIAL_FIRM_CASH} - Reset)")
    print(f"Defaults acumulados F1: {cum_def} (Esperado 1)")
    
    assert loan_1 == 0.0, "La deuda de la firma quebrada no se borró"
    assert cash_1 == INITIAL_FIRM_CASH, "La firma quebrada no se re-capitalizó"
    assert cum_def == 1, "No se contó el default"
    
    # 3. Balance del Banco 0
    print("\n--- Analizando Banco 0 ---")
    # Cash Inflow: 10 (F0) + 5 (F1) = 15
    # Equity Change: 
    #   Por F0: +Interés = +5
    #   Por F1: Recibió 5, Canceló Loan 100 -> Delta = 5 - 100 = -95
    #   Net Equity Change = 5 - 95 = -90
    
    final_bank_cash = state.bank_cash[0]
    final_bank_equity = state.bank_equity[0]
    
    delta_cash = final_bank_cash - initial_bank_cash
    delta_equity = final_bank_equity - initial_bank_equity
    
    print(f"Cambio Cash Banco: {delta_cash:.2f} (Esperado +15.00)")
    print(f"Cambio Equity Banco: {delta_equity:.2f} (Esperado -90.00)")
    print(f"Bad Debt registrado: {state.bank_bad_debt[0]:.2f} (Esperado 95.00)")
    
    assert np.isclose(delta_cash, 15.0), f"Flujo de caja bancario erróneo: {delta_cash}"
    assert np.isclose(delta_equity, -90.0), f"Cambio patrimonial erróneo: {delta_equity}"
    assert np.isclose(state.bank_bad_debt[0], 95.0), "Registro de bad debt incorrecto"
    
    print("\n✅ Verificación del Paso 5 completada con éxito.")

if __name__ == "__main__":
    verify_step5()
