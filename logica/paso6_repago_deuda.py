import numpy as np
from parametros import Parametros


def ejecutar_paso6(modelo):
    """
    Paso 6: Repago de Deudas.
    - Firmas y Bancos pagan deudas (Principal + Intereses).
    - Actualización de beneficios (restar intereses pagados, sumar cobrados).
    """

    tau = Parametros.DEBT_REPAYMENT_RATE

    # 1. Firms -> Banks
    # Principal Repayment
    principal_repayment_firms = modelo.matriz_credito_firmas * tau

    # Interest Payment
    # Interest = Debt * Rate
    interest_payment_firms = modelo.matriz_credito_firmas * modelo.matriz_tasas_firmas

    total_payment_firms_matrix = principal_repayment_firms + interest_payment_firms

    # Aggregates
    total_pay_firm_vec = np.sum(total_payment_firms_matrix, axis=1)  # (F,)
    total_receive_bank_vec = np.sum(total_payment_firms_matrix, axis=0)  # (B,)

    # Execute Payment (Liquidity)
    modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= total_pay_firm_vec
    modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] += total_receive_bank_vec

    # Update Debt Principal
    modelo.matriz_credito_firmas -= principal_repayment_firms

    # Update Profits
    modelo.current_step_profit_firms -= np.sum(interest_payment_firms, axis=1)
    modelo.current_step_profit_bancos += np.sum(interest_payment_firms, axis=0)

    # 2. Interbank
    principal_repayment_ib = modelo.matriz_interbancaria * tau
    interest_payment_ib = (
        modelo.matriz_interbancaria * modelo.matriz_tasas_interbancaria
    )

    total_payment_ib_matrix = principal_repayment_ib + interest_payment_ib

    total_pay_bank_ib_vec = np.sum(total_payment_ib_matrix, axis=1)
    total_receive_bank_ib_vec = np.sum(total_payment_ib_matrix, axis=0)

    modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] -= total_pay_bank_ib_vec
    modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] += total_receive_bank_ib_vec

    modelo.matriz_interbancaria -= principal_repayment_ib

    # Update Profits (Banks)
    # Expenses: Interest paid to other banks
    modelo.current_step_profit_bancos -= np.sum(interest_payment_ib, axis=1)
    # Revenues: Interest received from other banks
    modelo.current_step_profit_bancos += np.sum(interest_payment_ib, axis=0)
