import numpy as np
from parametros import Parametros


def ejecutar_paso5(modelo):
    """
    Paso 5: Dividendos y Quiebras de Firmas.
    - Pago de dividendos (Firmas y Bancos).
    - Quiebra de firmas con liquidez negativa.
    """

    # --- B. DIVIDENDS (Based on PROFITS) ---
    # 1. Firms
    # Profit = Revenue - Wages - Interest
    # Dividends = max(0, Profit) * Ratio
    distributable_f = np.maximum(0, modelo.current_step_profit_firms)
    dividends_f = distributable_f * Parametros.DIVIDEND_RATIO

    # Cap dividends at available liquidity to avoid immediate bankruptcy from payout
    firm_liq = modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]
    dividends_f = np.minimum(dividends_f, np.maximum(0, firm_liq))

    modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= dividends_f

    # Distribute to Owners using MATRIX (H x F)
    # HH Income = Matriz_Propiedad @ Dividends (F,)
    hh_div_income_f = modelo.matriz_propiedad_firmas @ dividends_f
    modelo.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_div_income_f

    # 2. Banks
    # Profit = Interest(Firms) + Interest(IB_In) - Interest(IB_Out)
    distributable_b = np.maximum(0, modelo.current_step_profit_bancos)
    dividends_b = distributable_b * Parametros.DIVIDEND_RATIO

    bank_liq = modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
    dividends_b = np.minimum(dividends_b, np.maximum(0, bank_liq))

    modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] -= dividends_b
    modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= dividends_b

    # Distribute using MATRIX (H x B)
    hh_div_income_b = modelo.matriz_propiedad_bancos @ dividends_b
    modelo.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_div_income_b

    # --- C. FIRM BANKRUPTCIES ---

    dead_firms_mask = modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] < 0
    dead_firms_indices = np.where(dead_firms_mask)[0]

    if len(dead_firms_indices) > 0:
        bad_loans = modelo.matriz_credito_firmas[dead_firms_indices, :]
        bank_losses = np.sum(bad_loans, axis=0)

        # Track Systemic Loss
        modelo.current_step_loss += np.sum(bank_losses)

        modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= bank_losses
        modelo.estado_bancos[:, Parametros.IDX_BANK_BAD_DEBT] += bank_losses

        modelo.matriz_credito_firmas[dead_firms_indices, :] = 0.0

        # Reset Firms
        n_dead = len(dead_firms_indices)
        init_liq = modelo.rng.uniform(
            Parametros.INIT_FIRM_ASSETS[0],
            Parametros.INIT_FIRM_ASSETS[1],
            size=n_dead,
        )
        init_price = Parametros.WAGE / Parametros.alpha

        modelo.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_LIQUIDITY] = (
            init_liq
        )
        modelo.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_EQUITY] = init_liq
        modelo.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_PRICE] = init_price
        modelo.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_LEVERAGE] = 0.0
        modelo.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_DEMAND] = 0.0

    # --- D. BANK BANKRUPTCIES (CASCADE) ---

    processed_defaults = set()

    while True:
        # Find banks with negative equity that haven't been processed
        equity = modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
        current_defaults = np.where(
            (equity < 0) & (~np.isin(np.arange(Parametros.B), list(processed_defaults)))
        )[0]

        if len(current_defaults) == 0:
            break

        for def_bank in current_defaults:
            processed_defaults.add(def_bank)
            modelo.current_step_defaults += 1

            # Find creditors (Banks that lent TO def_bank)
            # Row = Borrower, Col = Lender. So look at row def_bank.
            liabilities = modelo.matriz_interbancaria[def_bank, :]
            lenders = np.where(liabilities > 0)[0]

            for lender in lenders:
                loss = liabilities[lender]
                # Creditor takes the hit
                modelo.estado_bancos[lender, Parametros.IDX_BANK_EQUITY] -= loss
                modelo.estado_bancos[lender, Parametros.IDX_BANK_BAD_DEBT] += loss

                # Track Systemic Loss
                modelo.current_step_loss += loss

            # Wipe the debt (defaulted)
            modelo.matriz_interbancaria[def_bank, :] = 0.0

    # Reincarnate Dead Banks
    if len(processed_defaults) > 0:
        dead_banks = np.array(list(processed_defaults))
        n_dead = len(dead_banks)

        # Clear their lending (assets) and firm loans
        modelo.matriz_interbancaria[:, dead_banks] = 0.0
        modelo.matriz_credito_firmas[:, dead_banks] = 0.0

        # Reset State
        init_bank_assets = modelo.rng.uniform(
            Parametros.INIT_BANK_ASSETS[0], Parametros.INIT_BANK_ASSETS[1], size=n_dead
        )

        modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_TOTAL_ASSETS] = (
            init_bank_assets
        )
        modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_EQUITY] = (
            init_bank_assets * Parametros.INIT_CAPITAL_RATIO
        )
        modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_LIQUIDITY] = (
            init_bank_assets
        )
        modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_DEPOSITS] = (
            init_bank_assets
            - modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_EQUITY]
        )

        # Reset behaviors
        modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_OPERATING_COST_CHI] = (
            modelo.rng.uniform(
                Parametros.CHI_RANGE[0], Parametros.CHI_RANGE[1], size=n_dead
            )
        )
        modelo.estado_bancos[dead_banks, Parametros.IDX_BANK_INTERBANK_COST_PSI] = (
            modelo.rng.uniform(
                Parametros.PSI_RANGE[0], Parametros.PSI_RANGE[1], size=n_dead
            )
        )
