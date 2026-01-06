import numpy as np
from parametros import Parametros


def ejecutar_paso7(modelo):
    """
    Paso 7: Cierre Bancario y Gestión de Liquidez Imprevista.
    - Cascadas de default bancario.
    - Bailout / Reestructuración de bancos.
    """

    # 2. Bank Default Cascades
    processed_mask = np.zeros(Parametros.B, dtype=bool)

    while True:
        current_equity = modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
        dead_mask = current_equity < 0
        new_defaults = dead_mask & (~processed_mask)
        new_default_ids = np.where(new_defaults)[0]

        if len(new_default_ids) == 0:
            break

        for dead_bank in new_default_ids:
            obligations = modelo.matriz_interbancaria[dead_bank, :]
            modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= obligations
            loss_val = np.sum(obligations)
            modelo.current_step_loss += loss_val
            modelo.matriz_interbancaria[dead_bank, :] = 0.0

        modelo.current_step_defaults += len(new_default_ids)
        processed_mask[new_default_ids] = True

    # 3. Bailout / Reset
    all_dead_ids = np.where(processed_mask)[0]
    if len(all_dead_ids) > 0:
        n_dead_b = len(all_dead_ids)
        init_assets = modelo.rng.uniform(
            Parametros.INIT_BANK_ASSETS[0],
            Parametros.INIT_BANK_ASSETS[1],
            size=n_dead_b,
        )
        modelo.estado_bancos[all_dead_ids, Parametros.IDX_BANK_TOTAL_ASSETS] = (
            init_assets
        )
        modelo.estado_bancos[all_dead_ids, Parametros.IDX_BANK_EQUITY] = (
            init_assets * Parametros.INIT_CAPITAL_RATIO
        )
        modelo.estado_bancos[all_dead_ids, Parametros.IDX_BANK_LIQUIDITY] = init_assets
        modelo.estado_bancos[all_dead_ids, Parametros.IDX_BANK_DEPOSITS] = (
            init_assets * (1 - Parametros.INIT_CAPITAL_RATIO)
        )
        modelo.estado_bancos[all_dead_ids, Parametros.IDX_BANK_BAD_DEBT] = 0.0

        modelo.matriz_interbancaria[:, all_dead_ids] = 0.0
        modelo.matriz_credito_firmas[:, all_dead_ids] = 0.0
