# logica/paso5.py
import numpy as np
from parametros import Param as p


def paso5_resultados_y_quiebras(
    firm_liquidez,
    firm_ingresos,
    firm_coste_salarial,
    firm_costo_financiero_total,
    matriz_prestamos_firmas,  # (F, B) Principal + Intereses
    matriz_intereses_firmas,  # (F, B) Solo Intereses
    bancos_liquidez,
    bancos_patrimonio,
    matriz_interbancaria,
    matriz_intereses_ib,
    hogares_liquidez,
    hogares_indices_duenos_firmas,
    hogares_indices_duenos_bancos,
    fondo_rescate_acumulado,
):
    """
    Paso 5: Quiebras con Responsabilidad Personal, Contagio y Dividendos.
    CORREGIDO: Usa flujos de caja (tau) para cálculo de dividendos.
    """

    # --- 1. QUIEBRA DE EMPRESAS ---
    mascara_quiebra_firmas = firm_liquidez < -1e-5
    indices_quiebra_firmas = np.where(mascara_quiebra_firmas)[0]
    indices_firmas_vivas = np.where(~mascara_quiebra_firmas)[0]

    perdida_por_banco = np.zeros(p.B)

    if len(indices_quiebra_firmas) > 0:
        deuda_impagada = matriz_prestamos_firmas[
            indices_quiebra_firmas, :
        ]  # (N_fail, B)

        # Responsabilidad Personal
        duenos_afectados = hogares_indices_duenos_firmas[indices_quiebra_firmas]
        liquidez_duenos = hogares_liquidez[duenos_afectados]
        total_deuda_firma = np.sum(deuda_impagada, axis=1)

        recuperacion_total_firma = np.minimum(liquidez_duenos, total_deuda_firma)
        hogares_liquidez[duenos_afectados] -= recuperacion_total_firma

        ratio_recuperacion = np.divide(
            recuperacion_total_firma,
            total_deuda_firma,
            out=np.zeros_like(total_deuda_firma),
            where=total_deuda_firma > 1e-9,
        )

        matriz_recuperacion_bancos = deuda_impagada * ratio_recuperacion[:, np.newaxis]
        bancos_liquidez += np.sum(matriz_recuperacion_bancos, axis=0)

        # Pérdida Neta
        matriz_perdidas = deuda_impagada - matriz_recuperacion_bancos
        perdida_por_banco = np.sum(matriz_perdidas, axis=0)
        bancos_patrimonio -= perdida_por_banco

        # Limpieza
        firm_liquidez[indices_quiebra_firmas] = 0
        matriz_prestamos_firmas[indices_quiebra_firmas, :] = 0
        matriz_intereses_firmas[indices_quiebra_firmas, :] = 0

    # --- 2. DIVIDENDOS EMPRESAS (SOBREVIVIENTES) ---
    beneficio_firmas = np.zeros(p.F)

    # CORRECCIÓN FLJO: Gasto financiero es tau * Intereses Totales
    gasto_financiero_periodo = np.sum(matriz_intereses_firmas, axis=1) * p.tau

    beneficio_firmas[indices_firmas_vivas] = (
        firm_ingresos[indices_firmas_vivas]
        - firm_coste_salarial[indices_firmas_vivas]
        - gasto_financiero_periodo[indices_firmas_vivas]
    )

    mascara_dividendos = beneficio_firmas > 1e-5
    dividendos_firmas = np.zeros(p.F)
    dividendos_firmas[mascara_dividendos] = beneficio_firmas[mascara_dividendos] * 0.2

    dividendos_firmas = np.minimum(dividendos_firmas, np.maximum(firm_liquidez, 0))
    firm_liquidez -= dividendos_firmas

    if len(hogares_indices_duenos_firmas) > 0:
        hogares_liquidez[hogares_indices_duenos_firmas] += dividendos_firmas

    # --- 3. QUIEBRA DE BANCOS Y CONTAGIO ---
    bancos_activos = np.ones(p.B, dtype=bool)
    cola_quiebras = []

    nuevos_caidos = np.where((bancos_patrimonio < -1e-5) & bancos_activos)[0]
    for b in nuevos_caidos:
        bancos_activos[b] = False
        cola_quiebras.append(b)

    while len(cola_quiebras) > 0:
        banco_fallido = cola_quiebras.pop(0)
        pasivo_fallido = matriz_interbancaria[banco_fallido, :]
        indices_acreedores = np.where(pasivo_fallido > 1e-5)[0]

        if len(indices_acreedores) > 0:
            montos_perdidos = pasivo_fallido[indices_acreedores]
            bancos_patrimonio[indices_acreedores] -= montos_perdidos

            matriz_interbancaria[banco_fallido, indices_acreedores] = 0
            matriz_intereses_ib[banco_fallido, indices_acreedores] = 0

            for b_ac in indices_acreedores:
                if bancos_activos[b_ac] and bancos_patrimonio[b_ac] < -1e-5:
                    bancos_activos[b_ac] = False
                    cola_quiebras.append(b_ac)

    # --- 4. DIVIDENDOS BANCOS (SOBREVIVIENTES) ---
    # CORRECCIÓN FLUJO: Usamos p.tau para estimar ingreso/gasto del periodo
    ingresos_intereses_firmas = np.sum(matriz_intereses_firmas, axis=0) * p.tau
    ingresos_intereses_ib = np.sum(matriz_intereses_ib, axis=0) * p.tau
    gastos_intereses_ib = np.sum(matriz_intereses_ib, axis=1) * p.tau

    beneficio_operativo = (
        ingresos_intereses_firmas + ingresos_intereses_ib - gastos_intereses_ib
    )
    beneficio_neto_bancos = beneficio_operativo - perdida_por_banco

    mascara_dividendos_bancos = bancos_activos & (beneficio_neto_bancos > 1e-5)
    indices_bancos_div = np.where(mascara_dividendos_bancos)[0]

    dividendos_bancos = np.zeros(p.B)
    dividendos_bancos[indices_bancos_div] = (
        beneficio_neto_bancos[indices_bancos_div] * 0.2
    )

    dividendos_bancos = np.minimum(dividendos_bancos, np.maximum(bancos_liquidez, 0))
    bancos_liquidez -= dividendos_bancos

    if len(hogares_indices_duenos_bancos) > 0:
        hogares_liquidez[hogares_indices_duenos_bancos] += dividendos_bancos

    return (
        firm_liquidez,
        bancos_patrimonio,
        bancos_liquidez,
        bancos_activos,
        matriz_prestamos_firmas,
        matriz_interbancaria,
        hogares_liquidez,
        len(indices_quiebra_firmas),
        np.sum(~bancos_activos),
    )
