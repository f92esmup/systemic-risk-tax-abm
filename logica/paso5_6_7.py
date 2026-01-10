import numpy as np
from parametros import Param as p


def paso5(
    # --- Estado Empresas ---
    liquidez_empresas,  # (F,) Liquidez post-salarios (Paso 3)
    ingresos_ventas,  # (F,) Revenue (Paso 4)
    deuda_empresas,  # (F,) Principal acumulado
    tasa_empresas,  # (F,) Tasa de interés pactada
    equity_empresas,  # (F,) Patrimonio neto previo
    banco_acreedor_empresa,  # (F,) Índice del banco prestamista
    # --- Estado Bancos ---
    liquidez_bancos,  # (B,)
    equity_bancos,  # (B,)
    pasivos_interbancarios,  # (B,B) Matriz L_ij (Fila i debe a Col j)
    tasas_interbancarias,  # (B,B) Tasas r_ij
    # --- Estado Hogares ---
    depositos_hogares,  # (H,) Para referencia (no se modifica aquí, se devuelve dividendo per capita)
):
    """
    Paso 5: Contabilidad, Dividendos y Gestión de Quiebras (Cascadas).

    1. Empresas pagan deuda parcial (tau) y cobran ventas.
    2. Chequeo de quiebra de empresas -> Impacto en Bancos (Write-off).
    3. Bancos pagan deuda interbancaria parcial (tau).
    4. Chequeo de quiebra de bancos -> Cascada de impagos Interbancarios.
    5. Pago de dividendos (agentes sanos).
    6. Reset de agentes quebrados (Renacimiento).

    Ref: [cite: 193, 197, 227, 234-239]
    """
    F = p.F
    B = p.B

    # =========================================================================
    # A. CONTABILIDAD EMPRESAS
    # =========================================================================

    # 1. Calcular Obligaciones de Deuda (Intereses + Principal)
    # Ref: "repay tau percent of their outstanding debt"
    # Deuda Total = Principal * (1 + tasa)
    # (Asumiendo que tasa es por periodo, o ajustada)
    deuda_total_con_intereses = deuda_empresas * (1 + tasa_empresas)
    pago_requerido = deuda_total_con_intereses * p.TAU

    # Intereses devengados (solo la parte de interés)
    intereses_empresas = deuda_empresas * tasa_empresas

    # 2. Actualizar Liquidez (Cash Flow)
    # Liquidez Final = (Liquidez Inicial - Salarios) + Ventas - Pago Deuda
    # Nota: liquidez_empresas input ya descontó salarios en Paso 3.
    liquidez_final_empresas = liquidez_empresas + ingresos_ventas - pago_requerido

    # 3. Calcular Beneficio y Equity Pre-Dividendo
    # Método Stock-Flow: Equity_new = Liquidez_final - Deuda_Pendiente
    deuda_remanente_empresas = deuda_total_con_intereses - pago_requerido

    # Patrimonio Neto Actualizado
    equity_post_operaciones = liquidez_final_empresas - deuda_remanente_empresas

    # Beneficio del periodo (Cambio en Equity)
    beneficio_empresas = equity_post_operaciones - equity_empresas

    # 4. Detectar Quiebras (Illiquidity & Insolvency)
    # "Firms go bankrupt if they have negative liquidity" [cite: 219]
    # (También si equity < 0, técnicamente insolvente, aunque el modelo prioriza liquidez)
    mask_quiebra_empresas = (liquidez_final_empresas < 0) | (
        equity_post_operaciones < 0
    )

    # 5. Ejecutar Quiebras de Empresas -> Impacto en Bancos
    perdidas_bancos_por_empresas = np.zeros(B)

    indices_quiebra_F = np.where(mask_quiebra_empresas)[0]

    for f in indices_quiebra_F:
        b = int(banco_acreedor_empresa[f])
        # El banco pierde la deuda remanente que esperaba cobrar a futuro.
        # "Write off... as defaulted credits" [cite: 234]
        # Asumimos recuperación cero.
        monto_perdido = deuda_remanente_empresas[f]
        if monto_perdido > 0:
            perdidas_bancos_por_empresas[b] += monto_perdido

    # =========================================================================
    # B. CONTABILIDAD BANCOS (PRE-CASCADA)
    # =========================================================================

    # 1. Cobros de Empresas (Sanas)
    # Los pagos de las quebradas ya se "perdieron" o se cobró lo que se pudo en liquidez.
    # Simplificación: El banco cobra 'pago_requerido' solo de las empresas NO quebradas.
    cobros_empresas = np.zeros(B)
    np.add.at(
        cobros_empresas,
        banco_acreedor_empresa[~mask_quiebra_empresas].astype(int),
        pago_requerido[~mask_quiebra_empresas],
    )

    # 2. Pagos Interbancarios (Salidas)
    # Deuda IB Total = L_ij * (1 + r_ij)
    matriz_deuda_ib_total = pasivos_interbancarios * (1 + tasas_interbancarias)
    pago_ib_requerido_matriz = matriz_deuda_ib_total * p.TAU

    total_pagar_ib = np.sum(pago_ib_requerido_matriz, axis=1)  # Banco i paga
    total_cobrar_ib = np.sum(pago_ib_requerido_matriz, axis=0)  # Banco i cobra

    # 3. Liquidez Bancaria Pre-Dividendo
    liquidez_final_bancos = (
        liquidez_bancos + cobros_empresas + total_cobrar_ib - total_pagar_ib
    )

    # 4. Equity Bancario Pre-Cascada
    # Equity_new = Equity_old + (Intereses Ganados - Intereses Pagados) - Perdidas_Empresas

    # Intereses Ganados de Empresas (Solo de las vivas)
    intereses_cobrados_reales = np.zeros(B)
    intereses_potenciales = intereses_empresas.copy()
    intereses_potenciales[mask_quiebra_empresas] = 0  # No se cobra interés de muertos
    np.add.at(
        intereses_cobrados_reales,
        banco_acreedor_empresa.astype(int),
        intereses_potenciales,
    )

    # Intereses IB
    intereses_ib_ganados = np.sum(pasivos_interbancarios * tasas_interbancarias, axis=0)
    intereses_ib_pagados = np.sum(pasivos_interbancarios * tasas_interbancarias, axis=1)

    beneficio_operativo = (
        intereses_cobrados_reales + intereses_ib_ganados - intereses_ib_pagados
    )

    equity_bancos_actual = (
        equity_bancos + beneficio_operativo - perdidas_bancos_por_empresas
    )

    # =========================================================================
    # C. CASCADA DE QUIEBRAS BANCARIAS
    # =========================================================================
    # "Iterative default-event unfolds" [cite: 236]

    mask_quiebra_bancos = (equity_bancos_actual < 0) | (liquidez_final_bancos < 0)
    lista_quebrados = np.where(mask_quiebra_bancos)[0].tolist()

    # Matriz de exposición principal remanente (Principal pendiente)
    # Lo que queda por cobrar tras el pago de cuota tau
    matriz_ib_remanente = matriz_deuda_ib_total - pago_ib_requerido_matriz

    nuevos_defaults = True
    while nuevos_defaults:
        nuevos_defaults = False

        # Procesar impacto de los quebrados en sus acreedores
        # Nota: En una cascada real, procesamos solo los NUEVOS quebrados en cada ronda.
        # Aquí iteramos sobre todos los quebrados, pero chequeamos si el acreedor ya murió.

        for b_dead in lista_quebrados:
            # Quién le prestó a b_dead? (b_dead debe a 'acreedor')
            # Acreedor tiene un activo en la columna 'acreedor', fila 'b_dead'
            acreedores = np.where(matriz_ib_remanente[b_dead, :] > 0)[0]

            for acreedor in acreedores:
                if mask_quiebra_bancos[acreedor]:
                    continue  # Ya está muerto, no le afecta más

                # Pérdida por contagio (Write-off total interbancario)
                loss = matriz_ib_remanente[b_dead, acreedor]

                # Impacto en Equity del acreedor
                equity_bancos_actual[acreedor] -= loss

                # Borrar la deuda para no contabilizarla dos veces
                matriz_ib_remanente[b_dead, acreedor] = 0.0

                # Chequear solvencia tras el golpe
                if equity_bancos_actual[acreedor] < 0:
                    mask_quiebra_bancos[acreedor] = True
                    lista_quebrados.append(acreedor)
                    nuevos_defaults = True

    # =========================================================================
    # D. DIVIDENDOS Y RENACIMIENTO
    # =========================================================================

    # --- Dividendos Empresas ---
    div_empresas = np.zeros(F)
    # Pagan si están vivas y tuvieron beneficio positivo
    mask_paga_F = (~mask_quiebra_empresas) & (beneficio_empresas > 0)
    div_empresas[mask_paga_F] = beneficio_empresas[mask_paga_F] * p.DIVIDEND_RATIO

    liquidez_final_empresas[mask_paga_F] -= div_empresas[mask_paga_F]
    equity_post_operaciones[mask_paga_F] -= div_empresas[mask_paga_F]

    # --- Dividendos Bancos ---
    div_bancos = np.zeros(B)
    mask_paga_B = (~mask_quiebra_bancos) & (equity_bancos_actual > equity_bancos)
    # Pagan sobre el incremento de equity
    delta_equity = equity_bancos_actual - equity_bancos
    div_bancos[mask_paga_B] = delta_equity[mask_paga_B] * p.DIVIDEND_RATIO

    liquidez_final_bancos[mask_paga_B] -= div_bancos[mask_paga_B]
    equity_bancos_actual[mask_paga_B] -= div_bancos[mask_paga_B]

    # Transferir a Hogares
    total_div = np.sum(div_empresas) + np.sum(div_bancos)
    dividendos_per_capita = total_div / len(depositos_hogares)

    # --- RESET / RENACIMIENTO ---

    # Empresas Quebradas: Renacen "limpias" [cite: 200]
    if np.any(mask_quiebra_empresas):
        # Asignar capital semilla promedio de las sobrevivientes
        # (Para que no nazcan muertas)
        equity_medio = np.mean(equity_post_operaciones[~mask_quiebra_empresas])
        if np.isnan(equity_medio) or equity_medio <= 0:
            equity_medio = 1.0  # Fallback

        deuda_remanente_empresas[mask_quiebra_empresas] = 0.0
        equity_post_operaciones[mask_quiebra_empresas] = equity_medio
        liquidez_final_empresas[mask_quiebra_empresas] = equity_medio  # Cash inicial

        # El banco acreedor pierde la referencia a la deuda vieja (ya hizo write-off)
        # En el próximo paso 1/2, la empresa 'renacida' buscará banco nuevo.

    # Bancos Quebrados: Recapitalización Forzosa (para continuar simulación)
    if np.any(mask_quiebra_bancos):
        equity_medio_B = np.mean(equity_bancos_actual[~mask_quiebra_bancos])
        if np.isnan(equity_medio_B) or equity_medio_B <= 0:
            equity_medio_B = p.EQUITY_INICIAL_BANCOS

        equity_bancos_actual[mask_quiebra_bancos] = equity_medio_B
        liquidez_final_bancos[mask_quiebra_bancos] = equity_medio_B

        # Saneamiento de la red Interbancaria (Reset total de sus enlaces)
        pasivos_interbancarios[mask_quiebra_bancos, :] = 0  # No deben nada
        pasivos_interbancarios[:, mask_quiebra_bancos] = (
            0  # Nadie les debe (write-off asumido)
        )

    return (
        liquidez_final_empresas,
        equity_post_operaciones,
        deuda_remanente_empresas,
        mask_quiebra_empresas,  # Importante para Paso 1
        liquidez_final_bancos,
        equity_bancos_actual,
        mask_quiebra_bancos,  # Importante para estadísticas de riesgo sistémico
        pasivos_interbancarios,  # Matriz saneada
        dividendos_per_capita,  # Importante para Paso 4
    )
