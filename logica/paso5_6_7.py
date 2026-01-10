import numpy as np
from parametros import Param as p


def paso5(
    # --- Estado Empresas ---
    liquidez_empresas,  # (F,) Liquidez post-salarios (Paso 3)
    ingresos_ventas,  # (F,) Revenue (Paso 4)
    equity_empresas,  # (F,) Patrimonio neto previo
    
    # [Refactor] Relaciones Matriciales FB (F, B)
    pasivos_fb, # Deuda de Empresa F con Banco B
    tasas_fb,   # Tasa de interés pactada entre F y B
    
    # --- Estado Bancos ---
    liquidez_bancos,  # (B,)
    equity_bancos,  # (B,)
    pasivos_interbancarios,  # (B,B) Matriz L_ij (Fila i debe a Col j)
    tasas_interbancarias,  # (B,B) Tasas r_ij
    # --- Estado Hogares ---
    depositos_hogares,  # (H,) Para referencia
    tax_matrix_ib=None, # Matriz de impuestos
):
    """
    Paso 5: Contabilidad, Dividendos y Gestión de Quiebras.
    Refactorizado para relaciones matriciales completas.
    """
    F = p.F
    B = p.B
    
    # Manejo de default para tax_matrix
    if tax_matrix_ib is None:
        tax_matrix_ib = np.zeros((B, B))

    # =========================================================================
    # A. CONTABILIDAD EMPRESAS
    # =========================================================================

    # 1. Calcular Obligaciones de Deuda (Matricial)
    # Intereses = Principal * Tasa
    intereses_matriz = pasivos_fb * tasas_fb
    
    # Amortización = Principal * Tau
    # Ojo: El paper dice "repay tau percent of their outstanding debt".
    # Asumimos que es sobre el principal.
    amortizacion_matriz = pasivos_fb * p.TAU
    
    pago_total_requerido_matriz = intereses_matriz + amortizacion_matriz
    
    # Pagos totales por empresa (Suma filas)
    pago_total_empresas = np.sum(pago_total_requerido_matriz, axis=1) # Vector (F,)
    
    # 2. Actualizar Liquidez
    # Liquidez = (Liquidez prev + Ventas) - Pagos deuda
    liquidez_final_empresas = liquidez_empresas + ingresos_ventas - pago_total_empresas
    
    # 3. Calcular Beneficio y Equity Pre-Dividendo
    # Método Stock-Flow: Equity_new = Liquidez_final - Deuda_Pendiente
    deuda_remanente_empresas = pasivos_fb - amortizacion_matriz  # (Matrix F, B)
    
    # Total vector for accounting
    deuda_total_remanente = np.sum(deuda_remanente_empresas, axis=1)

    # Patrimonio Neto Actualizado
    equity_post_operaciones = liquidez_final_empresas - deuda_total_remanente
    
    beneficio_empresas = equity_post_operaciones - equity_empresas

    # 4. Detectar Quiebras (Illiquidity & Insolvency)
    # "Firms go bankrupt if they have negative liquidity" [cite: 219]
    # (También si equity < 0, técnicamente insolvente, aunque el modelo prioriza liquidez)
    mask_quiebra_empresas = (liquidez_final_empresas < 0) | (
        equity_post_operaciones < 0
    )

    # 5. Ejecutar Quiebras de Empresas -> Impacto en Bancos
    perdidas_bancos_por_empresas = np.zeros(B)
    
    # Logic is handled by matrix write-off below, but we calculate losses here for reporting
    perdidas_matriz = deuda_remanente_empresas.copy()
    perdidas_matriz[~mask_quiebra_empresas, :] = 0.0
    perdidas_bancos_por_empresas = np.sum(perdidas_matriz, axis=0)
    
    # No need for manual loop over indices_quiebra_F since we used matrix operations


    # =========================================================================
    # B. CONTABILIDAD BANCOS (PRE-CASCADA)
    # =========================================================================

    # 1. Cobros de Empresas (Sanas)
    # Los bancos cobran 'pago_total_requerido' de filas NO quebradas.
    cobros_matriz = pago_total_requerido_matriz.copy()
    cobros_matriz[mask_quiebra_empresas, :] = 0.0 # Los quebrados no pagan (o pagan con liquidacion, aqui simplificado a 0)
    
    cobros_empresas = np.sum(cobros_matriz, axis=0) # Vector (B,)
    
    # Intereses Ganados de Empresas (Realmente cobrados)
    intereses_cobrados_matriz = intereses_matriz.copy()
    intereses_cobrados_matriz[mask_quiebra_empresas, :] = 0.0
    intereses_cobrados_reales = np.sum(intereses_cobrados_matriz, axis=0)

    # 2. Pagos Interbancarios (Salidas)
    # Deuda IB Total = L_ij * (1 + r_ij) (Esto asume deuda con interes capitalizado?)
    # El código previo hacia: pasivos * (1+tasa).
    # Coherencia con FB: Intereses + Amortizacion.
    # Si pasivos_ibs es PRINCIPAL:
    intereses_ib = pasivos_interbancarios * tasas_interbancarias
    amortizacion_ib = pasivos_interbancarios * p.TAU
    pago_ib_matriz = intereses_ib + amortizacion_ib
    
    # Banco i paga (suma filas)
    total_pagar_ib = np.sum(pago_ib_matriz, axis=1)
    
    # Banco j cobra (suma cols)
    total_cobrar_ib = np.sum(pago_ib_matriz, axis=0)
    
    # 3. Liquidez Bancaria
    liquidez_final_bancos = (
        liquidez_bancos + cobros_empresas + total_cobrar_ib - total_pagar_ib
    )
    
    # 4. Equity Bancario
    # Profit = (Intereses Empresas + Intereses IB Ganados - Intereses IB Pagados) - Perdidas Credito - Taxes
    
    # Impuestos IB (Revenue Fiscal)
    # Tax Rate matrix applied to Principal
    revenue_fiscal_ib = pasivos_interbancarios * tax_matrix_ib
    total_revenue_fiscal = np.sum(revenue_fiscal_ib)
    
    # Intereses IB Ganados (Lender) = Base Interest (Total - Tax)
    # Asumimos que tasa total incluía tax.
    ganancia_intereses_lender_ib = intereses_ib - revenue_fiscal_ib
    
    intereses_ib_ganados = np.sum(ganancia_intereses_lender_ib, axis=0) # Sum col
    intereses_ib_pagados = np.sum(intereses_ib, axis=1) # Sum row
    
    beneficio_operativo = (
        intereses_cobrados_reales + intereses_ib_ganados - intereses_ib_pagados
    )
    
    equity_bancos_actual = (
        equity_bancos + beneficio_operativo - perdidas_bancos_por_empresas
    )

    # =========================================================================
    # C. CASCADA DE QUIEBRAS BANCARIAS (Optimizado)
    # =========================================================================
    # "Iterative default-event unfolds" [cite: 236]

    mask_quiebra_bancos = (equity_bancos_actual < 0) | (liquidez_final_bancos < 0)

    # Cola de procesamiento: Solo los que acaban de caer y no han sido procesados
    cola_para_procesar = np.where(mask_quiebra_bancos)[0].tolist()

    # Matriz de exposición principal remanente (Principal pendiente)
    # Lo que queda por cobrar tras el pago de cuota tau (que se asume pagada si había liquidez, 
    # o si no había, la deuda entera es lo que cuenta para contagio? 
    # Paper: "remaining interbank debt".
    # Asumimos que la amortización de este turno se descontó contablemente.
    # Remanente = Principal * (1 - TAU)
    matriz_ib_remanente = pasivos_interbancarios * (1.0 - p.TAU)
    
    total_perdidas_contagio = 0.0

    while cola_para_procesar:
        # Extraemos el lote actual de quebrados
        bancos_fallidos_ronda = cola_para_procesar
        cola_para_procesar = []  # Preparamos la siguiente ronda vacía

        for b_dead in bancos_fallidos_ronda:
            # Identificar a quién debe dinero el muerto (sus acreedores/víctimas)
            # Acreedor tiene activo > 0 contra b_dead
            # (Fila b_dead de matriz remanente tiene lo que b_dead LE DEBE a otros? NO)
            
            # CORRECCION IMPORTANTE LOGICA MATRICES:
            # pasivos_interbancarios[i, j] -> i debe a j.
            # matriz_ib_remanente[i, j] -> monto que i le quedó debiendo a j.
            # SI 'i' (=b_dead) muere, 'j' (=acreedor) sufre la pérdida.
            
            # Buscamos columnas j donde matriz[b_dead, j] > 0
            acreedores = np.where(matriz_ib_remanente[b_dead, :] > 0)[0]

            for acreedor in acreedores:
                if mask_quiebra_bancos[acreedor]:
                    continue  # Ya está muerto, no importa golpearlo de nuevo

                # Contagio: El acreedor pierde todo el activo
                loss = matriz_ib_remanente[b_dead, acreedor]
                equity_bancos_actual[acreedor] -= loss
                total_perdidas_contagio += loss

                # Quemamos el enlace para no procesarlo nunca más
                matriz_ib_remanente[b_dead, acreedor] = 0.0

                # Chequeo de insolvencia inducida
                if equity_bancos_actual[acreedor] < 0:
                    mask_quiebra_bancos[acreedor] = True
                    # Añadimos a la cola para que SU caída propague en la sig. ronda
                    cola_para_procesar.append(acreedor)

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
        # Nuevas métricas para plots
        np.sum(mask_quiebra_bancos), # Total quiebras bancos
        total_perdidas_contagio # Total dinero perdido por contagio
    )
