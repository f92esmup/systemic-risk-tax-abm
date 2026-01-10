# logica/paso2.py
import numpy as np
from parametros import Param as p
from logica.debtrank import calcular_impacto_sr


def paso2_ejecutar(
    # Inputs del Paso 1
    factura_esperada_salarial,
    # Estado de Empresas
    firm_ids,
    firm_liquidez,
    firm_deuda,
    prestamos_activos_empresas,  # Lista que se actualizará in-place
    # Estado de Bancos
    bancos_ids,
    bancos_liquidez,
    bancos_patrimonio,
    bancos_depositos,
    bancos_deuda_acumulada,
    # Estado del Sistema
    matriz_interbancaria,  # Matriz (B, B) que se actualizará
    tax_mode=p.TAX_MODE,
):
    """
    Orquesta el Paso 2 completo:
    1. Empresas calculan déficit y piden crédito.
    2. Bancos buscan liquidez en mercado interbancario (con impuestos SR).
    3. Se cierran contratos y SE ACTUALIZAN LOS BALANCES (Liquidez, Deuda, Matriz IB).

    Argumentos:
        factura_esperada_salarial (np.array): Output del Paso 1.
        [Los demás son vectores de estado del sistema]

    Retorna:
        No necesita retornar valores complejos, actualiza los arrays de numpy y listas in-place.
        (Opcionalmente devuelve estadísticas para logs).
    """

    # --- FASE A: Mercado de Crédito (Solicitud Preliminar) ---

    # 1. Identificar Demanda
    # Gap = Lo que tengo que pagar - Lo que tengo en caja
    credito_necesario = np.maximum(factura_esperada_salarial - firm_liquidez, 0)
    firms_necesitan_credito = firm_ids[credito_necesario > 1e-5]

    contratos_pendientes = []  # [firm_id, banco_id, monto, tasa_base]

    if len(firms_necesitan_credito) > 0:
        # Vectorización de fragilidad
        # Epsilon para evitar div por cero
        ratio_fragilidad = firm_deuda / (firm_liquidez + 1e-9)
        mu_empresas = np.tanh(ratio_fragilidad)

        # Especificidad aleatoria de bancos (chi)
        chi_bancos = np.random.uniform(0, 1, p.B)

        for f_id in firms_necesitan_credito:
            monto = credito_necesario[f_id]

            # Selección aleatoria de N bancos
            bancos_candidatos = np.random.choice(
                bancos_ids, size=p.n_bancos, replace=False
            )

            # Tasa ofrecida inicial (Ec. A1 del paper)
            tasas_oferta = p.r_bar * (
                1 + chi_bancos[bancos_candidatos] * mu_empresas[f_id]
            )

            # Elegir mejor oferta
            idx_mejor = np.argmin(tasas_oferta)
            mejor_banco = bancos_candidatos[idx_mejor]
            mejor_tasa_base = tasas_oferta[idx_mejor]

            # Racionamiento de crédito (Credit Rationing) si la tasa es muy alta
            monto_final = monto
            if mejor_tasa_base > p.r_max:
                monto_final = monto * p.phi

            contratos_pendientes.append(
                {
                    "firm": f_id,
                    "banco": mejor_banco,
                    "monto": monto_final,
                    "tasa_base": mejor_tasa_base,
                }
            )

    # --- FASE B: Mercado Interbancario (Gestión de Liquidez Bancaria) ---

    # 1. Calcular Liquidez Neta por Banco
    demanda_por_banco = np.zeros(p.B)
    mapa_solicitudes = {b: [] for b in bancos_ids}

    for c in contratos_pendientes:
        b_id = c["banco"]
        demanda_por_banco[b_id] += c["monto"]
        mapa_solicitudes[b_id].append(c)

    # Liquidez disponible antes de prestar
    balance_liquidez = bancos_liquidez - demanda_por_banco

    deficit_ids = bancos_ids[balance_liquidez < -1e-5]
    superavit_ids = bancos_ids[balance_liquidez > 1e-5]

    # Variables para cálculo de riesgo (DebtRank inputs)
    # Importante: Usamos una copia de la matriz IB para simular impactos sin corromper la actual
    matriz_ib_simulacion = matriz_interbancaria.copy()

    pasivo_total = bancos_depositos + bancos_deuda_acumulada
    patrimonio_seguro = np.maximum(bancos_patrimonio, 1e-9)
    mu_bancos = np.tanh(pasivo_total / patrimonio_seguro)
    psi_bancos = np.random.uniform(0, p.psi_max, p.B)  # Especificidad IB

    # Estructuras para "Pass-through" de costes
    coste_refinanciacion = np.zeros(p.B)  # Intereses totales a pagar por el banco
    monto_refinanciado = np.zeros(p.B)  # Principal total levantado

    # Mezclar aleatoriamente los deficitarios para no dar prioridad por ID
    np.random.shuffle(deficit_ids)

    # 2. Matching Interbancario
    for b_deudor in deficit_ids:
        necesidad = abs(balance_liquidez[b_deudor])
        if necesidad < 1e-5:
            continue

        # Buscar ofertas
        ofertas = []
        for b_acreedor in superavit_ids:
            disponible = bancos_liquidez[
                b_acreedor
            ]  # Usamos el array real que iremos actualizando
            if disponible < 1e-5:
                continue

            # Tasa Base Interbancaria (Ec. A2)
            r_ib = p.r_bar * (1 + psi_bancos[b_acreedor] * mu_bancos[b_deudor])

            # Cálculo de Impuesto (SRT / Tobin)
            tax_rate = 0.0

            if tax_mode == "SRT":
                # Simulamos el préstamo para ver cuánto sube el Riesgo Sistémico
                monto_sim = min(disponible, necesidad)
                if monto_sim > 0:
                    delta_el = calcular_impacto_sr(
                        b_deudor,
                        b_acreedor,
                        monto_sim,
                        matriz_ib_simulacion,
                        bancos_patrimonio,
                        bancos_depositos,
                        bancos_deuda_acumulada,
                    )
                    # Tasa impositiva = (Zeta * ImpactoEuros) / Monto
                    tax_rate = (p.ZETA * delta_el) / monto_sim

            elif tax_mode == "TOBIN":
                tax_rate = p.TOBIN_RATE

            tasa_total = r_ib + tax_rate
            ofertas.append(
                {"acreedor": b_acreedor, "rate": tasa_total, "disponible": disponible}
            )

        # Ordenar ofertas por tasa (Best Execution)
        ofertas.sort(key=lambda x: x["rate"])

        # Ejecutar préstamos IB
        for of in ofertas:
            if necesidad < 1e-5:
                break

            acreedor = of["acreedor"]
            # Verificar disponibilidad real (puede haber cambiado en el loop anterior)
            disponible_real = bancos_liquidez[acreedor]

            monto_real = min(necesidad, disponible_real)
            if monto_real < 1e-9:
                continue

            # --- ACTUALIZACIÓN DE ESTADO 1: Mercado Interbancario ---
            # 1. Transferencia de Liquidez
            bancos_liquidez[acreedor] -= monto_real
            bancos_liquidez[b_deudor] += monto_real

            # 2. Registro de Deuda en Matriz
            matriz_interbancaria[b_deudor, acreedor] += monto_real
            # También actualizamos la de simulación por si otro banco pide después
            matriz_ib_simulacion[b_deudor, acreedor] += monto_real

            # 3. Acumular Costes para Pass-through
            interes_operacion = monto_real * of["rate"]
            coste_refinanciacion[b_deudor] += interes_operacion
            monto_refinanciado[b_deudor] += monto_real

            necesidad -= monto_real

    # --- FASE C: Cierre de Préstamos a Empresas (Pass-through y Actualización) ---

    for b_id in bancos_ids:
        solicitudes = mapa_solicitudes[b_id]
        if not solicitudes:
            continue

        # Calcular Sobrecoste (Spread)
        # Ec. A9: El coste extra se reparte entre los préstamos comerciales
        total_demandado = sum(s["monto"] for s in solicitudes)
        spread_refinanciacion = 0.0

        if monto_refinanciado[b_id] > 0 and total_demandado > 0:
            # Simplificación válida: repartimos el coste total de intereses IB proporcionalmente
            spread_refinanciacion = coste_refinanciacion[b_id] / total_demandado

        # Verificar Liquidez Final del Banco
        # (Si tras ir al IB todavía no tiene suficiente, raciona a todos por igual)
        liquidez_disponible = bancos_liquidez[b_id]

        ratio_cumplimiento = 1.0
        if liquidez_disponible < (total_demandado - 1e-5):
            ratio_cumplimiento = max(0, liquidez_disponible / total_demandado)

        # Ejecutar Préstamos a Empresas
        for sol in solicitudes:
            f_id = sol["firm"]
            monto_aprobado = sol["monto"] * ratio_cumplimiento

            if monto_aprobado < 1e-5:
                continue

            # Tasa Final = Tasa Base + Spread IB
            tasa_final = sol["tasa_base"] + spread_refinanciacion

            # --- ACTUALIZACIÓN DE ESTADO 2: Mercado Crédito ---

            # 1. Movimiento de Caja
            bancos_liquidez[b_id] -= monto_aprobado
            firm_liquidez[f_id] += monto_aprobado

            # 2. Registro de Deuda (Empresa)
            # En lugar de una matriz gigante, usamos la lista de objetos préstamo
            nuevo_prestamo = {
                "firm_id": f_id,
                "banco_id": b_id,
                "principal": monto_aprobado,
                "tasa": tasa_final,
                "duration": p.DURATION_LOAN,  # Ejs: 10 periodos
                "edad": 0,
            }
            prestamos_activos_empresas.append(nuevo_prestamo)

            # Actualizamos el stock de deuda total (para cálculos rápidos de fragilidad)
            firm_deuda[f_id] += monto_aprobado

    # Al ser actualización in-place (referencias), no es obligatorio retornar nada,
    # pero retornamos True para control de flujo.
    return True

