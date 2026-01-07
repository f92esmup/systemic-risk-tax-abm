# Este scripp computa el paso 2
from logica.debtrank import calcular_impacto_sr
from parametros import Param as p
import numpy as np


def paso2_mercado_credito(
    firm_ids, firm_liquidez, firm_deuda, factura_esperada_salarial, bancos_ids
):
    """Ejecuta la interación del mercado de credito (empress piden a bancos).
    Argumentos:
        firm_ids: array de IDs de las empresas.
        firm_liquidez: array de liquidez de las empresas.
        firm_deuda: array de deuda actual de las empresas.
        factura_esperada_salarial: array de facturas salariales esperadas.
        bancos_ids: array de IDs de los bancos.
    Retorna:
    solicitudes: Matriz o lista de prestamos solicitados [Firm_ID, Bank_ID, Monto, Tasa].
    demanda_credito: Vector con la demanda de crédito de cada empresa (para tracking).
    """
    # 1. Identificar demanda de crédito:
    credito_necesario = np.maximum(factura_esperada_salarial - firm_liquidez, 0)

    # Filtramos empresas que necesitan crédito.
    firms_necesitan_credito = firm_ids[credito_necesario > 1e-5]

    # Si ninguna empresa necesita crédito, retornamos vacío.
    if len(firms_necesitan_credito) == 0:
        return [], credito_necesario

    # 2. Calcular Fragilidad crediticia de las empresas:
    # Definición de apendice A.1: ratio deuda/activos.
    # sumamos epsilon para evitar división por cero.r
    ratio_fragilidad = firm_deuda / (firm_liquidez + 1e-9)
    mu_fragilidad = np.tanh(ratio_fragilidad)

    # 3. Generar Especificidad de los Bancs (chi).
    # Variable aleatoria uniforme entre 0 y 1.
    chi_bancos = np.random.uniform(0, 1, p.B)

    # Registramos contratos exitosos (aún no aprobados por la liquidez del banco).
    # Formato: [Firm_ID, Bank_ID, Monto, Tasa]
    contratos_potenciales = []

    # 4. Loop de emparejamiento:

    for firm_id in firms_necesitan_credito:
        monto = credito_necesario[firm_id]

        # 4.1 Seleccionar n_bancos aleatorios sin reemplazo.
        bancos_visitados = np.random.choice(bancos_ids, size=p.n_bancos, replace=False)

        # 4.2 Los bancos cotizan tasas (ecuación A.1)
        # r_ik = r_bar * (1 + chi_i * mu_k)
        rates_ofrecidos = p.r_bar * (
            1 + chi_bancos[bancos_visitados] * mu_fragilidad[firm_id]
        )

        # 4.3 Empresa elige la mejor tasa.
        mejor_id = np.argmin(rates_ofrecidos)
        mejor_rate = rates_ofrecidos[mejor_id]
        mejor_banco = bancos_visitados[mejor_id]

        # 4.4 Racionamiento por tasa umbral.

        monto_final = monto

        if mejor_rate > p.r_max:
            # La empresa rudece su petición por miedo al costo financiero.
            monto_final = monto * p.phi

        # Guardamos el contrato potencial.
        contratos_potenciales.append([firm_id, mejor_banco, monto_final, mejor_rate])

    return np.array(contratos_potenciales), credito_necesario


def paso2_interbancario(
    bancos_ids,
    bancos_liquidez,
    bancos_patrimonio,
    bancos_depositos,
    bancos_deuda_acumulada,
    contratos_potenciales_empresas,
    matriz_interbancaria_anterior,
    tax_mode=p.TAX_MODE,
):
    """
    Gestiona el mercado interbancario. Los bancos con déficit piden a los bancos con superávit.
    Incluye la lógica de selección de tasa y aplicación de impuestos.

    Retorna:
        nuevos_prestamos_ib: Lista [Lender, Borrower, Amount, Rate]
        prestamos_empresas_finales: Lista de contratos de empresas que SI se financiaron.
        bancos_liquidez_final: Liquidez actualizada.
    """
    # 1. Calcular Demanda Total por Banco
    demanda_por_banco = np.zeros(p.B)

    # contratos_potenciales_empresas es [Firm, Bank, Amount, Rate]
    if len(contratos_potenciales_empresas) > 0:
        for contrato in contratos_potenciales_empresas:
            b_idx = int(contrato[1])
            monto = contrato[2]
            demanda_por_banco[b_idx] += monto

    # 2. Identificar bancos con déficit y superávit
    # balance_liquidez = bancos_liquidez - demanda_por_banco
    deficit_ids = bancos_ids[bancos_liquidez < 0]  # Necesitan pedir
    superavit_ids = bancos_ids[bancos_liquidez > 0]  # Pueden prestar

    # 3. Cálculo de Fragilidad Financiera
    # Referencia Appendix A.2:
    # Leverage = Pasivo Total / Patrimonio
    # Pasivo Total = Depósitos + Deuda Interbancaria Acumulada

    pasivo_total = bancos_depositos + bancos_deuda_acumulada
    # Evitamos división por cero en bancos quebrados o sin patrimonio
    patrimonio_seguro = np.maximum(bancos_patrimonio, 1e-9)
    leverage_bancos = pasivo_total / patrimonio_seguro
    mu_bancos = np.tanh(leverage_bancos)  # Función de fragilidad
    # Especificidad de los bancos (psi)
    psi_bancos = np.random.uniform(0, p.psi_max, p.B)
    # Lista para registrar los nuevos préstamos interbancarios
    nuevos_prestamos_ib = []

    # Copia temporal de liquidez para ir actualizando mientras se prestan
    bancos_liquidez_temp = bancos_liquidez.copy()

    # 4. Loop de Mercado Interbancario
    # Los bancos deficitarios buscan cubrir su hueco

    for banco_deudor in deficit_ids:
        necesidad = abs(bancos_liquidez[banco_deudor])
        monto_conseguido = 0

        if len(superavit_ids) == 0:
            continue

        # 4.1 Solicitar cotizaciones a TODOS los bancos con superávit
        posibles_prestamistas = []

        for banco_acreedor in superavit_ids:
            disponible_j = bancos_liquidez_temp[banco_acreedor]

            if disponible_j < 1e-5:
                continue  # No puede prestar nada

            # Calcular Tasa Ofertada (A2) + Impuestos (A5-A9)
            # r_ji = r_bar * (1 + psi_j * mu_i)
            r_base = p.r_bar * (
                1 + psi_bancos[banco_acreedor] * mu_bancos[banco_deudor]
            )

            impuesto_adicional = 0.0

            if tax_mode == "SRT":
                # A7: r_total = r + SRT / loan_size
                # Calculamos el SRT específico para esta transacción potencial
                monto_potencial = min(disponible_j, necesidad)
                if (
                    monto_potencial > 1e-5
                ):  # NO USO CERO para evitar pequeñas variaciones.
                    impacto_sr = calcular_impacto_sr(
                        banco_deudor,
                        banco_acreedor,
                        monto_potencial,
                        matriz_interbancaria_anterior,
                    )
                    impuesto_adicional = (p.ZETA * impacto_sr) / monto_potencial

            elif tax_mode == "TOBIN":
                # A6: r_total = r + 0.002
                impuesto_adicional = p.TOBIN_RATE

            r_total = r_base + impuesto_adicional

            posibles_prestamistas.append(
                {
                    "banco_prestamista_id": banco_acreedor,
                    "rate": r_total,
                    "monto_maximo": disponible_j,
                }
            )
        # 4.2 Ordenar por tasa (de menor a mayor)
        posibles_prestamistas.sort(key=lambda x: x["rate"])

        # 4.3 Cubrir la necesidad con las mejores ofertas.
        for oferta in posibles_prestamistas:
            if necesidad <= 1e-5:
                break  # Ya se cubrió la necesidad

            banco_prestamista_id = oferta["banco_prestamista_id"]
            tasa_ofrecida = oferta["rate"]

            # El monto es el mínimoo entre lo que necisita y lo que puede prestar
            monto_prestamo = min(necesidad, oferta["monto_maximo"])

            # Registrar el préstamo
            nuevos_prestamos_ib.append(
                [banco_prestamista_id, banco_deudor, monto_prestamo, tasa_ofrecida]
            )

            # Actualizar estados temporales
            bancos_liquidez_temp[banco_prestamista_id] -= monto_prestamo
            necesidad -= monto_prestamo
            monto_conseguido += monto_prestamo

        # Al final del proceso, actualizamos la liquidez real del banco_deudor con lo conseguido
        bancos_liquidez[banco_deudor] += monto_conseguido

    # 5. Resolución Final de Préstamos a Empresas
    # El banco solo paga a la empresa si tiene liquidez DESPUÉS del interbancario.
    prestamos_empresas_finales = []

    if len(contratos_potenciales_empresas) > 0:
        for contrato in contratos_potenciales_empresas:
            b_idx = int(contrato[1])
            monto = contrato[2]

            # Verificación estricta de fondos
            if bancos_liquidez[b_idx] >= (monto - 1e-5):
                prestamos_empresas_finales.append(contrato)
                bancos_liquidez[b_idx] -= monto
            else:
                # Credit Crunch: El banco no pudo fondearse y rechaza el crédito
                pass

    return (
        np.array(nuevos_prestamos_ib),
        np.array(prestamos_empresas_finales),
        bancos_liquidez,
    )
