# logica/paso2.py
from logica.debtrank import calcular_impacto_sr
from parametros import Param as p
import numpy as np


def paso2_mercado_credito(
    firm_ids, firm_liquidez, firm_deuda, factura_esperada_salarial, bancos_ids
):
    """
    (Sin cambios respecto a tu versión anterior)
    Ejecuta la interacción del mercado de credito (empresas piden a bancos).
    """
    # 1. Identificar demanda de crédito:
    credito_necesario = np.maximum(factura_esperada_salarial - firm_liquidez, 0)
    firms_necesitan_credito = firm_ids[credito_necesario > 1e-5]

    if len(firms_necesitan_credito) == 0:
        return [], credito_necesario

    # 2. Fragilidad crediticia de empresas
    ratio_fragilidad = firm_deuda / (firm_liquidez + 1e-9)
    mu_fragilidad = np.tanh(ratio_fragilidad)

    # 3. Especificidad de Bancos
    chi_bancos = np.random.uniform(0, 1, p.B)
    contratos_potenciales = []

    # 4. Loop de emparejamiento
    for firm_id in firms_necesitan_credito:
        monto = credito_necesario[firm_id]
        bancos_visitados = np.random.choice(bancos_ids, size=p.n_bancos, replace=False)
        rates_ofrecidos = p.r_bar * (
            1 + chi_bancos[bancos_visitados] * mu_fragilidad[firm_id]
        )
        mejor_id = np.argmin(rates_ofrecidos)
        mejor_rate = rates_ofrecidos[mejor_id]
        mejor_banco = bancos_visitados[mejor_id]

        monto_final = monto
        if mejor_rate > p.r_max:
            monto_final = monto * p.phi

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
    MODIFICADO: Ahora devuelve también la tasa impositiva aplicada (tax_rate)
    en la 5ta columna de nuevos_prestamos_ib.
    """
    # 1. Calcular Demanda Total por Banco
    demanda_por_banco = np.zeros(p.B)
    if len(contratos_potenciales_empresas) > 0:
        for contrato in contratos_potenciales_empresas:
            b_idx = int(contrato[1])
            monto = contrato[2]
            demanda_por_banco[b_idx] += monto

    # 2. Identificar bancos con déficit y superávit
    balance_liquidez = bancos_liquidez - demanda_por_banco
    deficit_ids = bancos_ids[balance_liquidez < -1e-5]
    superavit_ids = bancos_ids[balance_liquidez > 1e-5]

    # 3. Cálculo de Fragilidad Financiera
    pasivo_total = bancos_depositos + bancos_deuda_acumulada
    patrimonio_seguro = np.maximum(bancos_patrimonio, 1e-9)
    leverage_bancos = pasivo_total / patrimonio_seguro
    mu_bancos = np.tanh(leverage_bancos)
    psi_bancos = np.random.uniform(0, p.psi_max, p.B)

    nuevos_prestamos_ib = []
    bancos_liquidez_temp = bancos_liquidez.copy()

    # 4. Loop de Mercado Interbancario
    for banco_deudor in deficit_ids:
        necesidad = abs(balance_liquidez[banco_deudor])
        monto_conseguido = 0

        if len(superavit_ids) == 0:
            continue

        posibles_prestamistas = []

        for banco_acreedor in superavit_ids:
            disponible_j = bancos_liquidez_temp[banco_acreedor]
            if disponible_j < 1e-5:
                continue

            r_base = p.r_bar * (
                1 + psi_bancos[banco_acreedor] * mu_bancos[banco_deudor]
            )

            impuesto_rate = 0.0

            if tax_mode == "SRT":
                monto_potencial = min(disponible_j, necesidad)
                if monto_potencial > 1e-5:
                    impacto_euros_sr = calcular_impacto_sr(
                        banco_deudor,
                        banco_acreedor,
                        monto_potencial,
                        matriz_interbancaria_anterior,
                        bancos_patrimonio,
                        bancos_depositos,
                        bancos_deuda_acumulada,
                    )
                    impuesto_rate = (p.ZETA * impacto_euros_sr) / monto_potencial

            elif tax_mode == "TOBIN":
                impuesto_rate = p.TOBIN_RATE

            r_total = r_base + impuesto_rate

            posibles_prestamistas.append(
                {
                    "banco_prestamista_id": banco_acreedor,
                    "rate": r_total,
                    "tax_rate": impuesto_rate,  # GUARDAMOS EL TAX RATE APARTE
                    "monto_maximo": disponible_j,
                }
            )

        posibles_prestamistas.sort(key=lambda x: x["rate"])

        for oferta in posibles_prestamistas:
            if necesidad <= 1e-5:
                break

            banco_prestamista_id = oferta["banco_prestamista_id"]
            tasa_ofrecida = oferta["rate"]
            tasa_impuesto = oferta["tax_rate"]

            monto_prestamo = min(necesidad, oferta["monto_maximo"])

            # AHORA GUARDAMOS 5 VALORES: [Lender, Borrower, Amount, TotalRate, TaxRate]
            nuevos_prestamos_ib.append(
                [
                    banco_prestamista_id,
                    banco_deudor,
                    monto_prestamo,
                    tasa_ofrecida,
                    tasa_impuesto,
                ]
            )

            bancos_liquidez_temp[banco_prestamista_id] -= monto_prestamo
            necesidad -= monto_prestamo
            monto_conseguido += monto_prestamo

        bancos_liquidez[banco_deudor] += monto_conseguido

    # 5. Resolución Final de Préstamos a Empresas
    prestamos_empresas_finales = []
    if len(contratos_potenciales_empresas) > 0:
        for contrato in contratos_potenciales_empresas:
            b_idx = int(contrato[1])
            monto = contrato[2]
            if bancos_liquidez[b_idx] >= (monto - 1e-5):
                prestamos_empresas_finales.append(contrato)
                bancos_liquidez[b_idx] -= monto

    return (
        np.array(nuevos_prestamos_ib),
        np.array(prestamos_empresas_finales),
        bancos_liquidez,
    )

