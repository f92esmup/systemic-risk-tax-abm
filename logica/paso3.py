import numpy as np
from parametros import Param as p


def paso3(
    demanda_laboral_necesaria,  # Vector (F,) N_required calculado en Paso 1 (UNITS: Workers)
    liquidez_previa,  # Vector (F,) Liquidez que tenía la empresa al inicio
    nuevos_prestamos,  # Vector (F,) Lo que consiguió prestado en Paso 2
    inventario_acumulado,  # Vector (F,) Stock remanente
):
    """
    Paso 3: Producción y Mercado Laboral.

    1. Las empresas re-evalúan su fuerza laboral según restricción presupuestaria.
    2. Producen bienes físicos.
    3. Pagan salarios (transferencia de liquidez a hogares).

    Ref: [cite: 190-218]
    """

    # -------------------------------------------------------------------------
    # 1. Presupuesto Total (Budget Constraint)
    # -------------------------------------------------------------------------
    # El dinero disponible para pagar nóminas es la caja inicial + deuda nueva.
    presupuesto_disponible = liquidez_previa + nuevos_prestamos
    
    # Assert de seguridad lógica: Liquidez no debería ser negativa al entrar a producción
    # Si ocurrió, indica error en lógica de quiebra del paso anterior
    if np.any(presupuesto_disponible < 0):
        # En producción real esto podría ser un warning en lugar de crash
        presupuesto_disponible = np.maximum(presupuesto_disponible, 0.0)


    # -------------------------------------------------------------------------
    # 2. Determinar Empleo Real (Re-evaluación)
    # -------------------------------------------------------------------------
    # Capacidad máxima de contratación con el dinero actual
    # N_max = Budget / w
    max_trabajadores_pagables = presupuesto_disponible / p.W_BASE

    # Si conseguí el crédito, contrato lo que quería (demanda_objetivo).
    # Si no, contrato solo lo que puedo pagar.
    #  "firms re-evaluate the required workforce"
    empleo_real = np.minimum(demanda_laboral_necesaria, max_trabajadores_pagables)

    # -------------------------------------------------------------------------
    # 3. Producción Física
    # -------------------------------------------------------------------------
    # Función de producción lineal: Y = alpha * N [cite: 208]
    produccion_nueva = empleo_real * p.ALPHA

    # Oferta disponible para vender en el Paso 4
    oferta_total_bienes = inventario_acumulado + produccion_nueva

    # -------------------------------------------------------------------------
    # 4. Pago de Salarios (Salida de Caja)
    # -------------------------------------------------------------------------
    factura_salarial_real = empleo_real * p.W_BASE

    # Actualizamos la caja de la empresa.
    # Importante: Las empresas llegan al mercado de consumo (Paso 4) con la caja
    # mermada tras pagar sueldos. Solo se rellenará si venden.
    liquidez_empresas_post_prod = presupuesto_disponible - factura_salarial_real

    # Corrección de precisión numérica (evitar valores negativos infinitesimales)
    liquidez_empresas_post_prod = np.maximum(liquidez_empresas_post_prod, 0.0)

    return (
        produccion_nueva,
        oferta_total_bienes,
        empleo_real,
        factura_salarial_real,  # Input clave para el Paso 4 (Ingreso Hogares)
        liquidez_empresas_post_prod,  # Estado financiero actual de la empresa
    )
