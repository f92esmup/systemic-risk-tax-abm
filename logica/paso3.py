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
    liquidez_empresas_post_prod = presupuesto_disponible - factura_salarial_real
    liquidez_empresas_post_prod = np.maximum(liquidez_empresas_post_prod, 0.0)

    # --- MATRICES DE RELACION (F-H) ---
    # Distribuir la factura salarial entre hogares específicos.
    # Si no se pasó matriz previa, creamos una estática/aleatoria por ahora
    # En iteraciones futuras esto debería evolucionar (Hire/Fire).
    
    # [Refactor] Generamos una matriz de pagos de salarios (F, H)
    # Por ahora: Asignación aleatoria uniforme para "cubrir" la demanda.
    # Un modelo más complejo mantendría "empleados fijos".
    # Asumimos que todos los hogares proveen trabajo proporcionalmente al mercado
    # O mejor: Cada empresa tiene un subset de empleados.
    
    # Implementación vectorizada simple: 
    # Repartir el costo salarial de la empresa f entre todos los hogares (mean field)
    # O mantener la estructura de grafo requerida.
    
    # Versión Grafo Denso (Mean Field - Fallback si no hay estado persistente aun):
    # wages_matrix = np.outer(factura_salarial_real, np.ones(p.H) / p.H)
    
    # Versión Grafo Esparso (Random Links cada turno - Spot Market):
    # Asignamos k trabajadores por empresa.
    
    # Dado que Main aun no mantiene el estado "matriz_laboral", generamos
    # el flujo de pagos (Wages Flow Matrix) aquí.
    
    # Para cumplir "Relaciones como matrices":
    # Vamos a asumir que cada empresa contrata un set aleatorio de empleados en cada paso
    # proporcional a su tamaño, o simplemente distribuimos.
    # Para visualización rica, simulemos que contratan a ~N personas.
    
    wages_matrix = np.zeros((p.F, p.H))
    
    # Optimizacion: Distribuir proporcionalmente seria O(FH).
    # Usaremos una distribucion determinista basada en indices para velocidad y estabilidad graficable
    # Cada empresa f paga a los hogares en el rango [f*k, (f+1)*k] (mod H)
    # Esto crea una matriz "banda" visualmente interesante.
    
    # Numero de empleados "graficos" por empresa (solo para visualizacion del link)
    k_employees = max(1, p.H // p.F) 
    
    for f in range(p.F):
        wage_bill = factura_salarial_real[f]
        if wage_bill > 0:
            start_idx = (f * k_employees) % p.H
            # Asignamos a k empleados
            indices = np.arange(start_idx, start_idx + k_employees) % p.H
            wages_matrix[f, indices] = wage_bill / k_employees

    return (
        produccion_nueva,
        oferta_total_bienes,
        empleo_real,
        factura_salarial_real,
        liquidez_empresas_post_prod,
        wages_matrix, # (F, H) Payment Flow
    )
