import numpy as np
from parametros import Param as p


def paso1(
    precios_prev: np.ndarray,
    produccion_prev: np.ndarray,
    ventas_prev: np.ndarray,
    liquidez_prev: np.ndarray,
    inventario_acumulado: np.ndarray,
    mask_renacidas: np.ndarray,
):
    """
    Paso 1: Planificación de Empresas (Firms Planning).

    Calcula precio objetivo, producción deseada, demanda laboral y demanda de crédito.
    Basado en expectativas adaptativas y reglas de ajuste (Greenwald-Stiglitz).
    Ref: [cite: 213, 214, 215, 200]

    Args:
        precios_prev: Vector (F,) con precios del periodo anterior.
        produccion_prev: Vector (F,) con producción del periodo anterior.
        ventas_prev: Vector (F,) con ventas del periodo anterior.
        liquidez_prev: Vector (F,) con liquidez disponible actual.
        inventario_acumulado: Vector (F,) con stock de bienes no vendidos.
        mask_renacidas: Vector booleano (F,) True si la empresa renació en este turno.

    Returns:
        Tupla con (nuevos_precios, demanda_laboral, produccion_necesaria,
                   demanda_credito, factura_salarial, demanda_objetivo_total)
    """

    # -------------------------------------------------------------------------
    # 1. Cálculo de Referencias de Mercado (Vectorización tipo Matlab)
    # -------------------------------------------------------------------------
    # Definimos empresas sanas para el cálculo de promedios de mercado.
    # Si una empresa acaba de renacer, no debería influir en el promedio del mercado "maduro".
    idx_sanas = ~mask_renacidas

    if np.any(idx_sanas):
        avg_precio_mercado = np.mean(precios_prev[idx_sanas])
        avg_ventas_mercado = np.mean(ventas_prev[idx_sanas])
    else:
        # Fallback de seguridad si todas son nuevas o quebraron
        avg_precio_mercado = np.mean(precios_prev)
        avg_ventas_mercado = np.mean(ventas_prev) if np.mean(ventas_prev) > 0 else 1.0

    # -------------------------------------------------------------------------
    # 2. Clasificación de Estado (Adaptive Rule) [cite: 213]
    # -------------------------------------------------------------------------
    # Regla: Deviation of price & Excess supply (inventory)

    # a) Precio relativo
    precio_alto = precios_prev > avg_precio_mercado

    # b) Inventario excesivo (Umbral pequeño para evitar ruido numérico)
    exceso_inventario = inventario_acumulado > 1e-4

    # -------------------------------------------------------------------------
    # 3. Definición de Ajustes (Regla Heurística Greenwald-Stiglitz)
    # -------------------------------------------------------------------------
    # "Random variations in operating costs/strategy"

    magnitud_ajuste = np.random.uniform(p.RANGO_AJUSTE_MIN, p.RANGO_AJUSTE_MAX, p.F)

    delta_p = np.zeros(p.F)
    delta_q = np.zeros(p.F)

    # Lógica Vectorizada de Cuadrantes:

    # A: Precio Alto + Inventario -> Bajar P, Bajar Q
    mask_A = precio_alto & exceso_inventario
    delta_p[mask_A] = -magnitud_ajuste[mask_A]
    delta_q[mask_A] = -magnitud_ajuste[mask_A]

    # B: Precio Bajo + Sin Inventario -> Subir P, Subir Q
    mask_B = (~precio_alto) & (~exceso_inventario)
    delta_p[mask_B] = magnitud_ajuste[mask_B]
    delta_q[mask_B] = magnitud_ajuste[mask_B]

    # C: Precio Bajo + Inventario -> Mantener P, Bajar Q
    mask_C = (~precio_alto) & exceso_inventario
    delta_p[mask_C] = 0.0
    delta_q[mask_C] = -magnitud_ajuste[mask_C]

    # D: Precio Alto + Sin Inventario -> Mantener P, Subir Q
    mask_D = precio_alto & (~exceso_inventario)
    delta_p[mask_D] = 0.0
    delta_q[mask_D] = magnitud_ajuste[mask_D]

    # -------------------------------------------------------------------------
    # 4. Aplicación de Objetivos y Corrección "Zero Trap"
    # -------------------------------------------------------------------------

    # Nuevos Precios
    nuevos_precios = precios_prev * (1 + delta_p)
    # CORRECCIÓN: Guardrail relativo. No permitir precios absurdamente bajos comparados al mercado.
    nuevos_precios = np.maximum(nuevos_precios, 0.5 * avg_precio_mercado)

    # Demanda Esperada (Target Quantity)
    # Si tengo inventario, mi demanda real fue lo que vendí.
    # Si no tengo inventario, mi demanda fue AL MENOS lo que produje (quizás más).
    base_demanda = np.where(exceso_inventario, ventas_prev, produccion_prev)

    # CORRECCIÓN CRÍTICA: Evitar que una empresa muera si base_demanda es 0.
    base_demanda = np.maximum(base_demanda, 0.1 * avg_ventas_mercado)

    demanda_objetivo_total = base_demanda * (1 + delta_q)
    demanda_objetivo_total = np.maximum(demanda_objetivo_total, 0.0)

    # -------------------------------------------------------------------------
    # 5. Tratamiento de Renacidas (Ref [cite: 200])
    # -------------------------------------------------------------------------
    # "Initial estimates for D(t+1) and P(t+1) equals respective current averages"

    if np.any(mask_renacidas):
        nuevos_precios[mask_renacidas] = avg_precio_mercado
        demanda_objetivo_total[mask_renacidas] = avg_ventas_mercado

        # Virtualmente reseteamos inventario para el cálculo de producción de las renacidas
        # (El inventario real se limpia en el paso de bancarrota, esto es solo local para el cálculo)
        inventario_virtual = inventario_acumulado.copy()
        inventario_virtual[mask_renacidas] = 0
    else:
        inventario_virtual = inventario_acumulado

    # -------------------------------------------------------------------------
    # 6. Producción Necesaria y Demanda Laboral
    # -------------------------------------------------------------------------
    # Producción = Demanda Esperada - Inventario actual
    produccion_necesaria = demanda_objetivo_total - inventario_virtual
    produccion_necesaria = np.maximum(produccion_necesaria, 0.0)

    # Demanda Laboral (N) = Y / alpha [cite: 208, 214]
    demanda_laboral = produccion_necesaria / p.ALPHA
    factura_salarial = demanda_laboral * p.W_BASE

    # -------------------------------------------------------------------------
    # 7. Demanda de Crédito
    # -------------------------------------------------------------------------
    # "If wages... exceed current liquidity, it applies for a loan" [cite: 215]

    deficit = factura_salarial - liquidez_prev
    demanda_credito = np.maximum(deficit, 0.0)

    return (
        nuevos_precios,
        demanda_laboral,
        produccion_necesaria,
        demanda_credito,
        factura_salarial,
        demanda_objetivo_total,
    )
