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
    
    Implementa la heurística de expectativas adaptativas descrita en el Apéndice A.2
    [cite_start]del paper 1401.8026[cite: 2219]. Calcula P(t+1) y D(t+1) basándose en 
    señales de mercado y estado del inventario.

    Regla de Renacimiento:
    [cite_start]"Initial estimates for D(t+1) and P(t+1) equals respective current averages"[cite: 2206].
    """

    # -------------------------------------------------------------------------
    # 1. Cálculo de Referencias de Mercado (Vectorización Robusta)
    # -------------------------------------------------------------------------
    # Excluimos empresas recién nacidas para no sesgar el precio de mercado
    idx_sanas = ~mask_renacidas
    
    # Cálculo seguro de medias usando manejo de excepciones de NumPy implícito (size > 0)
    # Cálculo seguro de medias
    if np.any(idx_sanas):
        avg_precio_mercado = np.mean(precios_prev[idx_sanas])
        avg_ventas_mercado = np.mean(ventas_prev[idx_sanas])
    else:
        # Fallback extremo: todas las empresas son nuevas
        avg_precio_mercado = np.mean(precios_prev) if len(precios_prev) > 0 else 1.0
        avg_ventas = np.mean(ventas_prev) if len(ventas_prev) > 0 else 1.0
        avg_ventas_mercado = avg_ventas if avg_ventas > 1e-9 else 1.0


    # -------------------------------------------------------------------------
    # 2. Clasificación de Estado (Adaptive Rule)
    # -------------------------------------------------------------------------
    precio_alto = precios_prev > avg_precio_mercado
    exceso_inventario = inventario_acumulado > p.UMBRAL_INVENTARIO

    # -------------------------------------------------------------------------
    # 3. Definición de Ajustes (Greenwald-Stiglitz)
    # -------------------------------------------------------------------------
    magnitud_ajuste = np.random.uniform(p.RANGO_AJUSTE_MIN, p.RANGO_AJUSTE_MAX, p.F)

    # Inicializamos deltas
    delta_p = np.zeros(p.F)
    delta_q = np.zeros(p.F)

    # Cuadrante A: Precio Alto + Inventario -> Bajar P, Bajar Q
    mask_A = precio_alto & exceso_inventario
    delta_p[mask_A] = -magnitud_ajuste[mask_A]
    delta_q[mask_A] = -magnitud_ajuste[mask_A]

    # Cuadrante B: Precio Bajo + Sin Inventario -> Subir P, Subir Q
    mask_B = (~precio_alto) & (~exceso_inventario)
    delta_p[mask_B] = magnitud_ajuste[mask_B]
    delta_q[mask_B] = magnitud_ajuste[mask_B]

    # Cuadrante C: Precio Bajo + Inventario -> Mantener P, Bajar Q
    mask_C = (~precio_alto) & exceso_inventario
    # delta_p[mask_C] = 0.0 (Implicito)
    delta_q[mask_C] = -magnitud_ajuste[mask_C]

    # Cuadrante D: Precio Alto + Sin Inventario -> Mantener P, Subir Q
    mask_D = precio_alto & (~exceso_inventario)
    # delta_p[mask_D] = 0.0 (Implicito)
    delta_q[mask_D] = magnitud_ajuste[mask_D]

    # -------------------------------------------------------------------------
    # 4. Cálculo de Objetivos (Targets)
    # -------------------------------------------------------------------------
    
    # -- PRECIOS --
    nuevos_precios = precios_prev * (1 + delta_p)
    # Guardrail de estabilidad (Refactorizado a Param)
    nuevos_precios = np.maximum(nuevos_precios, p.SUELO_PRECIO_RELATIVO * avg_precio_mercado)

    # -- CANTIDADES --
    # Estimación de demanda base: Si sobró stock, la demanda real fue lo vendido.
    # Si no sobró, asumimos que podríamos haber vendido al menos lo producido.
    base_demanda = np.where(exceso_inventario, ventas_prev, produccion_prev)
    
    # Guardrail de estabilidad para evitar espiral de muerte (Zero Trap)
    base_demanda = np.maximum(base_demanda, p.SUELO_DEMANDA_RELATIVO * avg_ventas_mercado)

    demanda_objetivo_total = np.maximum(base_demanda * (1 + delta_q), 0.0)

    # -------------------------------------------------------------------------
    # [cite_start]5. Tratamiento de Renacidas (Reset Rules) [cite: 2206]
    # -------------------------------------------------------------------------
    if np.any(mask_renacidas):
        nuevos_precios[mask_renacidas] = avg_precio_mercado
        demanda_objetivo_total[mask_renacidas] = avg_ventas_mercado
        
        # [FIX] Reseteamos también la liquidez de las renacidas para el cálculo de crédito
        # Las renacidas "aparecen" con una dotación inicial de liquidez (equity injection?)
        liquidez_efectiva = np.where(mask_renacidas, p.LIQUIDEZ_INICIAL_FIRMAS, liquidez_prev)

        # Para las renacidas, ignoramos el inventario heredado (que se liquidará)
        # al calcular lo que necesitan producir hoy.
        inventario_efectivo = np.where(mask_renacidas, 0.0, inventario_acumulado)
    else:
        liquidez_efectiva = liquidez_prev
        inventario_efectivo = inventario_acumulado

    # -------------------------------------------------------------------------
    # 6. Producción Necesaria y Demanda Laboral
    # -------------------------------------------------------------------------
    # Y_required = Expected_Demand - Current_Inventory
    produccion_necesaria = np.maximum(demanda_objetivo_total - inventario_efectivo, 0.0)

    # [cite_start]N = Y / alpha [cite: 2220]
    demanda_laboral = produccion_necesaria / p.ALPHA
    factura_salarial = demanda_laboral * p.W_BASE

    # -------------------------------------------------------------------------
    # 7. Demanda de Crédito
    # -------------------------------------------------------------------------
    # [cite_start]"If wages... exceed current liquidity, it applies for a loan" [cite: 2221]
    # Usamos liquidez_efectiva (que ya tiene el reset para las nuevas)
    deficit = factura_salarial - liquidez_efectiva
    demanda_credito = np.maximum(deficit, 0.0)

    return (
        nuevos_precios,
        demanda_laboral,
        produccion_necesaria,
        demanda_credito,
        factura_salarial,
        demanda_objetivo_total,
    )
