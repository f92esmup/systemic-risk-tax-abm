import numpy as np
from parametros import Param as p

def paso7_cierre_y_metricas(
    indices_quiebra_firmas,    # IDs de empresas que quebraron en este t
    firm_ids,
    firm_precios,              # P_i(t)
    firm_produccion_real,      # Y_i(t)
    firm_ventas_reales,        # S_i(t)
    firm_inventario,           # Stock final paso 4
    firm_liquidez,
    firm_deuda,
    bancos_activos,
    fondo_rescate_acumulado,
    firm_trabajadores_reales,
    hogares_es_trabajador
):
    """
    Ejecuta el Paso 7: Actualización de Estado, 'Rebirth' y Métricas Macro.
    
    Lógica de Réplica Exacta [Apéndice A]:
    1. 'Bankrupt firms immediately start a new company'.
    2. 'Initial estimates equals averages' (Precio, Producción, Ventas Históricas).
    3. Cálculo de indicadores macroeconómicos.
    """
    
    # --- 1. REGLA DE RENACIMIENTO (REBIRTH) ---
    # Las empresas quebradas renacen como nuevas entidades.
    # Debemos resetear sus expectativas a la media del sistema para que puedan 
    # planificar correctamente en el Paso 1 del siguiente turno (t+1).
    
    if len(indices_quiebra_firmas) > 0:
        # Calculamos promedios de los SOBREVIVIENTES (para no contaminar con los datos de quiebra)
        mask_vivos = np.ones(p.F, dtype=bool)
        mask_vivos[indices_quiebra_firmas] = False
        
        # Si todos mueren (apocalipsis), usamos valores por defecto o globales
        if np.sum(mask_vivos) == 0:
            avg_precio = np.mean(firm_precios)
            avg_prod = np.mean(firm_produccion_real)
            avg_ventas = np.mean(firm_ventas_reales)
        else:
            avg_precio = np.mean(firm_precios[mask_vivos])
            avg_prod = np.mean(firm_produccion_real[mask_vivos])
            avg_ventas = np.mean(firm_ventas_reales[mask_vivos])
            
        # Asignamos valores promedio a las nuevas empresas
        firm_precios[indices_quiebra_firmas] = avg_precio
        
        # Para el Paso 1 (Planificación), la empresa usa sus ventas pasadas y producción pasada.
        # Les damos el promedio para que no planifiquen 0.
        firm_produccion_real[indices_quiebra_firmas] = avg_prod
        firm_ventas_reales[indices_quiebra_firmas] = avg_ventas
        
        # Inventario: Una empresa nueva empieza sin stock viejo.
        firm_inventario[indices_quiebra_firmas] = 0
        
        # Liquidez y Deuda ya fueron reseteadas a 0 en Paso 5.
        # Nota: Al tener Liquidez 0 y Deuda 0, su fragilidad en Paso 2 será 0 (seguras).
        # Pedirán crédito basándose en su plan de producción (derivado de avg_ventas).

    # --- 2. CÁLCULO DE MÉTRICAS MACRO ---
    
    # PIB Real (Suma de Producción o Ventas? Paper suele usar Ventas Totales como Proxy de actividad)
    pib_real = np.sum(firm_ventas_reales)
    
    # Desempleo
    total_empleo = np.sum(firm_trabajadores_reales)
    fuerza_laboral = np.sum(hogares_es_trabajador)
    tasa_desempleo = 1.0 - (total_empleo / fuerza_laboral) if fuerza_laboral > 0 else 0.0
    
    # Deuda Total Sistema
    deuda_total_empresas = np.sum(firm_deuda)
    
    # Salud Bancaria
    bancos_vivos_count = np.sum(bancos_activos)
    
    # --- 3. PREPARACIÓN PARA t+1 ---
    # Devolvemos los vectores actualizados que servirán de input para el Paso 1 del siguiente loop.
    # Nota: Python pasa arrays por referencia, pero devolvemos explícitamente para claridad del flujo.
    
    metricas = {
        "pib": pib_real,
        "desempleo": tasa_desempleo,
        "quiebras_firmas": len(indices_quiebra_firmas),
        "bancos_vivos": bancos_vivos_count,
        "deuda_total": deuda_total_empresas,
        "fondo_rescate": fondo_rescate_acumulado
    }
    
    return (
        firm_precios,        # Actualizados (Rebirth)
        firm_produccion_real,# Actualizados (Rebirth) - Será Y(t-1) en el paso 1
        firm_ventas_reales,  # Actualizados (Rebirth) - Será S(t-1) en el paso 1
        firm_inventario,     # Actualizados (Rebirth reset)
        metricas
    )
