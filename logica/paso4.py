# Script para implementar el paso 4 El consumo.
import numpy as np
from parametros import Param as p


def paso4_consumo(
    hogares_liquidez,
    firm_precios,
    firm_produccion_recien_hecha,
    firm_inventario_anterior,
):
    """
    Ejecuta el Paso 4: Mercado de Bienes de Consumo (Versión Vectorizada).

    Estrategia 'Matlab-style':
    1. Hogares eligen empresa más barata de su muestra 'z'.
    2. Se calcula la demanda agregada por empresa.
    3. Se calcula un 'ratio de satisfacción' basado en el stock disponible.
    4. Se ejecutan las compras simultáneamente aplicando el ratio (Racionamiento Proporcional).

    Args:
        hogares_liquidez: (H,) Dinero actual de los hogares.
        firm_precios: (F,) Precios actuales.
        firm_produccion_recien_hecha: (F,) Producción de este turno.
        firm_inventario_anterior: (F,) Stock no vendido del turno anterior.

    Returns:
        firm_ventas_reales: (F,) Unidades vendidas.
        firm_stock_final: (F,) Unidades sobrantes.
        firm_ingresos: (F,) Dinero ingresado.
        hogares_liquidez_post: (H,) Liquidez remanente del hogar.
        hogares_gasto_real: (H,) Gasto realizado.
    """

    # 0. Datos del Sistema
    H = len(hogares_liquidez)
    F = len(firm_precios)

    # Stock Total = Producción nueva + Inventario viejo
    firm_stock_total = firm_produccion_recien_hecha + firm_inventario_anterior

    # 1. Presupuesto y Elección de Proveedor
    # Presupuesto de gasto (c * Liquidez)
    hogares_presupuesto = hogares_liquidez * p.c

    # Cada hogar elige 'z' empresas al azar. Matriz (H, z)
    elecciones_indices = np.random.randint(0, F, size=(H, p.z))

    # Obtenemos los precios de esas empresas. Matriz (H, z)
    # Usamos 'advanced indexing'
    precios_muestra = firm_precios[elecciones_indices]

    # Encontramos la columna con el precio mínimo para cada fila (hogar)
    idx_col_min = np.argmin(precios_muestra, axis=1)

    # Obtenemos el ID real de la empresa elegida por cada hogar
    # indices[i, j] selecciona el elemento correcto de la matriz elecciones
    hogares_firm_elegida = elecciones_indices[np.arange(H), idx_col_min]

    # Obtenemos el precio de la empresa elegida
    hogares_precio_compra = firm_precios[hogares_firm_elegida]

    # 2. Demanda Nominal (Cuánto "quieren" comprar)
    # Demanda = Presupuesto / Precio
    hogares_demanda_cantidad = hogares_presupuesto / hogares_precio_compra

    # Agregamos la demanda total por empresa usando bincount (super rápido)
    # bincount suma los pesos (demanda) para cada bin (empresa ID)
    firm_demanda_agregada = np.bincount(
        hogares_firm_elegida, weights=hogares_demanda_cantidad, minlength=F
    )

    # 3. Racionamiento (Proporcional)
    # Si Stock > Demanda -> Ratio = 1.0 (Se vende todo lo pedido)
    # Si Stock < Demanda -> Ratio = Stock / Demanda (Se vende una fracción)
    # Añadimos epsilon para evitar división por cero
    ratio_cobertura = np.divide(
        firm_stock_total,
        firm_demanda_agregada,
        out=np.ones(F),
        where=firm_demanda_agregada > 1e-9,
    )

    # El ratio no puede ser mayor a 1 (no pueden vender más de lo que tienen)
    ratio_cobertura = np.minimum(1.0, ratio_cobertura)

    # 4. Ejecución de Transacciones
    # Cantidad Real = Demanda * Ratio de la empresa elegida
    ratio_aplicado_a_hogar = ratio_cobertura[hogares_firm_elegida]
    hogares_compra_real = hogares_demanda_cantidad * ratio_aplicado_a_hogar

    hogares_gasto_real = hogares_compra_real * hogares_precio_compra

    # 5. Agregación de Resultados para Empresas
    firm_ventas_reales = np.bincount(
        hogares_firm_elegida, weights=hogares_compra_real, minlength=F
    )
    firm_ingresos = np.bincount(
        hogares_firm_elegida, weights=hogares_gasto_real, minlength=F
    )

    # Stock Final = Stock Total - Ventas Reales
    # (Matemáticamente debería ser >= 0 debido al min(1.0) en el ratio)
    firm_stock_final = np.maximum(firm_stock_total - firm_ventas_reales, 0.0)

    # 6. Actualización Hogares
    hogares_liquidez_post = hogares_liquidez - hogares_gasto_real

    return (
        firm_ventas_reales,
        firm_stock_final,
        firm_ingresos,
        hogares_liquidez_post,
        hogares_gasto_real,
    )
