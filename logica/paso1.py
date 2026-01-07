# Este script computa el paso 1 "Las empresas definen la demanda de trabajo y capital".
from parametros import Param as p
import numpy as np


def paso1(precios, produccion, ventas):
    """Calcula la demanda esperada, precio nuevo y demanda de trabajadores.
    ARGumentos:
        prices: array de precios de las empresas en t-1.
        produccion: array de producción de las empresas en t-1.
        ventas: array de ventas de las empresas en t-1.
    Retorna:
        nuevos_precios: array de nuevos precios.
        demanda_trabajo: array de demanda de trabajo.
        demanda_esperada: array de demanda esperada.
        factura_esperada_salarial: array de factura salarial esperada.
    """

    # 1.1 Calcular el precio promedio del mercado actual.
    avg_precio = np.mean(precios)

    # 1.2 Determinar regla de ajuste:
    # comparar precio propio con promedio.
    precio_relativo = precios > avg_precio

    # Compara ventas con producción (exceso si ventas < produccion).
    # tenemos en cuenta el erro de punto flotante con un epsilon.
    exceso_inventario = (produccion - ventas) > 1e-5

    # definimos dirección de ajuste:

    cambio_precio = np.zeros(p.F)  # inicializamos array de cambios.
    cambio_cantidad = np.zeros(p.F)

    # Caso A: Tengo stock y soy caro -> bajar precio y producir menos.
    mascara_A = precio_relativo & exceso_inventario
    cambio_precio[mascara_A] = -p.SENSIBILIDAD_AJUSTE
    cambio_cantidad[mascara_A] = -p.SENSIBILIDAD_AJUSTE

    # Caso B: Tengo stock y soy barato -> subir precio y producir más.
    mascara_B = (~precio_relativo) & (~exceso_inventario)
    cambio_precio[mascara_B] = p.SENSIBILIDAD_AJUSTE
    cambio_cantidad[mascara_B] = p.SENSIBILIDAD_AJUSTE

    # Caso C: Si tengo stock y soy barato -> producir menos.
    mascara_C = exceso_inventario & (~precio_relativo)
    cambio_precio[mascara_C] = 0
    cambio_cantidad[mascara_C] = -p.SENSIBILIDAD_AJUSTE

    # Caso D: Si no tengo stock y soy caro -> producir más.
    mascara_D = (~exceso_inventario) & precio_relativo
    cambio_precio[mascara_D] = 0
    cambio_cantidad[mascara_D] = p.SENSIBILIDAD_AJUSTE

    # 1.3 Aplicar ajustes:
    # P(t+1) = P(t) * (1 + cambio)
    nuevos_precios = precios * (1 + cambio_precio)

    # D(t+1) = Y(t) * (1 + cambio)
    demanda_esperada = produccion * (1 + cambio_cantidad)

    # Cotas para evitar valores negativos
    nuevos_precios = np.maximum(nuevos_precios, 0.01)
    demanda_esperada = np.maximum(demanda_esperada, 0.1)

    # 1.4 Calcular demanda de trabajo:
    demanda_laboral = demanda_esperada / p.alpha
    factura_esperada_salarial = demanda_laboral * p.w_base

    return nuevos_precios, demanda_laboral, demanda_esperada, factura_esperada_salarial
