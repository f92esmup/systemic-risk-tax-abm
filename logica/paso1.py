# logica/paso1.py
import numpy as np
from parametros import Param as p


def paso1(precios, produccion, ventas):
    """
    Calcula la demanda esperada, precio nuevo y demanda de trabajadores.

    Referencia: Delli Gatti et al. [69], Gualdi et al. [70], Poledna et al. [167].
    Firms define labour and capital demand[cite: 188].
    """

    # --- 1. Definición de Estado ---
    avg_precio = np.mean(precios)
    precio_relativo = precios > avg_precio

    # Es crucial definir si hubo ventas totales o sobró stock
    # Usamos un epsilon para float comparison, pero cuidado con el stock acumulado
    exceso_inventario = (produccion - ventas) > 1e-5

    # --- 2. Matriz de Sensibilidad Estocástica ---
    # Para evitar sincronización, cada empresa tiene una reactividad ligeramente distinta
    # en este paso de tiempo.
    # U(-0.02, 0.02) + SENSIBILIDAD
    ruido = np.random.uniform(0.8, 1.2, p.F)
    ajuste_base = p.SENSIBILIDAD_AJUSTE * ruido

    # Inicializamos vectores de cambio
    cambio_precio = np.zeros(p.F)
    cambio_cantidad = np.zeros(p.F)

    # --- 3. Lógica de Ajuste (Cuadrantes Delli Gatti) ---

    # Caso A: Precio Alto + Stock Sobrante -> Bajar Precio, Bajar Cantidad
    mask_A = precio_relativo & exceso_inventario
    cambio_precio[mask_A] = -ajuste_base[mask_A]
    cambio_cantidad[mask_A] = -ajuste_base[mask_A]

    # Caso B: Precio Bajo + Sin Stock (Venta total) -> Subir Precio, Subir Cantidad
    mask_B = (~precio_relativo) & (~exceso_inventario)
    cambio_precio[mask_B] = ajuste_base[
        mask_B
    ]  # A veces se usa un ajuste más agresivo aquí
    cambio_cantidad[mask_B] = ajuste_base[mask_B]

    # Caso C: Precio Bajo + Stock Sobrante -> Mantener Precio, Bajar Cantidad
    # Si eres barato y no vendes, el problema es la demanda agregada, no tu precio.
    mask_C = (~precio_relativo) & exceso_inventario
    cambio_precio[mask_C] = 0.0
    cambio_cantidad[mask_C] = -ajuste_base[mask_C]

    # Caso D: Precio Alto + Sin Stock -> Mantener Precio (o subir poco), Subir Cantidad
    mask_D = precio_relativo & (~exceso_inventario)
    cambio_precio[mask_D] = 0.0  # O subir levemente 0.5 * ajuste
    cambio_cantidad[mask_D] = ajuste_base[mask_D]

    # --- 4. Aplicación de Ajustes y Anclaje ---

    nuevos_precios = precios * (1 + cambio_precio)

    # CORRECCIÓN CRÍTICA: Base de la demanda esperada
    # Si sobró inventario, la base para calcular el futuro son las VENTAS, no la producción.
    # Si faltó inventario (ventas == produccion), la base es la PRODUCCIÓN.

    base_demanda = np.where(exceso_inventario, ventas, produccion)

    # Para evitar que una empresa con 0 ventas se quede en 0 para siempre,
    # añadimos un "piso" de reactivación aleatoria o un mínimo base.
    base_demanda = np.maximum(base_demanda, 0.1)

    demanda_esperada = base_demanda * (1 + cambio_cantidad)

    # --- 5. Cotas de Seguridad (Guardrails) ---
    # Evitar precios negativos o cero
    nuevos_precios = np.maximum(nuevos_precios, 0.01)

    # Evitar explosiones o implosiones numéricas
    demanda_esperada = np.maximum(demanda_esperada, 0.1)

    # --- 6. Demanda de Factores ---
    # Labour demand = Desired Production / Productivity [cite: 208, 214]
    demanda_laboral = demanda_esperada / p.alpha

    # Coste estimado (Budget needed)
    factura_esperada_salarial = demanda_laboral * p.w_base

    return nuevos_precios, demanda_laboral, demanda_esperada, factura_esperada_salarial
