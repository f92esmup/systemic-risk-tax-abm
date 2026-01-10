import numpy as np
from parametros import Param as p


def paso4(
    precios_actuales,  # Vector (F,) Precios fijados en Paso 1
    oferta_total_bienes,  # Vector (F,) Inventario + Producción (Paso 3)
    factura_salarial_real,  # Escalar o Vector (F,): Dinero total pagado en salarios
    depositos_hogares,  # Vector (H,) Ahorros acumulados de los hogares
    dividendos_previos,  # Vector (H,) Dividendos recibidos en t-1 (Paso 5 previo)
):
    """
    Paso 4: Mercado de Bienes y Consumo.

    1. Hogares reciben ingresos (Salarios + Dividendos).
    2. Hogares definen consumo deseado (Budget).
    3. Selección de proveedores (Regla 'z' aleatoria).
    4. Ejecución de compras y actualización de inventarios.

    [cite_start]Ref: [cite: 185, 210, 211, 219]
    """

    H = p.H
    F = p.F

    # -------------------------------------------------------------------------
    # 1. Ingresos de los Hogares
    # -------------------------------------------------------------------------
    # [cite_start]"Households receive wages" [cite: 191]
    # Sumamos toda la masa salarial pagada por las empresas y la repartimos.
    # (Simplificación válida: asumimos pleno empleo o reparto solidario en el sector hogares)
    total_salarios = np.sum(factura_salarial_real)
    ingreso_salarial_per_capita = total_salarios / H

    # Riqueza Disponible = Ahorro previo + Salarios + Dividendos
    riqueza_hogares = (
        depositos_hogares + ingreso_salarial_per_capita + dividendos_previos
    )

    # -------------------------------------------------------------------------
    # 2. Presupuesto de Consumo
    # -------------------------------------------------------------------------
    # [cite_start]"At each time step every household spends a fixed percentage c" [cite: 210]
    presupuesto_consumo = riqueza_hogares * p.PROPENSION_CONSUMO

    # -------------------------------------------------------------------------
    # 3. Selección de Vendedores (Search Process)
    # -------------------------------------------------------------------------
    # [cite_start]"Households compare prices from z randomly chosen firms and buy the cheapest" [cite: 211]

    # Cada hogar ve 'z' empresas aleatorias
    indices_firmas_vistas = np.random.randint(0, F, size=(H, p.Z_CONSUMO), dtype=np.int64)

    # Extraer precios correspondientes
    precios_vistas = precios_actuales[indices_firmas_vistas]

    # Encontrar la opción más barata para cada hogar
    idx_min_local = np.argmin(precios_vistas, axis=1)  # 0..z-1

    # Mapear de vuelta al índice real de la empresa (0..F-1)
    firmas_elegidas = indices_firmas_vistas[np.arange(H), idx_min_local]

    # -------------------------------------------------------------------------
    # 4. Agregación de Demanda
    # -------------------------------------------------------------------------
    # Sumar todo el dinero dirigido a cada empresa
    demanda_monetaria = np.bincount(
        firmas_elegidas, weights=presupuesto_consumo, minlength=F
    )

    # Q_demandada = Dinero / Precio
    demanda_cantidad_teorica = np.divide(
        demanda_monetaria,
        precios_actuales,
        out=np.zeros_like(demanda_monetaria),
        where=precios_actuales != 0,
    )

    # -------------------------------------------------------------------------
    # 5. Transacción Real (Stock Constraint)
    # -------------------------------------------------------------------------
    # Venta Real = min(Lo que piden, Lo que tengo)
    ventas_cantidad_real = np.minimum(demanda_cantidad_teorica, oferta_total_bienes)

    # Ingresos (Revenue) para la empresa
    ingresos_ventas = ventas_cantidad_real * precios_actuales

    # Actualizar Inventario (Stock no vendido)
    inventario_final = oferta_total_bienes - ventas_cantidad_real
    # Evitar negativos por error flotante
    inventario_final = np.maximum(inventario_final, 0.0)

    # -------------------------------------------------------------------------
    # 6. Actualización de Hogares (Racionamiento Agregado)
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # 6. Actualización de Hogares (Racionamiento Específico)
    # -------------------------------------------------------------------------
    # [Refactor] Ahora el racionamiento es local: Si tu tienda no tiene, no gastas.
    
    # Ratio de satisfacción por empresa (0.0 a 1.0)
    # ratio_j = Ventas_j / Demanda_j
    
    # Manejo seguro de division por cero (si demanda es 0, ratio es 1.0 logicamente)
    ratio_satisfaccion_firma = np.divide(
        ventas_cantidad_real,
        demanda_cantidad_teorica,
        out=np.ones_like(ventas_cantidad_real), # Default 1.0
        where=demanda_cantidad_teorica > 1e-9
    )
    
    # Mapear el ratio de la empresa elegida a cada hogar
    # Hago lookup usando el índice de la firma
    ratio_satisfaccion_hogar = ratio_satisfaccion_firma[firmas_elegidas]
    
    # Gasto efectivo por hogar
    gasto_efectivo_hogar = presupuesto_consumo * ratio_satisfaccion_hogar

    # Nueva riqueza = Riqueza inicial - Gasto
    depositos_finales = riqueza_hogares - gasto_efectivo_hogar

    # Nota de consistencia: sum(depositos_finales) + sum(ingresos_ventas)
    # debería ser igual a sum(riqueza_hogares) [SFC Check]
    # Check de consistencia Stock-Flow (SFC)
    # El dinero que tenían los hogares (riqueza) debe estar ahora en depositos_finales o en ingresos_ventas
    assert np.isclose(
        np.sum(depositos_finales) + np.sum(ingresos_ventas), np.sum(riqueza_hogares)
    ), "Error SFC: El dinero se ha creado o destruido en el intercambio."

    return (
        ventas_cantidad_real,  # Q vendida
        ingresos_ventas,  # Revenue
        inventario_final,  # Stock t+1
        depositos_finales,  # Ahorro t+1
        demanda_cantidad_teorica,  # Dato para expectativas (D_expected)
        ingreso_salarial_per_capita,  # Dato informativo (opcional)
    )
