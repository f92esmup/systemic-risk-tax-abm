# Script para el paso 3 las firmas se preparan para producir. Y masa salarial.


import numpy as np
from parametros import Param as p


def paso3_produccion_y_mercado_laboral(
    demanda_trabajo_objetivo,
    firm_liquidez_actual,
    hogares_empleo_estado,
    hogares_es_trabajador,
):
    """
    Implementa la lógica de "Hire or Fire" del paper. Y estima la producción real.

    Args:
        demanda_trabajo_objetivo: (F,) Trabajadores ideales (calculado en Paso 1).
        firm_liquidez_actual: (F,) Liquidez disponible post-crédito.
        hogares_empleo_estado: (H,) Vector estado actual (-1: Paro, 0..F-1: ID Empresa).
        hogares_es_trabajador: (H,) Vector booleano, True si el hogar es obrero (no dueño).

    Returns:
        firm_produccion_real: (F,)
        firm_trabajadores_reales: (F,)
        firm_coste_salarial: (F,) Dinero que sale de cada empresa.
        ingresos_salariales_hogares: (H,) Dinero que recibe cada hogar individual.
        firm_liquidez_post: (F,)
        hogares_empleo_estado_nuevo: (H,) Estado actualizado tras despidos/contrataciones.
    """

    # 1. Recalcular restricción presupuestaria
    # cuantos pueden pagar con su liquidez actual?
    capacidad_contratacion = np.floor(firm_liquidez_actual / (p.w_base + 1e-9)).astype(
        int
    )

    # La demanda real está acotada por lo financiero
    target_trabajadores = np.minimum(
        demanda_trabajo_objetivo, capacidad_contratacion
    ).astype(int)

    # 2- Estado Obtener estado actual de la fuerza laboral.
    # Contamos cuantos empleados tiene cada empresa actualmente.
    # (bincount cuenta ocurrecias de 0 a F-1, ignorando -1 con un offset)

    # Filtramos solos los hogares que están empleados (>=0)
    empleados_activos = hogares_empleo_estado[hogares_empleo_estado >= 0]

    if len(empleados_activos) > 0:
        conteo_actual = np.bincount(empleados_activos, minlength=p.F)
    else:
        conteo_actual = np.zeros(p.F, dtype=int)

    # Diferencia: ¿Necesito contratar o despedir?
    delta_plantilla = target_trabajadores - conteo_actual

    # Creamos una copia del estado para modificarlo
    nuevo_estado_empleo = hogares_empleo_estado.copy()

    # 3. Bucle de Ajuste de Plantilla
    # Identificamos indices de hogares disponibles (Desempleados Y que sean Trabajadores)
    # Ojo: Los dueños (hogares_es_trabajador == False) no pueden ser contratados.

    for f_idx in range(p.F):
        cambio = delta_plantilla[f_idx]

        if cambio == 0:
            continue

        elif cambio < 0:
            # --- DESPIDOS (FIRE) ---
            num_despedir = abs(cambio)
            # Buscamos quiénes trabajan aquí
            mis_empleados = np.where(nuevo_estado_empleo == f_idx)[0]

            # Seleccionamos aleatoriamente a quién despedir
            if len(mis_empleados) > 0:
                # Protegemos por si hay error de conteo, no despedir más de los que hay
                num_real_despedir = min(len(mis_empleados), num_despedir)
                despedidos = np.random.choice(
                    mis_empleados, size=num_real_despedir, replace=False
                )

                # Pasan al paro (-1)
                nuevo_estado_empleo[despedidos] = -1

        elif cambio > 0:
            # --- CONTRATACIONES (HIRE) ---
            num_contratar = cambio
            # Buscamos desempleados que sean aptos (Workers)
            # Condición: (Estado == -1) AND (Es_Trabajador == True)
            mask_disponibles = (nuevo_estado_empleo == -1) & hogares_es_trabajador
            indices_disponibles = np.where(mask_disponibles)[0]

            if len(indices_disponibles) > 0:
                # Contratamos hasta donde haya oferta laboral
                num_real_contratar = min(len(indices_disponibles), num_contratar)
                nuevos_fichajes = np.random.choice(
                    indices_disponibles, size=num_real_contratar, replace=False
                )

                # Asignamos ID empresa
                nuevo_estado_empleo[nuevos_fichajes] = f_idx
            else:
                # No hay trabajadores disponibles (Labor Shortage)
                # La empresa produce menos de lo planeado por falta de mano de obra
                target_trabajadores[f_idx] -= (
                    num_contratar  # Ajustamos el target real conseguido
                )
    # 4. Producción y Pagos
    # Ahora target_trabajadores refleja la plantilla real tras hiring/firing/shortages

    # Producción Y = alpha * L
    firm_produccion_real = p.alpha * target_trabajadores

    # Coste Salarial (Dinero que sale de la empresa)
    firm_coste_salarial = target_trabajadores * p.w_base
    firm_liquidez_post = firm_liquidez_actual - firm_coste_salarial

    # Ingresos Salariales (Dinero que entra al hogar específico)
    ingresos_salariales_hogares = np.zeros(p.H)

    # Asignamos salario a todos los que tienen empleo (estado >= 0)
    # Usamos vectorización directa:
    mask_empleados = nuevo_estado_empleo >= 0
    ingresos_salariales_hogares[mask_empleados] = (
        p.w_base
    )  # Esto representa la nómina de ese mes actual.

    return (
        firm_produccion_real,
        target_trabajadores,
        firm_coste_salarial,
        ingresos_salariales_hogares,
        firm_liquidez_post,
        nuevo_estado_empleo,
    )
