import numpy as np
from parametros import Param as p


def paso3(
    labor_hired,  # Vector (F,) Trabajadores realmente contratados (input de paso 2)
    wages_paid_total,  # Vector (F,) Masa salarial total pagada (input de paso 2)
    inventario_previo,  # Vector (F,)
):
    """
    Paso 3: Producción Física (Adaptado a Vectorización)

    Nota: La contratación y pago de salarios ahora ocurre al final del Paso 2
    (Mercado de Crédito), por lo que este paso solo ejecuta la función de producción.
    """

    # 1. Producción Física
    # Función de producción lineal: Y = alpha * N
    produccion_nueva = labor_hired * p.ALPHA

    # 2. Oferta Total
    oferta_total_bienes = inventario_previo + produccion_nueva

    # 3. Generar Matriz de Flujo Salarial (F -> H)
    # Distribuir la masa salarial pagada (wages_paid_total) a los hogares.
    # Usamos la lógica determinista "banded" para visualización consistente.

    wages_matrix = np.zeros((p.F, p.H))

    # Numero de empleados "graficos" por empresa
    k_employees = max(1, p.H // p.F)

    # Vectorización del loop de asignación de salarios
    # Crear índices para broadcasting
    # F filas, k cols -> indices en H
    # indices[f, k] = (f * K + k) % H

    # Expandimos índices: (F, K)
    f_indices = np.arange(p.F)[:, np.newaxis]
    k_offsets = np.arange(k_employees)[np.newaxis, :]

    h_indices = (f_indices * k_employees + k_offsets) % p.H

    # Valores a asignar: WageBill[f] / K
    # wage_values = (wages_paid_total / k_employees)[:, np.newaxis]

    # Asignación flat
    # wages_matrix[f, h] = value
    # Usamos np.add.at en caso de colisiones (aunque aquí es disjunto por diseño si H >= F*K)
    # Pero para seguridad vectorizada simple, iteramos o flat asign.
    # Dado que es puramente visual/accounting, un loop simple es aceptable por claridad,
    # pero vectorizado es mejor.

    # Flatten arrays
    rows = np.repeat(np.arange(p.F), k_employees)
    cols = h_indices.flatten()
    vals = np.repeat(wages_paid_total / k_employees, k_employees)

    # Matriz dispersa densa
    wages_matrix[rows, cols] = vals

    # Comprobar consistencia
    # assert np.isclose(np.sum(wages_matrix), np.sum(wages_paid_total))

    return (produccion_nueva, oferta_total_bienes, wages_matrix)

