# Script para centralizar los parámetros de la simulación.


class Param:
    """Param almacena las constantes y parámetros de la simulación."""

    # Definimos el número de agentes.:
    B = 20  # Número de bancos
    F = 100  # Número de empresas
    H = 1300  # Número de hogares

    alpha = 0.1  # Productividad laboral
    w_base = 1.0  # Tasa salarial base.

    SENSIBILIDAD_AJUSTE = 0.05  # Sensibilidad de ajuste de precios y cantidades.
