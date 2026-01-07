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

    # Parámetros necesarios para el Mercado Crediticio.
    n_bancos = 5  # numero de bancos que visita cada empresa.
    r_bar = 0.02  # Tasa de refinanciación bancaria.
    phi = 0.8  # Contracción de demanda si la tasa es alta.
    r_max = 0.10  # Tasa umbral máxima.

    # Parámetros implícitos para la función de fragilidad (Tangente Hiperbólica)
    # El paper dice que usa tanh, pero no da la escala exacta.
    # Usaremos una escala estándar para que la tanh no sature inmediatamente.
    SCALE_FRAGILITY = 1.0

    psi_max = 0.1  # Máxima especificidad del banco prrestamista (uniforme 0-psi_max).

    # Parámetros de impuestos:
    TAX_MODE = "SRT"  # Opcione: SRT, NONE, TOBIN.
    TOBIN_RATE = 0.002  # tasa fija.
    ZETA = 0.02  # factor de proporcionalidad para SRT.

    # Parámetros de hogares:
    c = 0.8  # Propensión al consumo.
    z = 2  # Número de empresas que compara cada hogarñ
