# parametros.py


class Param:
    """
    Parámetros del Modelo CRISIS / SRT
    Valores extraídos de Table I del paper [cite: 713-736]
    """

    # --- Dimensiones del Sistema ---
    B = 20  # Número de Bancos [cite: 715]
    F = 100  # Número de Empresas [cite: 718]
    H = 1300  # Número de Hogares [cite: 720]
    T = 20  # Pasos de tiempo de la simulación [cite: 258]

    # --- Parámetros Económicos ---
    ALPHA = 0.1  # Productividad laboral [cite: 726]
    W_BASE = 1.0  # Tasa salarial [cite: 732]
    R_BAR = 0.02  # Tasa de interés de referencia (Refinancing rate) [cite: 724]
    DIVIDEND_RATIO = 0.2  # Porcentaje de beneficios pagados en dividendos [cite: 722]
    PROPENSION_CONSUMO = 0.8  # 'c' Propensity to consume [cite: 734]

    # --- Parámetros de Crédito y Deuda ---
    TAU = 0.05  # Tasa de reembolso de deuda (Rate of debt reimbursement) [cite: 730]
    PHI = 0.8  # Contracción de demanda de crédito si r > r_max [cite: 728]
    R_MAX = 0.25  # [Tuning] Aumentado umbral para tolerar tasas más altas al inicio


    # --- Parámetros de Red y Búsqueda ---
    N_BANCOS_CONTACTADOS = 5  # 'n' Number of applications in credit market [cite: 736]
    Z_CONSUMO = 2  # 'z' Number of applications in consumption goods market [cite: 733]

    # --- Parámetros del Mercado de Crédito (Paso 2) ---
    FACTOR_PROB_DEFAULT = 0.01  # Escalar para proxy de riesgo (Eq. A4)
    RANGO_PSI = 0.1             # Variabilidad idiosincrática interbancaria (0 a 0.1)
    DELTA_LOAN_TEST = 1.0       # Monto del préstamo hipotético para cálculo SRT
    PROB_BAILOUT = 0.5          # Probabilidad de rescate (bailout) de firma [cite: Paper 2]



    # --- Parámetros de Ajuste Adaptativo (Paso 1) ---
    # Rango de ajuste aleatorio para precios y cantidades (Greenwald-Stiglitz)
    RANGO_AJUSTE_MIN = 0.01
    RANGO_AJUSTE_MAX = 0.02
    
    # Parámetros de Estabilidad Numérica (Paso 1)
    UMBRAL_INVENTARIO = 1e-4        # Nivel mínimo para considerar exceso de stock
    SUELO_PRECIO_RELATIVO = 0.5     # Precio mínimo como fracción del promedio
    SUELO_DEMANDA_RELATIVO = 0.1    # Demanda mínima como fracción del promedio

    # --- Parámetros de Impuestos (Policy) ---
    ZETA = 0.02  # Sensibilidad del impuesto SRT (o 1.0 para full pricing) [cite: 259]
    TASA_TOBIN = 0.002  # 0.2% tasa fija [cite: 250]

    # --- Inicialización (Valores Semilla) ---
    PRECIO_INICIAL = 12.0  # Debe ser > W_BASE/ALPHA = 10.0
    PRODUCCION_INICIAL = 1.3  # Ajustado: 1300H / 100F * ALPHA(0.1) = 1.3
    EQUITY_INICIAL_FIRMAS = 100.0 # Aumentado para absorber choques iniciales
    LIQUIDEZ_INICIAL_FIRMAS = 100.0


    EQUITY_INICIAL_BANCOS = 5000.0   
    LIQUIDEZ_INICIAL_BANCOS = 5000.0 

    DEPOSITOS_INICIALES_HOGARES = 50.0

