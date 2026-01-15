class Param:
    """
    Parámetros del Modelo CRISIS / SRT - ESCALA REAL (EUROS)
    Calibración: 1 Unidad de Tiempo = 1 Mes aprox.
    Moneda: Euros (€)
    """

    # --- Dimensiones del Sistema ---
    B = 20  # Bancos
    F = 100  # Empresas
    H = 1300  # Hogares
    T = 200  # Tiempo

    # --- FACTOR DE ESCALA (Solo para referencia mental) ---
    # K = 2000. Si antes W=1.0, ahora W=2000.0

    # --- Parámetros Económicos ---
    ALPHA = 0.1  # Productividad (Físico: 1 trabajador produce 0.1 coches/mes)
    W_BASE = 2200.0  # Salario Base Mensual en €

    # Precio Inicial: Margen sobre costes.
    # Coste laboral unitario = W / ALPHA = 2200 / 0.1 = 22,000 €
    # Precio con margen = 35,000 € (Margen MASIVO para robustez extrema)
    # Aumentamos precio significativamente para asegurar márgenes altos a las empresas
    # Coste ~22k. Precio 35k -> Margen ~37%.
    # Esto reduce drásticamente la probabilidad de default inicial.
    PRECIO_INICIAL = 35000.0

    # --- Ajuste de Estabilidad con DIVIDENDOS ALTOS ---
    R_BAR = (
        0.015  # 1.5% mensual. Equilibrio entre ingresos bancarios y carga a empresas.
    )
    DIVIDEND_RATIO = 0.20  # <--- REQUERIMIENTO: 20% (Volvemos al valor del Paper)
    PROPENSION_CONSUMO = 0.9  # Alto consumo para mantener flujo de caja en empresas

    # --- Parámetros de Crédito y Deuda ---
    TAU = 0.01  # Amortización de deuda (1% mensual) - Ajuste: Lenta para reducir salida de caja
    PHI = 0.8  # Restricción de crédito
    R_MAX = 0.25  # Tasa usura (>25%)

    # --- Inicialización de Stocks (En Euros) ---
    # Producción Física (Mantenemos escala pequeña para cantidades)
    # L_demand ~ 13 trabajadores -> Producción ~ 1.3 unidades
    PRODUCCION_INICIAL = 1.3

    # Stocks Financieros (ESCENARIO "RICH SYSTEM")
    # Para sobrevivir al drenaje de dividendos (20%), necesitamos stocks masivos.

    # Equity Firmas: Colchón para no quebrar día 1.
    EQUITY_INICIAL_FIRMAS = 500000.0  # Antes 200k

    # Liquidez Firmas: Nómina de 1 mes aprox (13 emp * 2200€ = 28.600€)
    # Les damos menos para obligarlas a pedir crédito.
    LIQUIDEZ_INICIAL_FIRMAS = 200000.0  # Antes 100k

    # Equity Bancos: Capital Base. Lo hacemos bajo para fragilidad.
    # Equity Bancario x10 respecto a pruebas anteriores para soportar payout de 20%
    EQUITY_INICIAL_BANCOS = 2000000.0  # 2 Millones

    # Liquidez Bancos: Dinero prestable. Escaso para forzar interbancario.
    # Aumentado a 50k para sostener ciclos largos (t>200).
    LIQUIDEZ_INICIAL_BANCOS = 1000000.0  # 1 Millon

    # Depósitos Hogares: Ahorro inicial
    DEPOSITOS_INICIALES_HOGARES = 500000.0

    # --- Parámetros de Heterogeneidad ---
    SIGMA_SIZE = 0.8  # Dispersión Log-Normal para tamaños (Equity, Producción)
    SPREAD_PRICE = 0.05  # Dispersión Uniforme para precios iniciales (±5%)

    # --- Parámetros de Red y Búsqueda ---
    N_BANCOS_CONTACTADOS = 5
    Z_CONSUMO = 2

    # --- Parámetros del Mercado de Crédito ---
    FACTOR_PROB_DEFAULT = 0.01
    RANGO_PSI = 0.1

    # [IMPORTANTE] El préstamo de test debe ser relevante en Euros
    # Antes 1.0 -> Ahora 2000.0 € (Un micro-préstamo interbancario)
    DELTA_LOAN_TEST = 2000.0

    PROB_BAILOUT = 0.5

    # --- Parámetros de Ajuste Adaptativo ---
    RANGO_AJUSTE_MIN = 0.01
    RANGO_AJUSTE_MAX = 0.02

    # Umbrales (Ajustados a la escala de Cantidad vs Precio)
    # Inventario es FÍSICO (unidades), así que 1e-4 sigue bien.
    UMBRAL_INVENTARIO = 1e-4
    SUELO_PRECIO_RELATIVO = 0.5
    SUELO_DEMANDA_RELATIVO = 0.1

    # --- Impuestos ---
    ZETA = 0.02  # Sensibilidad SRT (Internalización Completa - Artículo 1)
    TASA_TOBIN = 0.002  # 0.2% (Adimensional, no cambia)

    # --- Configuración ---
    MODO_IMPUESTO = "NINGUNO"
