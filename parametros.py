# parametros.py

class Param:
    """
    Parámetros del Modelo CRISIS / SRT - ESCALA REAL (EUROS)
    Calibración: 1 Unidad de Tiempo = 1 Mes aprox.
    Moneda: Euros (€)
    """

    # --- Dimensiones del Sistema ---
    B = 20    # Bancos
    F = 100   # Empresas
    H = 1300  # Hogares
    T = 500   # Tiempo

    # --- FACTOR DE ESCALA (Solo para referencia mental) ---
    # K = 2000. Si antes W=1.0, ahora W=2000.0

    # --- Parámetros Económicos ---
    ALPHA = 0.1          # Productividad (Físico: 1 trabajador produce 0.1 coches/mes)
    W_BASE = 2200.0      # Salario Base Mensual en €
    
    # Precio Inicial: Margen sobre costes. 
    # Coste laboral unitario = W / ALPHA = 2200 / 0.1 = 22,000 €
    # Precio con margen = 24,000 €
    PRECIO_INICIAL = 24000.0 
    
    R_BAR = 0.02         # Tasa de interés (2%)
    DIVIDEND_RATIO = 0.2 # 20% de beneficios a dividendos
    PROPENSION_CONSUMO = 0.8 # Gastan el 80% de su sueldo

    # --- Parámetros de Crédito y Deuda ---
    TAU = 0.04           # Amortización de deuda (4% mensual)
    PHI = 0.8            # Restricción de crédito
    R_MAX = 0.25         # Tasa usura (>25%)

    # --- Inicialización de Stocks (En Euros) ---
    # Producción Física (Mantenemos escala pequeña para cantidades)
    # L_demand ~ 13 trabajadores -> Producción ~ 1.3 unidades
    PRODUCCION_INICIAL = 1.3 

    # Stocks Financieros (Escalados x2000 respecto a versión anterior)
    
    # Equity Firmas: Colchón para no quebrar día 1.
    EQUITY_INICIAL_FIRMAS = 40000.0 
    
    # Liquidez Firmas: Nómina de 1 mes aprox (13 emp * 2200€ = 28.600€)
    # Les damos menos para obligarlas a pedir crédito.
    LIQUIDEZ_INICIAL_FIRMAS = 5000.0 
    
    # Equity Bancos: Capital Base. Lo hacemos bajo para fragilidad.
    # Antes 20.0 -> Ahora 40.000 €
    EQUITY_INICIAL_BANCOS = 40000.0   
    
    # Liquidez Bancos: Dinero prestable. Escaso para forzar interbancario.
    # Antes 20.0 -> Ahora 40.000 €
    # [FIX] Aumentado a 150k para cubrir demanda de nóminas (2.8M total) -> Reducido a 30k para forzar IB
    LIQUIDEZ_INICIAL_BANCOS = 30000.0 

    # Depósitos Hogares: Ahorro inicial
    DEPOSITOS_INICIALES_HOGARES = 100000.0 

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
    ZETA = 0.02       # Sensibilidad SRT (Adimensional, no cambia)
    TASA_TOBIN = 0.002 # 0.2% (Adimensional, no cambia)