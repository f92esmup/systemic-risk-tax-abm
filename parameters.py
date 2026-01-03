
# Parámetros del Modelo (Tabla I - Poledna & Thurner, 2014)

# Dimensiones de la población
N_BANKS = 20          # B
N_FIRMS = 100         # F
N_HOUSEHOLDS = 1300   # H

# Parámetros Económicos
DIVIDENDS_SHARE = 0.2 # div: Porcentaje de beneficios pagados como dividendos
REFINANCING_RATE = 0.02 # r_bar: Tasa de refinanciación general
LABOR_PRODUCTIVITY = 0.1 # alpha: Productividad laboral
CREDIT_DEMAND_CONTRACTION = 0.8 # phi: Factor de contracción de demanda de crédito
DEBT_REIMBURSEMENT_RATE = 0.05 # tau: Tasa de reembolso de deuda por paso
WAGE_RATE = 1.0       # w_b: Salario base
PROPENSITY_TO_CONSUME = 0.8 # c: Propensión al consumo

# Parámetros de Interacción de Mercado
N_CONSUMPTION_APPS = 2 # z: Número de firmas que consulta un hogar para comprar
N_CREDIT_APPS = 5      # n: Número de bancos que consulta una firma/banco para crédito

# Configuración Inicial (Valores por defecto razonables para iniciar la simulación)
INITIAL_FIRM_CASH = 1000.0
INITIAL_BANK_EQUITY = 2000.0
INITIAL_HOUSEHOLD_CASH = 100.0
