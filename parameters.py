
# Parámetros del Modelo (Tabla I - Poledna & Thurner, 2014)

# Dimensiones de la población
N_BANKS = 20          # B
N_FIRMS = 100         # F
N_HOUSEHOLDS = 1300   # H

# Parámetros Económicos
DIVIDENDS_SHARE = 0.2 # div
REFINANCING_RATE = 0.02 # r_bar
LABOR_PRODUCTIVITY = 0.1 # alpha
CREDIT_DEMAND_CONTRACTION = 0.8 # phi
DEBT_REIMBURSEMENT_RATE = 0.05 # tau
WAGE_RATE = 1.0       # w_b
PROPENSITY_TO_CONSUME = 0.8 # c

# Parámetros de Tasas e Intereses (Appendix A)
MAX_INTEREST_RATE = 0.20 # r_max: Tasa máxima que una firma acepta (implicita en texto)
# Para la función tangente hiperbólica de riesgo:
RISK_SENSITIVITY = 1.0 # factor de escala implícito para tanh(mu * leverage)

# Parámetros de Interacción de Mercado
N_CONSUMPTION_APPS = 2 # z
N_CREDIT_APPS = 5      # n

# Parámetros Mercado Interbancario y Tax
LIQUIDITY_BUFFER_RATIO = 0.10 # % de Activos (Loans) que se desea en Cash
SRT_SENSITIVITY = 0.5   # zeta ajustado. 1.0 puede ser alto para redes pequeñas. Probamos 0.5.
INTERBANK_LGD = 1.0     # Loss Given Default en interbancario (Asumimos pérdida total si quiebra)
INTERBANK_RATE = 0.005  # Tasa interbancaria base (muy baja, overnight)


# Configuración Inicial
INITIAL_FIRM_CASH = 1000.0
INITIAL_BANK_EQUITY = 2000.0
INITIAL_HOUSEHOLD_CASH = 100.0
