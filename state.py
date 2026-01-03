import numpy as np
from parameters import *

class EconomyState:
    """
    Mantiene el estado completo de la economía utilizando tensores (arrays de NumPy).
    Evitamos objetos individuales para Agentes; su estado está en el índice correspondiente de cada vector.
    """
    def __init__(self):
        # --- Sector Bancario (Dimension B) ---
        # Capital/Patrimonio de los bancos (Equity)
        self.bank_equity = np.full(N_BANKS, INITIAL_BANK_EQUITY, dtype=np.float64)
        # Liquidez disponible (Cash)
        self.bank_cash = np.zeros(N_BANKS, dtype=np.float64)
        
        # --- Sector Productivo (Dimension F) ---
        # Efectivo/Liquidez de las firmas
        self.firm_cash = np.full(N_FIRMS, INITIAL_FIRM_CASH, dtype=np.float64)
        # Precio estimado por cada firma P_i(t)
        self.firm_prices = np.ones(N_FIRMS, dtype=np.float64)
        # Demanda esperada D_i(t)
        self.firm_expected_demand = np.zeros(N_FIRMS, dtype=np.float64)
        # Stock de bienes producidos (Inventario)
        self.firm_stock = np.zeros(N_FIRMS, dtype=np.float64)
        
        # --- Variables de Flujo y Planificación (Paso 1) ---
        # Producción planeada Y_i(t)
        self.firm_planned_production = np.zeros(N_FIRMS, dtype=np.float64)
        # Demanda de trabajadores N_i(t)
        self.firm_labor_demand = np.zeros(N_FIRMS, dtype=np.float64)
        # Costo salarial total esperado
        self.firm_wage_bill = np.zeros(N_FIRMS, dtype=np.float64)
        # Demanda de crédito (si el efectivo no alcanza para salarios)
        self.firm_credit_demand = np.zeros(N_FIRMS, dtype=np.float64)
        
        # --- Sector Hogares (Dimension H) ---
        # Efectivo de los hogares
        self.household_cash = np.full(N_HOUSEHOLDS, INITIAL_HOUSEHOLD_CASH, dtype=np.float64)
        # Empleador actual de cada hogar (-1 = desempleado)
        self.household_employer = np.full(N_HOUSEHOLDS, -1, dtype=np.int32)
        
        # --- Matrices de Red (Grafos de Deuda) ---
        
        # Matriz de Préstamos Interbancarios L_ij (B x B)
        # Fila i (prestamista), Columna j (prestatario) -> Paper convención inversa: L_ij son pasivos de i hacia j.
        # Ajustaremos a la convención estándar de Adyacencia: 
        # A[i, j] = Préstamo del Banco i al Banco j (Activo para i, Pasivo para j)
        self.interbank_matrix = np.zeros((N_BANKS, N_BANKS), dtype=np.float64)
        
        # Matriz de Préstamos Bancos a Firmas (B x F)
        # A[i, j] = Préstamo del Banco i a la Firma j
        self.bank_firm_loans = np.zeros((N_BANKS, N_FIRMS), dtype=np.float64)
        
        # --- Asignación de Propiedad (Ownership) ---
        # Los hogares son dueños de firmas y bancos. 
        # Asignación aleatoria inicial.
        self.firm_owners = np.random.randint(0, N_HOUSEHOLDS, size=N_FIRMS)
        self.bank_owners = np.random.randint(0, N_HOUSEHOLDS, size=N_BANKS)

    def verify_consistency(self):
        """
        Verificación simple de tipos y dimensiones.
        """
        assert self.bank_equity.shape == (N_BANKS,)
        assert self.firm_cash.shape == (N_FIRMS,)
        assert self.household_cash.shape == (N_HOUSEHOLDS,)
        assert self.interbank_matrix.shape == (N_BANKS, N_BANKS)
        return True
