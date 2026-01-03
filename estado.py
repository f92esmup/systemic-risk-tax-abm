import numpy as np
from parametros import *

class EstadoEconomia:
    """
    Mantiene el estado completo de la economía utilizando tensores (arrays de NumPy).
    Evitamos objetos individuales para Agentes; su estado está en el índice correspondiente de cada vector.
    """
    def __init__(self):
        # --- Sector Bancario (Dimension B) ---
        # Capital/Patrimonio de los bancos (Equity)
        self.patrimonio_bancos = np.full(N_BANCOS, PATRIMONIO_INICIAL_BANCO, dtype=np.float64)
        # Liquidez disponible (Cash) - Inicialmente es todo el Patrimonio
        self.efectivo_bancos = self.patrimonio_bancos.copy()
        
        # --- Sector Productivo (Dimension F) ---
        # Efectivo/Liquidez de las empresas
        self.efectivo_empresas = np.full(N_EMPRESAS, EFECTIVO_INICIAL_EMPRESA, dtype=np.float64)
        # Precio estimado por cada empresa P_i(t)
        self.precios_empresas = np.ones(N_EMPRESAS, dtype=np.float64)
        # Demanda esperada D_i(t)
        self.demanda_esperada_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)
        # Stock de bienes producidos (Inventario)
        self.stock_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)
        
        # --- Variables de Flujo y Planificación (Paso 1) ---
        # Producción planeada Y_i(t)
        self.produccion_planeada_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)
        # Demanda de trabajadores N_i(t)
        self.demanda_trabajo_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)
        # Costo salarial total esperado (Masa Salarial)
        self.masa_salarial_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)
        # Demanda de crédito (si el efectivo no alcanza para salarios)
        self.demanda_credito_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)
        
        # --- Sector Hogares (Dimension H) ---
        # Efectivo de los hogares
        self.efectivo_hogares = np.full(N_HOGARES, EFECTIVO_INICIAL_HOGAR, dtype=np.float64)
        # Empleador actual de cada hogar (-1 = desempleado)
        self.empleador_hogares = np.full(N_HOGARES, -1, dtype=np.int32)
        
        # --- Matrices de Red (Grafos de Deuda) ---
        
        # Matriz de Préstamos Interbancarios A_ij (B x B)
        # A[i, j] = Préstamo del Banco i al Banco j (Activo para i, Pasivo para j)
        self.matriz_interbancaria = np.zeros((N_BANCOS, N_BANCOS), dtype=np.float64)
        
        # Matriz de Préstamos Bancos a Empresas (B x F) Activos para bancos
        self.prestamos_banco_empresa = np.zeros((N_BANCOS, N_EMPRESAS), dtype=np.float64)
        
        # Tasas de interés actuales cobradas a cada empresa (B x F)
        # Se actualizan cuando se emite un nuevo préstamo.
        self.tasas_interes_prestamos = np.zeros((N_BANCOS, N_EMPRESAS), dtype=np.float64)
        
        # --- Variables de Flujo Paso 2 (Mercado Crédito) ---
        # Solicitudes de crédito aceptadas en este paso (para verificación)
        self.nuevos_prestamos_otorgados = np.zeros(N_EMPRESAS, dtype=np.float64)
        # Banco prestamista escogido para el nuevo préstamo (-1 = ninguno)
        self.eleccion_prestamista_empresa = np.full(N_EMPRESAS, -1, dtype=np.int32)
        
        # --- Variables de Flujo Paso 5 (Repago) ---
        # Deuda incobrable (bad debt) reconocida por los bancos en este paso
        self.deuda_incobrable_bancos = np.zeros(N_BANCOS, dtype=np.float64)
        # Contador de defaults acumulados por empresa (opcional para estadísticas)
        self.defaults_acumulados_empresas = np.zeros(N_EMPRESAS, dtype=np.int32)
        
        # --- Variables de Flujo Paso 6 (Interbancario) ---
        # Nivel de Riesgo Sistémico (DebtRank promedio o total) del paso actual
        self.riesgo_sistemico_total = 0.0
        # Impuesto sistémico recaudado en este paso
        self.impuesto_recaudado = 0.0
        
        # --- Asignación de Propiedad (Ownership) ---
        # Los hogares son dueños de empresas y bancos. 
        # Asignación aleatoria inicial.
        self.duenos_empresas = np.random.randint(0, N_HOGARES, size=N_EMPRESAS)
        self.duenos_bancos = np.random.randint(0, N_HOGARES, size=N_BANCOS)
        
        # --- Variables adicionales para seguimiento (Paso 7 - Evolucion) ---
        # Ventas diarias para calcular ingresos
        self.ventas_diarias_empresas = np.zeros(N_EMPRESAS, dtype=np.float64)


    def verificar_consistencia(self):
        """
        Verificación simple de tipos y dimensiones.
        """
        assert self.patrimonio_bancos.shape == (N_BANCOS,)
        assert self.efectivo_empresas.shape == (N_EMPRESAS,)
        assert self.efectivo_hogares.shape == (N_HOGARES,)
        assert self.matriz_interbancaria.shape == (N_BANCOS, N_BANCOS)
        return True
