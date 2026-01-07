# Script principal. Orquestador de la simulación del Modelo CRISIS.
# Cubre inicialización y Pasos 1, 2 y 3.

import numpy as np
from parametros import Param as p

# Importación de Módulos de Lógica
from logica.paso1 import paso1
from logica.paso2 import paso2_mercado_credito, paso2_interbancario

# Importamos la función con el nombre correcto que definiste en tu archivo paso3.py
from logica.paso3 import paso3_produccion_y_mercado_laboral

# Semilla para reproducibilidad
np.random.seed(42)

# ==========================================
# 0. INICIALIZACIÓN DEL ESTADO (t-1)
# ==========================================
print("--- [INIT] Inicializando Simulación (t-1) ---")

# --- 0.1 Variables de Empresas (Firms) ---
firm_ids = np.arange(p.F)
firm_prices = np.random.uniform(0.9, 1.1, p.F)  # P_i(t-1)
firm_produccion = np.random.uniform(8, 12, p.F)  # Y_i(t-1)
firm_ventas = firm_produccion * np.random.uniform(0.8, 1.0, p.F)  # S_i(t-1)
firm_liquidez = np.random.uniform(100, 200, p.F)  # L_i(t-1)
firm_deuda = np.random.uniform(0, 50, p.F)  # Deuda externa inicial

# --- 0.2 Variables de Bancos (Banks) ---
bancos_ids = np.arange(p.B)
bancos_liquidez = np.random.uniform(200, 1000, p.B)
bancos_patrimonio = np.random.uniform(50, 100, p.B)  # Equity (C_j)
bancos_depositos = np.random.uniform(500, 2000, p.B)  # Depósitos de clientes

# Deuda interbancaria inicial (Matriz L)
matriz_interbancaria_anterior = np.zeros((p.B, p.B))
# Generamos algunas deudas iniciales para probar DebtRank
mask_deuda = np.random.rand(p.B, p.B) > 0.8
matriz_interbancaria_anterior[mask_deuda] = np.random.uniform(1, 10, np.sum(mask_deuda))
np.fill_diagonal(matriz_interbancaria_anterior, 0)
bancos_deuda_acumulada = np.sum(matriz_interbancaria_anterior, axis=1)

# --- 0.3 Variables de Hogares (Fuerza Laboral) ---
hogares_ids = np.arange(p.H)
hogares_liquidez = np.random.uniform(10, 50, p.H)

# Definir Roles: Dueños vs Trabajadores (Para evitar contratar dueños en Paso 3)
# Asumimos: 0..F-1 dueños firmas, F..F+B-1 dueños bancos. Resto obreros.
hogares_es_trabajador = np.ones(p.H, dtype=bool)
indices_duenos = np.arange(p.F + p.B)
hogares_es_trabajador[indices_duenos] = False  # Los dueños NO buscan empleo

# Estado de Empleo Inicial (-1: Desempleado, 0..F-1: ID Empresa)
hogares_empleo_estado = np.full(p.H, -1, dtype=int)

# Asignación inicial aleatoria (90% de ocupación inicial para evitar shock de demanda)
indices_obreros = np.where(hogares_es_trabajador)[0]
num_ocupados_inicial = int(len(indices_obreros) * 0.90)
obreros_activos = np.random.choice(
    indices_obreros, size=num_ocupados_inicial, replace=False
)
empresas_asignadas = np.random.randint(0, p.F, size=num_ocupados_inicial)

# Registramos quién trabaja para quién (Persistencia necesaria para Paso 3)
hogares_empleo_estado[obreros_activos] = empresas_asignadas

print(
    f"   > Empresas: {p.F}, Bancos: {p.B}, Hogares: {p.H} ({len(indices_obreros)} trabajadores)"
)


# ==========================================
# PASO 1: PLANIFICACIÓN DE EMPRESAS
# ==========================================
print("\n>>> EJECUTANDO PASO 1: Planificación (Precios y Cantidades)...")

nuevos_precios, demanda_esperada, demanda_trabajo, factura_esperada_salarial = paso1(
    firm_prices, firm_produccion, firm_ventas
)

# --- Verificación Paso 1 ---
avg_p = np.mean(firm_prices)
# Chequeo: Empresa con inventario y precio alto debería bajar precio
mask_test = ((firm_produccion - firm_ventas) > 1e-5) & (firm_prices > avg_p)
if np.any(mask_test):
    idx = np.where(mask_test)[0][0]
    print(
        f"   [Check Lógica] Empresa {idx} (Stock alto/Cara): Precio {firm_prices[idx]:.2f} -> {nuevos_precios[idx]:.2f}"
    )


# ==========================================
# PASO 2: MERCADO DE CRÉDITO
# ==========================================
print("\n>>> EJECUTANDO PASO 2: Mercado de Crédito y Bancos...")

# 2.1 Empresas piden a Bancos (Matching Empresas-Bancos)
contratos_potenciales, demanda_credito_empresas = paso2_mercado_credito(
    firm_ids, firm_liquidez, firm_deuda, factura_esperada_salarial, bancos_ids
)

# 2.2 Bancos se financian (Mercado Interbancario + Impuestos)
nuevos_prestamos_ib, contratos_finales_empresas, bancos_liquidez_final = (
    paso2_interbancario(
        bancos_ids,
        bancos_liquidez,
        bancos_patrimonio,
        bancos_depositos,
        bancos_deuda_acumulada,
        contratos_potenciales,
        matriz_interbancaria_anterior,
        tax_mode=p.TAX_MODE,
    )
)

# --- Actualización de Liquidez Empresarial (Post-Crédito) ---
# Sumamos el dinero prestado a la caja de las empresas ANTES de pagar nóminas
if len(contratos_finales_empresas) > 0:
    indices = contratos_finales_empresas[:, 0].astype(int)  # IDs Empresas
    montos = contratos_finales_empresas[:, 2]  # Cantidad prestada

    # Actualizamos Liquidez y Deuda
    np.add.at(firm_liquidez, indices, montos)
    np.add.at(firm_deuda, indices, montos)

# --- Verificación Paso 2 ---
solicitudes_totales = len(contratos_potenciales)
prestamos_reales = len(contratos_finales_empresas)
print(f"   [Check] Solicitudes de Crédito: {solicitudes_totales}")
print(f"   [Check] Préstamos Otorgados: {prestamos_reales}")
if prestamos_reales < solicitudes_totales:
    print(
        f"   [Atención] Hubo 'Credit Crunch': {solicitudes_totales - prestamos_reales} préstamos denegados por iliquidez bancaria."
    )

print(f"   [Check] Transacciones Interbancarias: {len(nuevos_prestamos_ib)}")


# ==========================================
# PASO 3: PRODUCCIÓN Y MERCADO LABORAL
# ==========================================
print("\n>>> EJECUTANDO PASO 3: Producción y Nóminas (Hire/Fire)...")

# Llamada a la lógica estricta (Paso 3 corregido)
(
    firm_produccion_real,
    firm_trabajadores_reales,
    firm_coste_salarial,
    hogares_ingresos_nomina,
    firm_liquidez,  # Liquidez actualizada (menos salarios pagados)
    hogares_empleo_estado,  # Estado de empleo actualizado (t)
) = paso3_produccion_y_mercado_laboral(
    demanda_trabajo,  # Objetivo (target)
    firm_liquidez,  # Capacidad financiera (con préstamos incluidos)
    hogares_empleo_estado,  # Estado t-1 (Persistencia)
    hogares_es_trabajador,  # Máscara (Solo contratamos obreros)
)

# Actualización Liquidez Hogares (Cobran nómina)
# Sumamos el ingreso salarial a sus ahorros previos
hogares_liquidez += hogares_ingresos_nomina

# --- Verificación Paso 3 ---
total_prod = np.sum(firm_produccion_real)
total_empleo = np.sum(firm_trabajadores_reales)
posibles_trabajadores = np.sum(hogares_es_trabajador)
tasa_paro = 1.0 - (total_empleo / posibles_trabajadores)

print(f"   [Result] Producción Total del Sistema: {total_prod:.2f}")
print(
    f"   [Result] Empleo Total: {total_empleo} / {posibles_trabajadores} trabajadores"
)
print(f"   [Result] Tasa de Desempleo: {tasa_paro * 100:.2f}%")
print(f"   [Result] Masa Salarial Total Pagada: {np.sum(firm_coste_salarial):.2f}")

# Check Consistencia Contable (El dinero que sale = dinero que entra)
diff_nomina = abs(np.sum(firm_coste_salarial) - np.sum(hogares_ingresos_nomina))
if diff_nomina > 1e-5:
    print(f"   !!! ERROR CRÍTICO: Discrepancia en nóminas ({diff_nomina})")
else:
    print(
        "   [Check Contable] OK: Salarios pagados por empresas = Recibidos por hogares."
    )

# Check Racionamiento de Producción
# Comparamos lo que querían producir (Paso 1) vs lo que pudieron producir (Paso 3)
deficit_prod = demanda_esperada - firm_produccion_real
empresas_racionadas = np.sum(deficit_prod > 0.1)
if empresas_racionadas > 0:
    print(
        f"   [Info] {empresas_racionadas} empresas produjeron menos de lo planeado (Falta de liquidez o de mano de obra)."
    )
else:
    print("   [Info] Todas las empresas cumplieron sus objetivos de producción.")

# ==========================================
# FIN POR AHORA (PASO 4 EN ESPERA)
# ==========================================
print("\n--- Simulación detenida antes del Paso 4 ---")
