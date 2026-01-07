# Script principal. Orquestador de la simulación del Modelo CRISIS.
# Cubre inicialización y Pasos 1, 2 y 3.

import numpy as np
from parametros import Param as p

# Importación de Módulos de Lógica
from logica.paso1 import paso1
from logica.paso2 import paso2_mercado_credito, paso2_interbancario
from logica.paso3 import paso3_produccion_y_mercado_laboral
from logica.paso4 import paso4_consumo
from logica.paso5 import paso5_resultados_y_quiebras

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
firm_inventario = np.maximum(firm_produccion - firm_ventas, 0)
firm_liquidez = np.random.uniform(100, 200, p.F)  # L_i(t-1)
firm_deuda = np.random.uniform(0, 50, p.F)  # Deuda externa inicial

# --- 0.2 Variables de Bancos (Banks) ---
bancos_ids = np.arange(p.B)
bancos_liquidez = np.random.uniform(200, 1000, p.B)
bancos_patrimonio = np.random.uniform(50, 100, p.B)  # Equity (C_j)
bancos_depositos = np.random.uniform(500, 2000, p.B)  # Depósitos de clientes

# Deuda interbancaria inicial (Matriz L)
matriz_interbancaria_anterior = np.zeros((p.B, p.B))
# Acumulador del Fondo de Rescate (SRT recaudado)
fondo_rescate_acumulado = 0.0
# Variables de Estado para Deuda Detallada
# Matriz Préstamos Total (Principal + Interés)
matriz_prestamos_firmas = np.zeros((p.F, p.B))
# Matriz SOLO Intereses (Para cálculo de beneficios Paso 5)
matriz_intereses_firmas = np.zeros((p.F, p.B))
# Matriz Intereses Interbancarios
matriz_intereses_ib = np.zeros((p.B, p.B))
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
firm_costo_financiero_iteracion = np.zeros(p.F)  # Reset para este paso

if len(contratos_finales_empresas) > 0:
    indices_f = contratos_finales_empresas[:, 0].astype(int)  # IDs Empresas
    indices_b = contratos_finales_empresas[:, 1].astype(int)
    montos = contratos_finales_empresas[:, 2]  # Cantidad prestada
    tasas = contratos_finales_empresas[:, 3]

    # Intereses de estos contratos
    intereses_nuevos = montos * tasas
    deuda_total = montos + intereses_nuevos

    # Actualizamos Liquidez y Deuda
    np.add.at(firm_liquidez, indices_f, montos)
    np.add.at(firm_deuda, indices_f, deuda_total)
    np.add.at(firm_costo_financiero_iteracion, indices_f, intereses_nuevos)

    # Actualizar MATRICES (Bucle rápido)
    for f, b, d_tot, int_only in zip(
        indices_f, indices_b, deuda_total, intereses_nuevos
    ):
        matriz_prestamos_firmas[f, b] += d_tot
        matriz_intereses_firmas[f, b] += int_only

# --- Verificación Paso 2 ---
solicitudes_totales = len(contratos_potenciales)
prestamos_reales = len(contratos_finales_empresas)
print(f"   [Check] Solicitudes de Crédito: {solicitudes_totales}")
print(f"   [Check] Préstamos Otorgados: {prestamos_reales}")
if prestamos_reales < solicitudes_totales:
    print(
        f"   [Atención] Hubo 'Credit Crunch': {solicitudes_totales - prestamos_reales} préstamos denegados por iliquidez bancaria."
    )

# Actualizar Matriz Interbancaria con Intereses y FONDO DE RESCATE
if len(nuevos_prestamos_ib) > 0:
    # Ahora la matriz tiene 5 columnas: [Lender, Borrower, Amount, TotalRate, TaxRate]
    lenders = nuevos_prestamos_ib[:, 0].astype(int)
    borrowers = nuevos_prestamos_ib[:, 1].astype(int)
    amounts = nuevos_prestamos_ib[:, 2]
    total_rates = nuevos_prestamos_ib[:, 3]
    tax_rates = nuevos_prestamos_ib[:, 4]  # Nueva Columna

    # Cálculo exacto de intereses y deuda
    interest_ib = amounts * total_rates
    total_ib = amounts + interest_ib

    # Cálculo exacto del impuesto recaudado
    tax_collected = np.sum(amounts * tax_rates)
    fondo_rescate_acumulado += tax_collected  # Acumulamos en la variable global

    # Sumar a las matrices globales
    for l, b, tot, intr in zip(lenders, borrowers, total_ib, interest_ib):
        matriz_interbancaria_anterior[b, l] += tot
        matriz_intereses_ib[b, l] += intr

print(f"   [Check] Transacciones Interbancarias: {len(nuevos_prestamos_ib)}")
print(f"   [Check] Fondo de Rescate Acumulado: {fondo_rescate_acumulado:.4f}")


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
# PASO 4: CONSUMO (HOGARES COMPRAN BIENES)
# ==========================================
print("\n>>> EJECUTANDO PASO 4: Consumo (Mercado de Bienes)...")

(
    firm_ventas_reales,  # S_i(t)
    firm_inventario_final,  # Stock para t+1
    firm_ingresos,  # Flujo de caja
    hogares_liquidez,  # Ahorro post-consumo
    hogares_gasto_total,  # Check
) = paso4_consumo(
    hogares_liquidez,
    nuevos_precios,  # P_i(t) fijados en Paso 1
    firm_produccion_real,  # Y_i(t) producidos en Paso 3
    firm_inventario,  # Inventario que venía de t-1
)

# --- Actualización de Estado (Firms) ---
firm_liquidez += firm_ingresos  # Entra dinero a la caja

# Preparamos variables para la siguiente iteración (t+1)
# En t+1, el Paso 1 usará estas ventas y este inventario para decidir precios.
firm_ventas = firm_ventas_reales
firm_inventario = firm_inventario_final

# --- Verificación Paso 4 ---
pib_gasto = np.sum(firm_ingresos)
stock_sobrante = np.sum(firm_inventario_final)
print(
    f"   [Result] Ventas Totales (PIB): {pib_gasto:.2f} (Stock sobrante: {stock_sobrante:.2f})"
)

# Check de consistencia: Racionamiento
demanda_potencial = np.sum(hogares_gasto_total)

if stock_sobrante < 1e-5 and pib_gasto > 0:
    print("   [Info] Mercado 'vaciado' (Todo el stock vendido).")
elif stock_sobrante > 0:
    print("   [Info] Exceso de Oferta: Quedó inventario sin vender.")

# ==========================================
# PASO 5: RESULTADOS, DIVIDENDOS Y QUIEBRAS
# ==========================================
print("\n>>> EJECUTANDO PASO 5: Resultados Financieros y Quiebras...")

(
    firm_liquidez,
    bancos_patrimonio,
    bancos_liquidez,
    bancos_activos,
    matriz_prestamos_firmas,
    matriz_interbancaria_anterior,
    hogares_liquidez,
    num_quiebras_firmas,
    num_quiebras_bancos,
) = paso5_resultados_y_quiebras(
    firm_liquidez,
    firm_ingresos,
    firm_coste_salarial,
    firm_costo_financiero_iteracion,
    matriz_prestamos_firmas,
    matriz_intereses_firmas,  # NUEVO
    bancos_liquidez,
    bancos_patrimonio,
    matriz_interbancaria_anterior,
    matriz_intereses_ib,  # NUEVO
    hogares_liquidez,
    np.arange(p.F),  # Dueños firmas
    np.arange(p.F, p.F + p.B),  # Dueños bancos
    fondo_rescate_acumulado,  # NUEVO: Fondo de Rescate
)

print(f"   [Result] Quiebras de Empresas: {num_quiebras_firmas}")
print(f"   [Result] Quiebras de Bancos: {num_quiebras_bancos}")
