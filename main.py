# Script principal. Orquestador de la simulación.

from parametros import Param as p
import numpy as np
from logica.paso1 import paso1

# Importamos las funciones del Paso 2
from logica.paso2 import paso2_mercado_credito, paso2_interbancario

np.random.seed(42)

# ==========================================
# INICIALIZACIÓN (t-1)
# ==========================================
print("--- Inicializando Simulación ---")

# 1. Variables de Empresas
firm_ids = np.arange(p.F)
firm_prices = np.random.uniform(0.9, 1.1, p.F)  # P_i(t -1)
firm_produccion = np.random.uniform(8, 12, p.F)  # Y_i(t -1)
firm_ventas = firm_produccion * np.random.uniform(0.8, 1.0, p.F)  # S_i(t -1)
firm_liquidez = np.random.uniform(100, 200, p.F)  # L_i(t -1)
firm_deuda = np.random.uniform(0, 50, p.F)  # Deuda externa inicial

# 2. Variables de Bancos
bancos_ids = np.arange(p.B)
# Liquidez: Algunos tendrán mucha, otros poca para forzar el interbancario
bancos_liquidez = np.random.uniform(200, 1000, p.B)
bancos_patrimonio = np.random.uniform(50, 100, p.B)  # Equity (C_j)
bancos_depositos = np.random.uniform(500, 2000, p.B)  # Depósitos de clientes
# Deuda interbancaria inicial (Matriz L)
matriz_interbancaria_anterior = np.zeros((p.B, p.B))
# Para probar DebtRank, creamos algunas deudas iniciales aleatorias
mask_deuda = np.random.rand(p.B, p.B) > 0.8
matriz_interbancaria_anterior[mask_deuda] = np.random.uniform(1, 10, np.sum(mask_deuda))
# Rellenamos diagonal con 0
np.fill_diagonal(matriz_interbancaria_anterior, 0)
# Vector de deuda acumulada por banco (suma de pasivos)
bancos_deuda_acumulada = np.sum(matriz_interbancaria_anterior, axis=1)


# ==========================================
# PASO 1: PLANIFICACIÓN DE EMPRESAS
# ==========================================
nuevos_precios, demanda_esperada, demanda_trabajo, factura_esperada_salarial = paso1(
    firm_prices, firm_produccion, firm_ventas
)

print("\n--- Resultados Paso 1 (Verificación) ---")
avg_p_prev = np.mean(firm_prices)
# Verificación: Empresa con inventario y cara -> Baja precio
mask_down = ((firm_produccion - firm_ventas) > 1e-5) & (firm_prices > avg_p_prev)
idx_example = np.where(mask_down)[0]

if len(idx_example) > 0:
    idx = idx_example[0]
    print(f"Empresa {idx} (Caso A): Tenía inventario y era cara.")
    print(f"  > Precio t-1: {firm_prices[idx]:.4f} -> Nuevo: {nuevos_precios[idx]:.4f}")
    print(
        f"  > Factura Salarial Necesaria: {factura_esperada_salarial[idx]:.2f} (Liquidez: {firm_liquidez[idx]:.2f})"
    )


# ==========================================
# PASO 2: MERCADO DE CRÉDITO Y BANCOS
# ==========================================

# 2.1 Empresas solicitan crédito a Bancos
contratos_potenciales, demanda_credito_empresas = paso2_mercado_credito(
    firm_ids, firm_liquidez, firm_deuda, factura_esperada_salarial, bancos_ids
)

# 2.2 Bancos gestionan liquidez (Mercado Interbancario + Impuestos)
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

print("\n--- Resultados Paso 2 (Verificación) ---")
num_solicitantes = np.sum(demanda_credito_empresas > 1e-5)
print(f"1. Demanda de Crédito:")
print(f"   - Empresas que necesitan dinero: {num_solicitantes}")
print(f"   - Total solicitado: {np.sum(demanda_credito_empresas):.2f}")
print(f"   - Contratos potenciales (Matching): {len(contratos_potenciales)}")

print(f"2. Mercado Interbancario ({p.TAX_MODE}):")
print(f"   - Transacciones entre bancos: {len(nuevos_prestamos_ib)}")

if len(nuevos_prestamos_ib) > 0:
    # Mostramos detalle de una transacción interbancaria
    lender = int(nuevos_prestamos_ib[0][0])
    borrower = int(nuevos_prestamos_ib[0][1])
    amount = nuevos_prestamos_ib[0][2]
    rate = nuevos_prestamos_ib[0][3]
    print(f"   - EJEMPLO: Banco {borrower} pidió {amount:.2f} a Banco {lender}")
    print(f"     > Tasa aplicada (con impuesto): {rate:.4f} (Base: {p.r_bar})")
else:
    print("   - No hubo actividad interbancaria (¿Sobra liquidez en el sistema?)")

print(f"3. Resultado Final:")
print(
    f"   - Préstamos a empresas otorgados: {len(contratos_finales_empresas)} de {len(contratos_potenciales)}"
)
if len(contratos_finales_empresas) < len(contratos_potenciales):
    print("   ! ATENCIÓN: Hubo Racionamiento de Crédito (Credit Crunch).")

# Guardamos estado para el siguiente paso
# (En un bucle real, esto actualizaría las variables state t -> t+1)
