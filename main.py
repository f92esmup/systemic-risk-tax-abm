# Script principal. Orquestador de la simulación.

from parametros import Param as p
import numpy as np
from logica.paso1 import paso1

np.random.seed(42)

# Inicializamos las variables del paso t-1. Ya que el paso t depende de t-1.
# Vamos a usar valores aleatorios.

# Matrices de empresas
# NOTE: Podría eliminar los parámetros hardcodeados.
firm_prices = np.random.uniform(0.9, 1.1, p.F)  # P_i(t -1)
firm_produccion = np.random.uniform(8, 12, p.F)  # Y_i(t -1).
firm_ventas = firm_produccion * np.random.uniform(0.8, 1.0, p.F)  # S_i(t -1).
firm_liquidez = np.random.uniform(100, 200, p.F)  # L_i(t -1). Liquidez inicial.

# Ahora procdemos con el paso 1.
nuevos_precios, demanda_trabajo, demanda_esperada, factura_esperada_salarial = paso1(
    firm_prices, firm_produccion, firm_ventas
)
print(" --- Resultados Paso 1 --- ")
# Verificación de lógica:
# Tomemos una empresa que tenía inventario y era cara (debería bajar precio)
avg_p_preview = np.mean(firm_prices)
mask_down = ((firm_produccion - firm_ventas) > 1e-5) & (firm_prices > avg_p_preview)
idx_example = np.where(mask_down)[0]

if len(idx_example) > 0:
    idx = idx_example[0]
    print(f"Empresa {idx} tenía inventario y era cara.")
    print(
        f"Precio anterior: {firm_prices[idx]:.4f}, Nuevo precio: {nuevos_precios[idx]:.4f}"
    )
    print(
        f"Demanda esperada: {demanda_esperada[idx]:.4f}, Producción anterior: {firm_produccion[idx]:.4f}"
    )
