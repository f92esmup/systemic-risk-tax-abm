
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from parametros import Param as p

# Importación de Módulos Lógicos
from logica.paso1 import paso1
from logica.paso2 import paso2, calcular_debtrank_vector
from logica.paso3 import paso3
from logica.paso4 import paso4
from logica.paso5_6_7 import paso5

# =============================================================================
# CONFIGURACIÓN VISUAL Y SALIDA
# =============================================================================
plt.style.use('ggplot')
PARAMS = {
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': (10, 6)
}
plt.rcParams.update(PARAMS)

OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIMULATIONS_PER_MODE = 5  # Cantidad de simulaciones para suavizado estadístico

# =============================================================================
# MOTOR DE SIMULACIÓN
# =============================================================================
# =============================================================================
# MOTOR DE SIMULACIÓN Y LOGGING
# =============================================================================
import pandas as pd
import shutil

def guardar_datos_parquet(run_id, t, agents_data, networks_data, output_dir="outputdata"):
    """
    Guarda el estado del sistema en Parquet.
    Estructura: outputdata/run_{id}/step_{t}/...
    """
    step_dir = os.path.join(output_dir, f"run_{run_id}", f"step_{t}")
    os.makedirs(step_dir, exist_ok=True)
    
    # 1. Agentes
    for name, df in agents_data.items():
        df.to_parquet(os.path.join(step_dir, f"{name}.parquet"))
        
    # 2. Redes (Matrices) - Guardar como Edgelist para eficiencia
    for name, matrix in networks_data.items():
        # Convertir a Sparse COnvertiblo o Edgelist
        # Si es densa pequeña (20x20) guardar directa, si es grande (100x1300) edgelist
        if matrix.size < 10000:
             # Guardar como CSV/Parquet matriz completa
             pd.DataFrame(matrix).to_parquet(os.path.join(step_dir, f"{name}_matrix.parquet"))
        else:
             # Edgelist: row, col, val
             rows, cols = np.nonzero(matrix)
             vals = matrix[rows, cols]
             if len(vals) > 0:
                 df_edge = pd.DataFrame({'source': rows, 'target': cols, 'weight': vals})
                 df_edge.to_parquet(os.path.join(step_dir, f"{name}_edges.parquet"))

def ejecutar_simulacion(modo_impuesto="NINGUNO", semilla=None, run_id="test"):
    """
    Ejecuta una simulación completa del modelo CRISIS con lógica matricial.
    """
    if semilla is not None:
        np.random.seed(semilla)
        
    # Limpiar/Crear directorio run si es t=0 (hecho por caller o aqui)
    # create_run_dir(run_id) 
        
    start_time = time.time()

    # 1. INICIALIZACIÓN
    F, B, H = p.F, p.B, p.H
    
    # AGENTS STATE
    precios = np.full(F, p.PRECIO_INICIAL)
    produccion = np.full(F, p.PRODUCCION_INICIAL)
    ventas = produccion.copy()
    inventario = np.zeros(F)
    
    equity_empresas = np.full(F, p.EQUITY_INICIAL_FIRMAS)
    liquidez_empresas = np.full(F, p.LIQUIDEZ_INICIAL_FIRMAS)
    
    # [Refactor] Matrix State
    # Pasivos FB: Deuda (F, B). Inicialmente 0.
    pasivos_fb = np.zeros((F, B))
    tasas_fb = np.zeros((F, B)) # Tasas de esos contratos
    
    # Matriz Laboral (F, H): Inicialmente asignada aleatoria o vacia?
    # Para cumplir "full employment" inicial:
    # Asignamos H a F aleatoriamente
    labor_matrix = np.zeros((F, H)) 
    # (Se poblará/usará en paso 3 dinámicamente si es spot, o static)
    
    equity_bancos = np.full(B, p.EQUITY_INICIAL_BANCOS)
    liquidez_bancos = np.full(B, p.LIQUIDEZ_INICIAL_BANCOS)
    
    pasivos_interbancarios = np.zeros((B, B))
    tasas_interbancarias = np.zeros((B, B)) 

    depositos_hogares = np.full(H, p.DEPOSITOS_INICIALES_HOGARES)
    # [Refactor] Households-Banks Relationship
    # Asignamos cada hogar a un banco principal aleatorio
    banco_principal_hogar = np.random.randint(0, B, size=H)
    
    dividendos_previos = np.zeros(H)
    mask_renacidas = np.zeros(F, dtype=bool)

    # Historia (Aggregates for Plots)
    historia = {
        "t": [],
        "DebtRank_Promedio": [],
        "Total_Equity_Bancos": [],
        "Total_Deuda_Interbancaria": [],
        "Eventos_Cascada_Size": [],
        "Eventos_Perdida_Total": [],
        "Snapshots": {},
        "SRT_Scatter": {}
    }

    # 2. BUCLE
    for t in range(p.T):
        # Paso 1: Planificación
        (nuevos_precios, demanda_laboral, produccion_necesaria, 
         demanda_credito, factura_salarial, demanda_obj) = paso1(
            precios, produccion, ventas, liquidez_empresas, 
            inventario, mask_renacidas
        )
        precios = nuevos_precios

        # Paso 2: Crédito (Matrix FB)
        # Necesitamos pasar deuda total actual para leverage calc
        deuda_total_empresas = np.sum(pasivos_fb, axis=1)
        
        (nuevos_prestamos_matrix, tasas_finales, nueva_matriz_ib, 
         liquidez_bancos_post, bancos_elegidos, matriz_impuestos, debug_data) = paso2(
            demanda_credito, liquidez_bancos, equity_bancos,
            pasivos_interbancarios, equity_empresas, deuda_total_empresas,
            modo_impuesto=modo_impuesto
        )
        
        # Debemos actualizar Pasivos FB y Tasas FB con los nuevos prestamos
        # Nuevos prestamos (F,B). Sumamos al stock.
        # Tasas: Si hay nuevo prestamo, actualizamos la tasa (simplificación: average rate or marginal?)
        # Asumimos marginal rate replaces old rate for simplicity or weighted?
        # Modelo simple: New loan updates rate.
        mask_new_loans = nuevos_prestamos_matrix > 0
        pasivos_fb += nuevos_prestamos_matrix
        
        # Actualizamos tasa donde hubo nuevo prestamo.
        # tasas_finales es vector (F). Explotamos a (F,B) usando bancos_elegidos?
        # paso2 returns tasas_finales vector (best offer).
        # We need to map this to the specific bank column.
        # bancos_elegidos (F) indices.
        # Logic: tasas_fb[f, bancos_elegidos[f]] = tasas_finales[f] IF loan > 0
        for f in range(F):
            if mask_new_loans[f, bancos_elegidos[f]]:
                tasas_fb[f, bancos_elegidos[f]] = tasas_finales[f]

        # Guardar datos scatter SRT
        if "delta_el" in debug_data and t % 50 == 0:
            historia["SRT_Scatter"][f"t_{t}"] = {
                "Delta_EL": debug_data["delta_el"].copy(),
                "Pasivos_IB": pasivos_interbancarios.copy()
            }

        pasivos_interbancarios = nueva_matriz_ib
        liquidez_bancos = liquidez_bancos_post
        
        # Calcular DebtRank (Riesgo Sistémico)
        total_lending = np.sum(pasivos_interbancarios, axis=0)
        V_total = np.sum(total_lending)
        if V_total > 1e-6:
            v_sys = total_lending / V_total
            dr_vector = calcular_debtrank_vector(pasivos_interbancarios, equity_bancos, v_sys)
            avg_dr = np.mean(dr_vector)
        else:
            dr_vector = np.zeros(B)
            avg_dr = 0.0

        # Paso 3: Producción (Matrix FH implied)
        # Sumamos nuevos prestamos por empresa para caja
        nuevos_prestamos_total = np.sum(nuevos_prestamos_matrix, axis=1)
        
        (produccion_real, oferta_bienes, empleo_real, 
         factura_pagada, liquidez_empresas_post, wages_matrix_FH) = paso3(
            demanda_laboral, liquidez_empresas, nuevos_prestamos_total, inventario
        )
        produccion = produccion_real
        liquidez_empresas = liquidez_empresas_post
        
        # Paso 4: Consumo (Matrix HF)
        (ventas_real, ingresos_ventas, inventario_final, 
         depositos_post, demanda_teorica, _, consumption_matrix_HF) = paso4(
            precios, oferta_bienes, factura_pagada, depositos_hogares,
            dividendos_previos
        )
        ventas = ventas_real
        inventario = inventario_final
        depositos_hogares = depositos_post
        
        # Actualizar H-B Deposits Matrix (Virtual)
        # deposits_hb[h, banco_principal[h]] = depositos_hogares[h]
        
        # Paso 5: Contabilidad (Matrix FB)
        (liquidez_empresas_end, equity_empresas_end, deuda_remanente_fb, 
         mask_quiebra_F, liquidez_bancos_end, equity_bancos_end, 
         mask_quiebra_B, pasivos_ib_end, dividendos_pc, 
         total_quiebras_B, total_losses_contagion) = paso5(
            liquidez_empresas, ingresos_ventas, equity_empresas,
            pasivos_fb, tasas_fb, # Matrix inputs
            liquidez_bancos, equity_bancos, pasivos_interbancarios, tasas_interbancarias,
            depositos_hogares,
            tax_matrix_ib=matriz_impuestos
        )
        
        # Actualización final
        liquidez_empresas = liquidez_empresas_end
        equity_empresas = equity_empresas_end
        pasivos_fb = deuda_remanente_fb # Matrix update
        
        liquidez_bancos = liquidez_bancos_end
        equity_bancos = equity_bancos_end
        pasivos_interbancarios = pasivos_ib_end
        dividendos_previos = np.full(H, dividendos_pc)
        mask_renacidas = mask_quiebra_F
        
        # Reset de Tasas/Relaciones FB si quiebra
        # paso5 already handles write-off in deuda_remanente_fb (sets row to 0)
        # Pero tasas? Deberíamos resetear tasas a 0 para muertos.
        tasas_fb[mask_quiebra_F, :] = 0.0

        # Registro Aggregado
        historia["t"].append(t)
        historia["DebtRank_Promedio"].append(avg_dr)
        historia["Total_Deuda_Interbancaria"].append(V_total)
        historia["Total_Equity_Bancos"].append(np.sum(equity_bancos))
        
        if total_quiebras_B > 0:
            historia["Eventos_Cascada_Size"].append(int(total_quiebras_B))
            historia["Eventos_Perdida_Total"].append(total_losses_contagion)
            
        if t == 100 or t == (p.T - 1):
            historia["Snapshots"][f"t_{t}"] = {
                "DebtRank": dr_vector.copy(),
                "Equity_Bancos": equity_bancos.copy()
            }
            
        # --- PARQUET LOGGING ---
        # Data Preparation
        agents = {
            "firms": pd.DataFrame({
                "id": range(F), 
                "liq": liquidez_empresas, 
                "eq": equity_empresas, 
                "prod": produccion
            }),
            "banks": pd.DataFrame({
                "id": range(B), 
                "liq": liquidez_bancos, 
                "eq": equity_bancos,
                "dr": dr_vector # [NEW] Saved specifically for plotting Fig 3b
            }),
            "households": pd.DataFrame({
                "id": range(H), 
                "dep": depositos_hogares, 
                "bank_id": banco_principal_hogar
            })
        }
        
        # Global Metrics (Scalars)
        # Saved as a single-row DataFrame for consistency
        metrics_df = pd.DataFrame([{
            "t": t,
            "volume_ib": V_total,
            "avg_dr": avg_dr,
            "cascade_size": total_quiebras_B,
            "contagion_loss": total_losses_contagion,
            "total_eq_banks": np.sum(equity_bancos)
        }])
        
        # Add metrics to agents dict for saving (hacky but keeps signature clean)
        # or separate category. Let's iterate `agents` as "Tabular Data"
        agents["globals"] = metrics_df

        # SRT Scatter Data (Delta EL)
        # Only relevant for SRT mode step-analysis, but if we want to reproduce Fig 3d:
        # We need the Delta_EL matrix.
        if "delta_el" in debug_data:
            # Flatten or save matrix? Matrix is better.
            # We add it to networks/matrices list.
            # debug_data["delta_el"] is (B,B)
            # Pass it as a network with a special name
            pass 
            
        # Construir matrices "completas" para graph
        # Deposits HB Matrix: Sparse
        deposits_hb_matrix = np.zeros((H, B))
        deposits_hb_matrix[np.arange(H), banco_principal_hogar] = depositos_hogares
        
        networks = {
            "net_FB": pasivos_fb, # Debt
            "net_BB": pasivos_interbancarios, # Interbank
            "net_FH": wages_matrix_FH, # Labor Flows
            "net_HF": consumption_matrix_HF, # Consumption Flows
            "net_HB": deposits_hb_matrix # Deposits Stocks
        }
        
        if "delta_el" in debug_data:
            networks["matrix_delta_el"] = debug_data["delta_el"]
        
        guardar_datos_parquet(run_id, t, agents, networks)

    return historia

# =============================================================================
# EXPERIMENTO
# =============================================================================
def run_experiment():
    modes = ["NINGUNO", "TOBIN", "SRT"]
    
    print(f"--- Iniciando Experimento Comparativo ({SIMULATIONS_PER_MODE} runs/modo) ---")
    
    for mode in modes:
        print(f"Modo: {mode} ", end="")
        for i in range(SIMULATIONS_PER_MODE):
            run_id = f"{mode}_sim_{i}"
            # Ejecutar y guardar en disco
            ejecutar_simulacion(modo_impuesto=mode, semilla=42+i, run_id=run_id)
            
            if i % 5 == 0: print(".", end="", flush=True)
        print(" OK")

if __name__ == "__main__":
    print("=== Systemic Risk Tax ABM: Orchestrator ===")
    
    # 1. Ejecutar Simulaciones
    run_experiment()
    
    print(f"Simulaciones completadas. Datos en ./{OUTPUT_DIR}")
    print("Para generar gráficas, ejecute: python figuras.py")
