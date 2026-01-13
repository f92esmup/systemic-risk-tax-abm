
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

from logger import SimulationLogger

# Eliminar antigua función guardar_datos_parquet ya que usamos SimulationLogger


def ejecutar_simulacion(modo_impuesto="NINGUNO", semilla=None, run_id="test"):
    """
    Ejecuta una simulación completa del modelo CRISIS con lógica matricial.
    """
    if semilla is not None:
        np.random.seed(semilla)
        
    # Limpiar/Crear directorio run si es t=0 (hecho por caller o aqui)
    # create_run_dir(run_id) 
        
    start_time = time.time()
    logger = SimulationLogger()


    # 1. INICIALIZACIÓN
    F, B, H = p.F, p.B, p.H
    
    state = {
        # Empresas (Firms)
        'firms_prices': np.full(F, p.PRECIO_INICIAL),
        'firms_production': np.full(F, p.PRODUCCION_INICIAL),
        'firms_demand': np.full(F, p.PRODUCCION_INICIAL),
        'firms_inventory': np.zeros(F),
        'firms_equity': np.full(F, p.EQUITY_INICIAL_FIRMAS),
        'firms_liquidity': np.full(F, p.LIQUIDEZ_INICIAL_FIRMAS),
        'firms_wage': np.full(F, p.W_BASE), # [NEW] Salarios
        'firms_target_production': np.full(F, p.PRODUCCION_INICIAL),
        'firms_labor_demand': np.zeros(F, dtype=int),
        'mask_renacidas': np.zeros(F, dtype=bool),
        
        # Bancos (Banks)
        'banks_equity': np.full(B, p.EQUITY_INICIAL_BANCOS),
        'banks_liquidity': np.full(B, p.LIQUIDEZ_INICIAL_BANCOS),
        
        # Hogares (Households)
        'households_deposits': np.full(H, p.DEPOSITOS_INICIALES_HOGARES),
        'households_bank': np.random.randint(0, B, size=H),
        'households_dividends': np.zeros(H),
        
        # Redes (Networks)
        'net_FB': np.zeros((F, B)),
        'net_BB': np.zeros((B, B)),
        'rates_FB': np.zeros((F, B)),
        'rates_BB': np.zeros((B, B)),
        
        # Matrices de Flujo (para logger/stats)
        'labor_matrix': np.zeros((F, H))
    }

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
        # Paso 1: Planificación (Adaptativo - Mark I)
        update_p1 = paso1(state, p)
        state['firms_target_production'] = update_p1['firms_target_production']
        state['firms_prices'] = update_p1['firms_prices']
        state['firms_labor_demand'] = update_p1['firms_labor_demand']

        # Paso 2: Crédito y Salarios (Matrix FB & BB)
        # Inyectar MODO_IMPUESTO en params dinámicamente
        p.MODO_IMPUESTO = modo_impuesto
        
        # Llamada vectorizada
        res_p2 = paso2(state, p)
        
        # Actualización de Estado (Post-Paso 2)
        state['net_FB'] = res_p2['net_FB']
        state['net_BB'] = res_p2['net_BB']
        state['firms_liquidity'] = res_p2['firms_liquidity']
        state['banks_liquidity'] = res_p2['banks_liquidity']
        state['firms_labor_demand'] = res_p2['firms_labor_demand'] # Real hired
        
        # Actualizar tasas FB donde hubo préstamos nuevos
        # res_p2['new_rates_FB'] vector (F,)
        # res_p2['bank_indices'] vector (F,)
        # Asignar la tasa nueva a la celda correspondiente
        state['rates_FB'][np.arange(F), res_p2['bank_indices']] = res_p2['new_rates_FB']

        # Guardar datos scatter SRT
        if 'delta_el' in res_p2 and t % 50 == 0:
            historia["SRT_Scatter"][f"t_{t}"] = {
                "Delta_EL": res_p2['delta_el'].copy(),
                "Pasivos_IB": state['net_BB'].copy()
            }

        # Calcular DebtRank (Reporting)
        total_lending = np.sum(state['net_BB'], axis=0)
        V_total = np.sum(total_lending)
        if V_total > 1e-6:
            v_sys = total_lending / V_total
            dr_vector = calcular_debtrank_vector(state['net_BB'], state['banks_equity'], v_sys)
            avg_dr = np.mean(dr_vector)
        else:
            dr_vector = np.zeros(B)
            avg_dr = 0.0

        # Paso 3: Producción Física (Ya se pagaron salarios en P2)
        (produccion_real, oferta_bienes, wages_matrix_FH) = paso3(
            state['firms_labor_demand'], # Hired
            res_p2['wages_paid_vector'], # Wage Bill
            state['firms_inventory']
        )
        state['firms_production'] = produccion_real
        state['labor_matrix'] = wages_matrix_FH
        
        # --- CORRECCIÓN STOCK-FLOW: PAGO DE SALARIOS A HOGARES ---
        # Sumar los salarios recibidos por cada hogar (sumando sobre empresas)
        total_wages_H = np.sum(wages_matrix_FH, axis=0) # (H,)
        state['households_deposits'] += total_wages_H
        # ---------------------------------------------------------
        
        # Paso 4: Consumo (Matrix HF)
        # Factura pagada es wages_paid_vector
        res_p4 = paso4(state, p)
        
        # IMPORTANTE: Guardamos la demanda para el próximo paso 1
        state['firms_demand'] = res_p4['firms_demand_received']
        state['firms_inventory'] = res_p4['firms_inventory']
        state['households_deposits'] = res_p4['households_deposits']
        state['firms_revenue'] = res_p4['firms_revenue'] # Necesario para paso5
        
        # Paso 5: Contabilidad (Matrix FB)
        res_p5 = paso5(state, p)
        
        # Sincronización final del estado
        state['firms_liquidity'] = res_p5['firms_liquidity']
        state['firms_equity'] = res_p5['firms_equity']
        state['net_FB'] = res_p5['net_FB']
        
        state['banks_liquidity'] = res_p5['banks_liquidity']
        state['banks_equity'] = res_p5['banks_equity']
        state['net_BB'] = res_p5['net_BB']
        
        dividendos_total = res_p5['dividends_total']
        dividendos_pc = dividendos_total / H
        state['households_dividends'] = np.full(H, dividendos_pc)
        state['households_deposits'] += state['households_dividends'] # Pagar dividendos a hogares (Income)
        
        state['mask_renacidas'] = res_p5['mask_bankrupt_F']
        
        # Reset de Tasas FB para empresas que murieron (y renacieron)
        state['rates_FB'][state['mask_renacidas'], :] = 0.0

        # Registro Aggregado
        total_quiebras_B = res_p5['bankruptcies_B']
        total_losses_contagion = res_p5['contagion_loss']

        historia["t"].append(t)
        historia["DebtRank_Promedio"].append(avg_dr)
        historia["Total_Deuda_Interbancaria"].append(V_total)
        historia["Total_Equity_Bancos"].append(np.sum(state['banks_equity']))
        
        if total_quiebras_B > 0:
            historia["Eventos_Cascada_Size"].append(int(total_quiebras_B))
            historia["Eventos_Perdida_Total"].append(total_losses_contagion)
            
        if t == 100 or t == (p.T - 1):
            historia["Snapshots"][f"t_{t}"] = {
                "DebtRank": dr_vector.copy(),
                "Equity_Bancos": state['banks_equity'].copy()
            }
            
        # --- PARQUET LOGGING ---
        # Data Preparation
        agents = {
            "firms": pd.DataFrame({
                "id": range(F), 
                "liq": state['firms_liquidity'], 
                "eq": state['firms_equity'], 
                "prod": state['firms_production']
            }),
            "banks": pd.DataFrame({
                "id": range(B), 
                "liq": state['banks_liquidity'], 
                "eq": state['banks_equity'],
                "dr": dr_vector # [NEW] Saved specifically for plotting Fig 3b
            }),
            "households": pd.DataFrame({
                "id": range(H), 
                "dep": state['households_deposits'], 
                "bank_id": state['households_bank']
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
            "total_eq_banks": np.sum(state['banks_equity'])
        }])
        
        # Add metrics to agents dict for saving (hacky but keeps signature clean)
        # or separate category. Let's iterate `agents` as "Tabular Data"
        agents["globals"] = metrics_df

        # SRT Scatter Data (Delta EL)
        # Only relevant for SRT mode step-analysis, but if we want to reproduce Fig 3d:
        # We need the Delta_EL matrix.
        if "delta_el" in res_p2:
            # Flatten or save matrix? Matrix is better.
            # We add it to networks/matrices list.
            # debug_data["delta_el"] is (B,B)
            # Pass it as a network with a special name
            pass 
            
        # Construir matrices "completas" para graph
        # Deposits HB Matrix: Sparse
        deposits_hb_matrix = np.zeros((H, B))
        deposits_hb_matrix[np.arange(H), state['households_bank']] = state['households_deposits']
        
        # Consumption Matrix HF reconstruction
        # res_p4['consumption_flows'] is tuple (chosen_firms, actual_spending_H)
        (chosen_firms, actual_spending_H) = res_p4['consumption_flows']
        consumption_matrix_HF = np.zeros((H, F)) # Warning: Dense matrix might be heavy? H=1300 F=100 -> 130k elements. OK.
        # Vectorized fill
        # consumption_matrix_HF[h, chosen_firm[h]] = spending[h]
        consumption_matrix_HF[np.arange(H), chosen_firms] = actual_spending_H
        
        networks = {
            "net_FB": state['net_FB'],
            "net_BB": state['net_BB'],
            "net_FH": state['labor_matrix'],
            "net_HF": consumption_matrix_HF,
            "net_HB": deposits_hb_matrix
        }

        # [FIX] Record data in logger
        logger.log_step(t, agents, networks)


    logger.flush(run_id)
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
