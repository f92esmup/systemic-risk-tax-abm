
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

        # Cálculo de variables intermedias para crédito
        # Factura salarial teórica (N * w)
        factura_salarial_teorica = state['firms_labor_demand'] * p.W_BASE
        
        # Liquidez para crédito: Si renació t-1, tiene dotación inicial
        liq_para_credito = np.where(state['mask_renacidas'], p.LIQUIDEZ_INICIAL_FIRMAS, state['firms_liquidity'])
        demanda_credito = np.maximum(factura_salarial_teorica - liq_para_credito, 0.0)
        
        deuda_total_empresas = np.sum(state['net_FB'], axis=1)

        # Paso 2: Crédito (Matrix FB)
        (nuevos_prestamos_matrix, tasas_finales, nueva_matriz_ib, 
         liquidez_bancos_post, bancos_elegidos, matriz_impuestos, debug_data) = paso2(
            demanda_credito, state['banks_liquidity'], state['banks_equity'],
            state['net_BB'], state['firms_equity'], deuda_total_empresas,
            modo_impuesto=modo_impuesto
        )
        
        # Actualización de Deuda FB y Tasas
        state['net_FB'] += nuevos_prestamos_matrix
        # Si hubo préstamo, actualizamos la tasa pactada en esa relación
        for f in range(F):
            if np.sum(nuevos_prestamos_matrix[f]) > 0:
                state['rates_FB'][f, bancos_elegidos[f]] = tasas_finales[f]

        # Guardar datos scatter SRT
        if "delta_el" in debug_data and t % 50 == 0:
            historia["SRT_Scatter"][f"t_{t}"] = {
                "Delta_EL": debug_data["delta_el"].copy(),
                "Pasivos_IB": state['net_BB'].copy()
            }

        state['net_BB'] = nueva_matriz_ib
        state['banks_liquidity'] = liquidez_bancos_post
        
        # Calcular DebtRank (Riesgo Sistémico)
        total_lending = np.sum(state['net_BB'], axis=0)
        V_total = np.sum(total_lending)
        if V_total > 1e-6:
            v_sys = total_lending / V_total
            dr_vector = calcular_debtrank_vector(state['net_BB'], state['banks_equity'], v_sys)
            avg_dr = np.mean(dr_vector)
        else:
            dr_vector = np.zeros(B)
            avg_dr = 0.0

        # Paso 3: Producción (Matrix FH)
        nuevos_prestamos_total = np.sum(nuevos_prestamos_matrix, axis=1)
        
        (produccion_real, oferta_bienes, empleo_real, 
         factura_pagada, liquidez_empresas_post, wages_matrix_FH) = paso3(
            state['firms_labor_demand'], state['firms_liquidity'], nuevos_prestamos_total, state['firms_inventory']
        )
        state['firms_production'] = produccion_real
        state['firms_liquidity'] = liquidez_empresas_post
        state['labor_matrix'] = wages_matrix_FH
        
        # Paso 4: Consumo (Matrix HF)
        (ventas_real, ingresos_ventas, inventario_final, 
         depositos_post, demanda_teorica, _, consumption_matrix_HF) = paso4(
            state['firms_prices'], oferta_bienes, factura_pagada, state['households_deposits'],
            state['households_dividends']
        )
        # IMPORTANTE: Guardamos la demanda para el próximo paso 1
        state['firms_demand'] = demanda_teorica
        state['firms_inventory'] = inventario_final
        state['households_deposits'] = depositos_post
        
        # Paso 5: Contabilidad (Matrix FB)
        (liquidez_empresas_end, equity_empresas_end, deuda_remanente_fb, 
         mask_quiebra_F, liquidez_bancos_end, equity_bancos_end, 
         mask_quiebra_B, pasivos_ib_end, dividendos_pc, 
         total_quiebras_B, total_losses_contagion) = paso5(
            state['firms_liquidity'], ingresos_ventas, state['firms_equity'],
            state['net_FB'], state['rates_FB'],
            state['banks_liquidity'], state['banks_equity'], state['net_BB'], state['rates_BB'],
            state['households_deposits'],
            tax_matrix_ib=matriz_impuestos
        )
        
        # Sincronización final del estado
        state['firms_liquidity'] = liquidez_empresas_end
        state['firms_equity'] = equity_empresas_end
        state['net_FB'] = deuda_remanente_fb
        
        state['banks_liquidity'] = liquidez_bancos_end
        state['banks_equity'] = equity_bancos_end
        state['net_BB'] = pasivos_ib_end
        state['households_dividends'] = np.full(H, dividendos_pc)
        state['mask_renacidas'] = mask_quiebra_F
        
        # Reset de Tasas FB para empresas que murieron
        state['rates_FB'][state['mask_renacidas'], :] = 0.0

        # Registro Aggregado
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
                "dr": dr_vector 
            }),
            "households": pd.DataFrame({
                "id": range(H), 
                "dep": state['households_deposits'], 
                "bank_id": state['households_bank']
            }),
            "globals": pd.DataFrame([{
                "t": t,
                "volume_ib": V_total,
                "avg_dr": avg_dr,
                "cascade_size": total_quiebras_B,
                "contagion_loss": total_losses_contagion,
                "total_eq_banks": np.sum(state['banks_equity'])
            }])
        }
        
        # Matrices para grafo
        deposits_hb_matrix = np.zeros((H, B))
        deposits_hb_matrix[np.arange(H), state['households_bank']] = state['households_deposits']
        
        networks = {
            "net_FB": state['net_FB'],
            "net_BB": state['net_BB'],
            "net_FH": state['labor_matrix'],
            "net_HF": consumption_matrix_HF,
            "net_HB": deposits_hb_matrix
        }
        
        if "delta_el" in debug_data:
            networks["matrix_delta_el"] = debug_data["delta_el"]
        
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
