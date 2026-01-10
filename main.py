
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

SIMULATIONS_PER_MODE = 20  # Cantidad de simulaciones para suavizado estadístico

# =============================================================================
# MOTOR DE SIMULACIÓN
# =============================================================================
def ejecutar_simulacion(modo_impuesto="NINGUNO", semilla=None):
    """
    Ejecuta una simulación completa del modelo CRISIS.
    """
    if semilla is not None:
        np.random.seed(semilla)
        
    start_time = time.time()
    # print(f"--- Iniciando Simulación: {modo_impuesto} ---")

    # 1. INICIALIZACIÓN
    F, B, H = p.F, p.B, p.H
    
    precios = np.full(F, p.PRECIO_INICIAL)
    produccion = np.full(F, p.PRODUCCION_INICIAL)
    ventas = produccion.copy()
    inventario = np.zeros(F)
    
    equity_empresas = np.full(F, p.EQUITY_INICIAL_FIRMAS)
    liquidez_empresas = np.full(F, p.LIQUIDEZ_INICIAL_FIRMAS)
    deuda_empresas = np.zeros(F)
    
    banco_acreedor_empresa = np.random.randint(0, B, size=F)
    tasa_empresas = np.full(F, p.R_BAR)

    equity_bancos = np.full(B, p.EQUITY_INICIAL_BANCOS)
    liquidez_bancos = np.full(B, p.LIQUIDEZ_INICIAL_BANCOS)
    
    pasivos_interbancarios = np.zeros((B, B))
    tasas_interbancarias = np.zeros((B, B)) 

    depositos_hogares = np.full(H, p.DEPOSITOS_INICIALES_HOGARES)
    dividendos_previos = np.zeros(H)
    mask_renacidas = np.zeros(F, dtype=bool)

    # Historia
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

        # Paso 2: Crédito
        (nuevos_prestamos, tasas_finales, nueva_matriz_ib, 
         liquidez_bancos_post, bancos_elegidos, matriz_impuestos, debug_data) = paso2(
            demanda_credito, liquidez_bancos, equity_bancos,
            pasivos_interbancarios, equity_empresas, deuda_empresas,
            modo_impuesto=modo_impuesto
        )
        
        # Guardar datos scatter SRT
        if "delta_el" in debug_data and t % 50 == 0:
            historia["SRT_Scatter"][f"t_{t}"] = {
                "Delta_EL": debug_data["delta_el"].copy(),
                "Pasivos_IB": pasivos_interbancarios.copy()
            }

        # Actualizar estado financiero intermedio
        deuda_empresas += nuevos_prestamos
        tasa_empresas = tasas_finales
        banco_acreedor_empresa = bancos_elegidos
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

        # Paso 3: Producción
        (produccion_real, oferta_bienes, empleo_real, 
         factura_pagada, liquidez_empresas_post) = paso3(
            demanda_laboral, liquidez_empresas, nuevos_prestamos, inventario
        )
        produccion = produccion_real
        liquidez_empresas = liquidez_empresas_post
        
        # Paso 4: Consumo
        (ventas_real, ingresos_ventas, inventario_final, 
         depositos_post, demanda_teorica, _) = paso4(
            precios, oferta_bienes, factura_pagada, depositos_hogares,
            dividendos_previos
        )
        ventas = ventas_real
        inventario = inventario_final
        depositos_hogares = depositos_post
        
        # Paso 5: Contabilidad
        (liquidez_empresas_end, equity_empresas_end, deuda_empresas_end, 
         mask_quiebra_F, liquidez_bancos_end, equity_bancos_end, 
         mask_quiebra_B, pasivos_ib_end, dividendos_pc, 
         total_quiebras_B, total_losses_contagion) = paso5(
            liquidez_empresas, ingresos_ventas, deuda_empresas, tasa_empresas,
            equity_empresas, banco_acreedor_empresa,
            liquidez_bancos, equity_bancos, pasivos_interbancarios, tasas_interbancarias,
            depositos_hogares,
            tax_matrix_ib=matriz_impuestos
        )
        
        # Actualización final
        liquidez_empresas = liquidez_empresas_end
        equity_empresas = equity_empresas_end
        deuda_empresas = deuda_empresas_end
        liquidez_bancos = liquidez_bancos_end
        equity_bancos = equity_bancos_end
        pasivos_interbancarios = pasivos_ib_end
        dividendos_previos = np.full(H, dividendos_pc)
        mask_renacidas = mask_quiebra_F

        # Registro
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

    # print(f"Simulacion finalizada ({mode_label}) en {time.time()-start_time:.2f}s")
    return historia

# =============================================================================
# EXPERIMENTO Y PLOTTING
# =============================================================================
def run_experiment():
    modes = ["NINGUNO", "TOBIN", "SRT"]
    colors = {"NINGUNO": "red", "TOBIN": "blue", "SRT": "green"}
    
    results = {m: {
        "debtrank_profiles": [],
        "scatter_data": [],
        "total_losses": [],
        "cascade_sizes": [],
        "volumes": []
    } for m in modes}

    print(f"--- Iniciando Experimento Comparativo ({SIMULATIONS_PER_MODE} runs/modo) ---")
    
    for mode in modes:
        print(f"Modo: {mode} ", end="")
        for i in range(SIMULATIONS_PER_MODE):
            h = ejecutar_simulacion(modo_impuesto=mode, semilla=42+i)
            
            # Recolectar datos
            # Fig 3b
            if "t_499" in h["Snapshots"]:
                results[mode]["debtrank_profiles"].append(h["Snapshots"]["t_499"]["DebtRank"])
            
            # Fig 3d
            for k, data in h["SRT_Scatter"].items():
                l = data["Pasivos_IB"].flatten()
                d = data["Delta_EL"].flatten()
                mask = (l > 1e-6) & (d > 1e-9)
                if np.any(mask):
                    results[mode]["scatter_data"].append((l[mask], d[mask]))
            
            # Fig 4
            results[mode]["total_losses"].append(np.sum(h["Eventos_Perdida_Total"]))
            
            cascades = h["Eventos_Cascada_Size"]
            if cascades:
                results[mode]["cascade_sizes"].extend(cascades)
            else:
                results[mode]["cascade_sizes"].append(0)
            
            results[mode]["volumes"].append(np.mean(h["Total_Deuda_Interbancaria"]))
            
            if i % 5 == 0: print(".", end="", flush=True)
        print(" OK")

    return results, colors

def plot_fig_3b(results, colors):
    plt.figure()
    B = p.B
    x = np.arange(1, B + 1)
    bar_width = 0.25
    offsets = {"NINGUNO": -bar_width, "TOBIN": 0, "SRT": bar_width}
    
    for mode in results:
        data = results[mode]["debtrank_profiles"]
        if not data: continue
        sorted_runs = np.sort(np.array(data), axis=1)[:, ::-1]
        avg_profile = np.mean(sorted_runs, axis=0)
        std_profile = np.std(sorted_runs, axis=0)
        
        plt.bar(x + offsets[mode], avg_profile, width=bar_width, 
                color=colors[mode], label=mode, alpha=0.8, yerr=std_profile, capsize=2)
        
    plt.xlabel('Bank Rank (by DebtRank)')
    plt.ylabel('DebtRank $R_i$')
    plt.title('Fig 3b: Perfil de Riesgo Sistémico')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_3b_DebtRank.png")
    plt.close()

def plot_fig_3d(results, colors):
    plt.figure()
    for mode in results:
        if not results[mode]["scatter_data"]: continue
        
        all_l, all_d = [], []
        for l, d in results[mode]["scatter_data"]:
            all_l.extend(l)
            all_d.extend(d)
        
        plt.scatter(all_l, all_d, color=colors[mode], alpha=0.3, label=mode, s=10, edgecolors='none')
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Loan Size (Log)')
    plt.ylabel(r'$\Delta EL^{syst}$ (Log)')
    plt.title('Fig 3d: Contribución Marginal al Riesgo')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_3d_Scatter.png")
    plt.close()

def plot_fig_4(results, colors):
    # 4a Losses
    plt.figure()
    for mode in results:
        plt.hist(results[mode]["total_losses"], bins=15, color=colors[mode], alpha=0.5, label=mode, density=True)
    plt.title('Fig 4a: Pérdidas Totales')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/Fig_4a_Losses.png")
    plt.close()

    # 4b Cascades
    plt.figure()
    bins = np.arange(0, p.B + 2) - 0.5
    for mode in results:
        plt.hist(results[mode]["cascade_sizes"], bins=bins, color=colors[mode], alpha=0.5, label=mode, density=True, histtype='stepfilled')
    plt.title('Fig 4b: Tamaño de Cascadas')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/Fig_4b_Cascades.png")
    plt.close()

    # 4c Volume
    plt.figure()
    for mode in results:
        plt.hist(results[mode]["volumes"], bins=15, color=colors[mode], alpha=0.5, label=mode, density=True)
    plt.title('Fig 4c: Volumen Interbancario')
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/Fig_4c_Volume.png")
    plt.close()

if __name__ == "__main__":
    print("=== Systemic Risk Tax ABM: Orchestrator ===")
    data, colors = run_experiment()
    print("Generando Gráficos...")
    plot_fig_3b(data, colors)
    plot_fig_3d(data, colors)
    plot_fig_4(data, colors)
    print(f"Listo. Resultados en ./{OUTPUT_DIR}")
