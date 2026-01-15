import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from logger import SimulationLogger
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
plt.style.use("ggplot")
PARAMS = {
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": (10, 6),
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

# Eliminar antigua función guardar_datos_parquet ya que usamos SimulationLogger


def ejecutar_simulacion(modo_impuesto="NINGUNO", semilla=None, run_id="test"):
    """
    Ejecuta una simulación completa del modelo CRISIS con lógica matricial.
    """
    if semilla is not None:
        np.random.seed(semilla)

    # Limpiar/Crear directorio run si es t=0 (hecho por caller o aqui)
    # create_run_dir(run_id)

    # start_time = time.time()
    logger = SimulationLogger()

    # GENERADORES DE HETEROGENEIDAD
    rng = np.random.default_rng(semilla)  # Usar el generador moderno de numpy

    # Función auxiliar para generar distribución Log-Normal (con media target)
    def lognorm_vec(mean_val, sigma, size):
        # Calcular mu para que la media de la distribución sea mean_val
        # E[X] = exp(mu + sigma^2/2) => mu = ln(mean) - sigma^2/2
        mu = np.log(mean_val) - (sigma**2 / 2)
        return rng.lognormal(mu, sigma, size)

    # Función auxiliar para distribución Uniforme centrada
    def uniform_vec(mean_val, spread_pct, size):
        low = mean_val * (1 - spread_pct)
        high = mean_val * (1 + spread_pct)
        return rng.uniform(low, high, size)

    # 1. INICIALIZACIÓN HETEROGÉNEA
    F, B, H = p.F, p.B, p.H

    # Parámetros de dispersión (sigma para lognormal, pct para uniforme)
    SIGMA_SIZE = 0.5  # Dispersión de tamaño (Equity, Producción)
    SPREAD_PRICE = 0.05  # 5% de variación en precios iniciales

    # Generar vectores base
    # Empresas: Tamaños diversos
    prod_ini_vec = lognorm_vec(p.PRODUCCION_INICIAL, SIGMA_SIZE, F)
    # Equity proporcional al tamaño para mantener ratios sanos al inicio
    equity_firms_vec = (prod_ini_vec / p.PRODUCCION_INICIAL) * p.EQUITY_INICIAL_FIRMAS
    liq_firms_vec = (prod_ini_vec / p.PRODUCCION_INICIAL) * p.LIQUIDEZ_INICIAL_FIRMAS

    # Bancos: Tamaños diversos (Power Law es común en bancos)
    equity_banks_vec = lognorm_vec(p.EQUITY_INICIAL_BANCOS, SIGMA_SIZE, B)
    liq_banks_vec = (
        equity_banks_vec / p.EQUITY_INICIAL_BANCOS
    ) * p.LIQUIDEZ_INICIAL_BANCOS

    # Hogares: Riqueza diversa
    deposits_H_vec = lognorm_vec(p.DEPOSITOS_INICIALES_HOGARES, 0.8, H)

    state = {
        # Empresas (Firms)
        "firms_prices": uniform_vec(
            p.PRECIO_INICIAL, SPREAD_PRICE, F
        ),  # Precios ~ Normales
        "firms_production": prod_ini_vec,
        "firms_demand": prod_ini_vec.copy(),  # Asumimos equilibrio inicial
        "firms_inventory": np.zeros(F),
        "firms_equity": equity_firms_vec,
        "firms_liquidity": liq_firms_vec,
        "firms_wage": uniform_vec(p.W_BASE, 0.02, F),  # Salarios casi iguales al inicio
        "firms_target_production": prod_ini_vec.copy(),
        "firms_labor_demand": np.ceil(prod_ini_vec / p.ALPHA).astype(
            int
        ),  # Calculado según prod específica
        "mask_renacidas": np.zeros(F, dtype=bool),
        # Bancos (Banks)
        "banks_equity": equity_banks_vec,
        "banks_liquidity": liq_banks_vec,
        # Hogares (Households)
        "households_deposits": deposits_H_vec,
        "households_bank": rng.integers(0, B, size=H),  # Asignación aleatoria inicial
        "households_dividends": np.zeros(H),
        # Redes (Networks)
        "net_FB": np.zeros((F, B)),
        "net_BB": np.zeros((B, B)),
        "rates_FB": np.zeros((F, B)),
        "rates_BB": np.zeros((B, B)),
        "tax_rates_BB": np.zeros((B, B)), # Tasa de impuesto Interbancario
        "bailout_fund": 0.0,              # Fondo de Rescate (Acumulador SRT)
        # Matrices de Flujo (para logger/stats)
        "labor_matrix": np.zeros((F, H)),
    }

    # Inicializar red de pasivos HETEROGÉNEA
    # Préstamos iniciales proporcionales al tamaño de la empresa
    initial_loans_FB = np.zeros((F, B))
    init_banks_F = rng.integers(0, B, F)

    # El préstamo inicial depende del equity de la empresa (ej. 10% del equity)
    amount_vec_FB = equity_firms_vec * 0.1

    initial_loans_FB[np.arange(F), init_banks_F] = amount_vec_FB
    state["net_FB"] = initial_loans_FB

    # Inicializar red interbancaria (BB) - Libre de escala (tipo Barabási-Albert)
    # El Paper 1 usa redes empíricas o libres de escala. Aleatorio es demasiado homogéneo.
    initial_loans_BB = np.zeros((B, B))

    # 1. Crear un núcleo de bancos conectados (los primeros m bancos totalmente conectados)
    m = 2  # Enlaces por nuevo nodo
    if B > m:
        for i in range(m):
            for j in range(m):
                if i != j:
                    loan_size = (
                        equity_banks_vec[i] * 0.1
                    )  # Préstamos iniciales pequeños
                    initial_loans_BB[i, j] = loan_size

        # 2. Agregar bancos restantes con conexión preferencial (enlaces entrantes ~ grado)
        # Usamos el grado de entrada (quién tiene muchos prestamistas) como proxy de "prominencia" o simplemente aleatorio + grado
        # Por simplicidad en grafo dirigido: conectar a nodos con alto grado total (entrada + salida)

        for i in range(m, B):
            # Calcular probabilidades basadas en el grado actual (suma de conexiones binarias)
            adjacency = (initial_loans_BB > 0).astype(int)
            degrees = np.sum(adjacency, axis=0) + np.sum(
                adjacency, axis=1
            )  # Grado total
            total_degree = np.sum(degrees)

            if total_degree == 0:
                probs = np.ones(B) / B
            else:
                probs = degrees / total_degree

            # Seleccionar m objetivos para PRESTAR (¿i se convierte en acreedor? ¿o deudor?)
            # Usualmente los nuevos entrantes piden prestado a los hubs. Digamos que i pide prestado a hubs existentes.
            # Así que i es Deudor (Fila), Hubs son Acreedores (Columna).

            # Muestrear m objetivos únicos de nodos existentes 0..B-1 (¿pero probs solo no-cero para 0..i-1 típicamente?)
            # Barabasi usualmente crece. Aquí solo elegimos de todos, ponderado por grado actual.
            # Enmascarar a sí mismo
            probs[i] = 0
            norm = np.sum(probs)
            if norm > 0:
                probs /= norm
            else:
                probs = np.ones(B) / B
                probs[i] = 0
                probs /= np.sum(probs)

            targets = rng.choice(np.arange(B), size=m, replace=False, p=probs)

            for t in targets:
                # i pide prestado a t
                loan_size = equity_banks_vec[i] * 0.2
                initial_loans_BB[i, t] = loan_size

    state["net_BB"] = initial_loans_BB

    # Ajustar liquidez (ingresos/desembolsos de préstamos)
    # Las empresas ganan liquidez de los préstamos FB
    state["firms_liquidity"] += np.sum(initial_loans_FB, axis=1)

    # Bancos:
    # - Pierden liquidez por préstamos FB (Prestar a empresas)
    # - Ganan liquidez por préstamos BB (Lado prestatario)
    # - Pierden liquidez por préstamos BB (Lado prestamista)
    state["banks_liquidity"] -= np.sum(initial_loans_FB, axis=0)  # Prestar a empresas
    state["banks_liquidity"] += np.sum(initial_loans_BB, axis=1)  # Lado prestatario
    state["banks_liquidity"] -= np.sum(initial_loans_BB, axis=0)  # Lado prestamista

    # Historia (Aggregates for Plots)
    historia = {
        "t": [],
        "DebtRank_Promedio": [],
        "Total_Equity_Bancos": [],
        "Total_Deuda_Interbancaria": [],
        "Eventos_Cascada_Size": [],
        "Eventos_Perdida_Total": [],
        "Snapshots": {},
        "SRT_Scatter": {},
    }

    # 2. BUCLE
    for t in range(p.T):
        # Paso 1: Planificación (Adaptativo - Mark I)
        update_p1 = paso1(state, p)
        state["firms_target_production"] = update_p1["firms_target_production"]
        state["firms_prices"] = update_p1["firms_prices"]
        state["firms_labor_demand"] = update_p1["firms_labor_demand"]

        # Paso 2: Crédito y Salarios (Matrix FB & BB)
        # Inyectar MODO_IMPUESTO en params dinámicamente
        p.MODO_IMPUESTO = modo_impuesto  # type: ignore

        # Llamada vectorizada
        res_p2 = paso2(state, p)

        # Actualización de Estado (Post-Paso 2)
        state["net_FB"] = res_p2["net_FB"]
        state["net_BB"] = res_p2["net_BB"]
        state["firms_liquidity"] = res_p2["firms_liquidity"]
        state["banks_liquidity"] = res_p2["banks_liquidity"]
        state["firms_labor_demand"] = res_p2["firms_labor_demand"]  # Contratados reales
        state["firms_wages_paid"] = res_p2[
            "wages_paid_vector"
        ]  # Guardar salarios reales para Paso 5
        
        # Actualizar Tasas Interbancarias
        state["rates_BB"] = res_p2["rates_BB"]
        state["tax_rates_BB"] = res_p2["tax_rates_BB"]

        # Actualizar tasas FB donde hubo préstamos nuevos
        # res_p2['new_rates_FB'] vector (F,)
        # res_p2['bank_indices'] vector (F,)
        # Asignar la tasa nueva a la celda correspondiente
        state["rates_FB"][np.arange(F), res_p2["bank_indices"]] = res_p2["new_rates_FB"]

        # Guardar datos scatter SRT
        if "delta_el" in res_p2 and t % 50 == 0:
            historia["SRT_Scatter"][f"t_{t}"] = {
                "Delta_EL": res_p2["delta_el"].copy(),
                "Pasivos_IB": state["net_BB"].copy(),
            }

        # Calcular DebtRank (Reporting)
        # v debe basarse en Pasivos (Liabilities), que son la suma por filas (axis=1)
        total_liabilities = np.sum(state["net_BB"], axis=1)
        V_total = np.sum(total_liabilities)
        if V_total > 1e-6:
            v_sys = total_liabilities / V_total
            dr_vector = calcular_debtrank_vector(
                state["net_BB"], state["banks_equity"], v_sys
            )
            avg_dr = np.mean(dr_vector)
        else:
            dr_vector = np.zeros(B)
            avg_dr = 0.0

        # Paso 3: Producción Física (Ya se pagaron salarios en P2)
        (produccion_real, oferta_bienes, wages_matrix_FH) = paso3(
            state["firms_labor_demand"],  # Contratados
            res_p2["wages_paid_vector"],  # Masa salarial
            state["firms_inventory"],
            p,
        )
        state["firms_production"] = produccion_real
        state["firms_inventory"] = oferta_bienes  # Agregar producción al inventario
        state["labor_matrix"] = wages_matrix_FH

        # --- CORRECCIÓN STOCK-FLOW: PAGO DE SALARIOS A HOGARES ---
        # Sumar los salarios recibidos por cada hogar (sumando sobre empresas)
        total_wages_H = np.sum(wages_matrix_FH, axis=0)  # (H,)
        state["households_deposits"] += total_wages_H
        # ---------------------------------------------------------

        # Paso 4: Consumo (Matrix HF)
        # Factura pagada es wages_paid_vector
        res_p4 = paso4(state, p)

        # IMPORTANTE: Guardamos la demanda para el próximo paso 1
        state["firms_demand"] = res_p4["firms_demand_received"]
        state["firms_inventory"] = res_p4["firms_inventory"]
        state["households_deposits"] = res_p4["households_deposits"]
        state["firms_revenue"] = res_p4["firms_revenue"]  # Necesario para paso5

        # Paso 5: Contabilidad (Matrix FB)
        res_p5 = paso5(state, p)

        # Sincronización final del estado
        state["firms_liquidity"] = res_p5["firms_liquidity"]
        state["firms_equity"] = res_p5["firms_equity"]
        state["net_FB"] = res_p5["net_FB"]

        state["banks_liquidity"] = res_p5["banks_liquidity"]
        state["banks_equity"] = res_p5["banks_equity"]
        state["net_BB"] = res_p5["net_BB"]
        
        # Actualizar Fondo de Rescate
        state["bailout_fund"] = res_p5["bailout_fund"]

        # Actualizar depósitos de hogares (menos costos de rescate)
        state["households_deposits"] = res_p5["households_deposits"]

        dividendos_total = res_p5["dividends_total"]
        dividendos_pc = dividendos_total / H
        state["households_dividends"] = np.full(H, dividendos_pc)
        state["households_deposits"] += state[
            "households_dividends"
        ]  # Pagar dividendos a hogares (Ingreso)

        state["mask_renacidas"] = res_p5["mask_bankrupt_F"]

        # Reset de Tasas FB para empresas que murieron (y renacieron)
        state["rates_FB"][state["mask_renacidas"], :] = 0.0

        # Condición de parada: Todos los bancos insolventes
        if np.sum(state["banks_equity"] > 0) == 0:
            print(
                f"Colapso total del sistema bancario en t={t}. Deteniendo simulación."
            )
            break

        # Registro Aggregado
        total_quiebras_B = res_p5["bankruptcies_B"]
        total_losses_contagion = res_p5["contagion_loss"]

        historia["t"].append(t)
        historia["DebtRank_Promedio"].append(avg_dr)
        historia["Total_Deuda_Interbancaria"].append(V_total)
        historia["Total_Equity_Bancos"].append(np.sum(state["banks_equity"]))

        if total_quiebras_B > 0:
            historia["Eventos_Cascada_Size"].append(int(total_quiebras_B))
            historia["Eventos_Perdida_Total"].append(total_losses_contagion)

        if t == 100 or t == (p.T - 1):
            historia["Snapshots"][f"t_{t}"] = {
                "DebtRank": dr_vector.copy(),
                "Equity_Bancos": state["banks_equity"].copy(),
            }

        # --- PARQUET LOGGING ---
        # Preparación de Datos
        agents = {
            "firms": pd.DataFrame(
                {
                    "id": range(F),
                    "liq": state["firms_liquidity"],
                    "eq": state["firms_equity"],
                    "prod": state["firms_production"],
                }
            ),
            "banks": pd.DataFrame(
                {
                    "id": range(B),
                    "liq": state["banks_liquidity"],
                    "eq": state["banks_equity"],
                    "dr": dr_vector,  # Guardado específicamente para graficar Fig 3b
                }
            ),
            "households": pd.DataFrame(
                {
                    "id": range(H),
                    "dep": state["households_deposits"],
                    "bank_id": state["households_bank"],
                }
            ),
        }

        # Métricas Globales (Escalares)
        # Guardado como un DataFrame de una sola fila para consistencia
        metrics_df = pd.DataFrame(
            [
                {
                    "t": t,
                    "volume_ib": V_total,
                    "avg_dr": avg_dr,
                    "cascade_size": total_quiebras_B,
                    "contagion_loss": total_losses_contagion,
                    "total_eq_banks": np.sum(state["banks_equity"]),
                    "bailout_fund_total": state["bailout_fund"],
                }
            ]
        )

        # Agregar métricas al diccionario de agentes para guardar (truco sucio pero mantiene la firma limpia)
        # o categoría separada. Iteremos agentes como "Datos Tabulares"
        agents["globals"] = metrics_df

        # Construir matrices "completas" para graph
        # Matriz de Depósitos HB: Dispersa
        deposits_hb_matrix = np.zeros((H, B))
        deposits_hb_matrix[np.arange(H), state["households_bank"]] = state[
            "households_deposits"
        ]

        # Reconstrucción de Matriz de Consumo HF
        # res_p4['consumption_flows'] es tupla (empresas_elegidas, gasto_real_H)
        (chosen_firms, actual_spending_H) = res_p4["consumption_flows"]
        consumption_matrix_HF = np.zeros(
            (H, F)
        )  # Advertencia: ¿La matriz densa podría ser pesada? H=1300 F=100 -> 130k elementos. OK.
        # Relleno vectorizado
        # consumption_matrix_HF[h, chosen_firm[h]] = spending[h]
        consumption_matrix_HF[np.arange(H), chosen_firms] = actual_spending_H

        networks = {
            "net_FB": state["net_FB"],
            "net_BB": state["net_BB"],
            "net_FH": state["labor_matrix"],
            "net_HF": consumption_matrix_HF,
            "net_HB": deposits_hb_matrix,
        }

        # Capturar transacciones si están disponibles
        if "transactions" in res_p2 and res_p2["transactions"]:
            # Convertir lista de diccionarios a DataFrame
            df_trans = pd.DataFrame(res_p2["transactions"])
            # Agregar al diccionario de agentes (Logger trata las entradas 'agents' como datos tabulares para guardar como parquet)
            agents["transactions"] = df_trans

        # Registrar datos en el logger
        logger.log_step(t, agents, networks)

    logger.flush(run_id)
    return historia


# =============================================================================
# EXPERIMENTO
# =============================================================================
def run_experiment():
    modes = ["NINGUNO", "TOBIN", "SRT"]

    print(
        f"--- Iniciando Experimento Comparativo ({SIMULATIONS_PER_MODE} runs/modo) ---"
    )

    for mode in modes:
        print(f"Modo: {mode} ", end="")
        for i in range(SIMULATIONS_PER_MODE):
            run_id = f"{mode}_sim_{i}"
            # Ejecutar y guardar en disco
            ejecutar_simulacion(modo_impuesto=mode, semilla=42 + i, run_id=run_id)

            if i % 5 == 0:
                print(".", end="", flush=True)
        print(" OK")


if __name__ == "__main__":
    print("=== Systemic Risk Tax ABM: Orchestrator ===")

    # 1. Ejecutar Simulaciones
    run_experiment()

    print(f"Simulaciones completadas. Datos en ./{OUTPUT_DIR}")
    print("Para generar gráficas, ejecute: python figuras.py")
