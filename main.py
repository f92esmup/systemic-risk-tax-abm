# Script principal. Orquestador de la simulación del Modelo CRISIS.
# Ejecuta el bucle temporal completo para los 3 modos: NONE, TOBIN, SRT.

import numpy as np
import matplotlib.pyplot as plt
from parametros import Param as p

# Importación de Módulos de Lógica
from logica.paso1 import paso1
from logica.paso2 import paso2_mercado_credito, paso2_interbancario
from logica.paso3 import paso3_produccion_y_mercado_laboral
from logica.paso4 import paso4_consumo
from logica.paso5 import paso5_resultados_y_quiebras
from logica.paso6 import paso6_repago_deuda
from logica.paso7 import paso7_cierre_y_metricas

# Configuración de Tiempo
T_STEPS = 500
N_RUNS = 100  # Número de ejecuciones Monte CarloN_RUNS = 10 # Número de ejecuciones Monte Carlo


def run_simulation(mode_name, run_id=0):
    print(f"\n==========================================")
    print(f"--- INICIANDO SIMULACIÓN: MODO {mode_name} (Run {run_id}) ---")
    print(f"==========================================\n")

    # Configurar modo en Parametros
    p.TAX_MODE = mode_name

    # ==========================================
    # 0. INICIALIZACIÓN DEL ESTADO (t=0)
    # ==========================================

    # MACRO: Estable pero competitiva
    firm_ids = np.arange(p.F)
    firm_prices = np.random.uniform(1.0, 1.2, p.F)
    firm_produccion = np.random.uniform(5.0, 8.0, p.F)
    firm_ventas = firm_produccion * np.random.uniform(0.9, 1.0, p.F)
    firm_inventario = np.maximum(firm_produccion - firm_ventas, 0)

    # Liquidez FIRMAS: Tensa pero sobrevivible (cubre ~5-10 salarios)
    firm_liquidez = np.random.uniform(80, 150, p.F)

    # Deuda FIRMAS: Alta (40-80). Pago intereses ~ 2-4.
    firm_deuda_inicial_total = np.random.uniform(40, 80, p.F)

    # BANCOS: Zona de Riesgo "Ricitos de Oro"
    bancos_ids = np.arange(p.B)
    bancos_liquidez = np.random.uniform(500, 1000, p.B)

    # Capital: 12-25. Suficiente para absorber un par de impagos pequeños,
    # pero vulnerable a un impago grande (Interbancario > 15).
    bancos_patrimonio = np.random.uniform(12, 25, p.B)
    bancos_depositos = np.random.uniform(1000, 2000, p.B)

    # Red Interbancaria: Peligrosa
    matriz_interbancaria = np.zeros((p.B, p.B))
    matriz_intereses_ib = np.zeros((p.B, p.B))
    matriz_prestamos_firmas = np.zeros((p.F, p.B))
    matriz_intereses_firmas = np.zeros((p.F, p.B))

    mask_deuda = np.random.rand(p.B, p.B) > 0.70
    # Exposiciones grandes (10-30). Si un banco debe 30 y quiebra,
    # arrastra a un acreedor con capital 20.
    matriz_interbancaria[mask_deuda] = np.random.uniform(10, 30, np.sum(mask_deuda))
    np.fill_diagonal(matriz_interbancaria, 0)
    bancos_deuda_acumulada = np.sum(matriz_interbancaria, axis=1)

    for f in range(p.F):
        monto = firm_deuda_inicial_total[f]
        if monto > 0:
            bancos_acreedores = np.random.choice(bancos_ids, 2, replace=False)
            matriz_prestamos_firmas[f, bancos_acreedores] += monto / 2

    fondo_rescate_acumulado = 0.0

    # Hogares
    hogares_ids = np.arange(p.H)
    hogares_liquidez = np.random.uniform(15, 30, p.H)
    hogares_es_trabajador = np.ones(p.H, dtype=bool)
    indices_duenos = np.arange(p.F + p.B)
    hogares_es_trabajador[indices_duenos] = False
    hogares_empleo_estado = np.full(p.H, -1, dtype=int)

    indices_obreros = np.where(hogares_es_trabajador)[0]
    num_ocupados_inicial = int(len(indices_obreros) * 0.90)
    obreros_activos = np.random.choice(
        indices_obreros, size=num_ocupados_inicial, replace=False
    )
    empresas_asignadas = np.random.randint(0, p.F, size=num_ocupados_inicial)
    hogares_empleo_estado[obreros_activos] = empresas_asignadas

    # Historial
    historia = {
        "pib": [],
        "desempleo": [],
        "quiebras_firmas": [],
        "quiebras_bancos": [],
        "deuda_privada": [],
        "fondo_rescate": [],
        "sr_total": [],
    }

    # ==========================================
    # BUCLE TEMPORAL
    # ==========================================

    for t in range(T_STEPS):
        # PASO 1
        nuevos_precios, demanda_esperada, demanda_trabajo, factura_esperada_salarial = (
            paso1(firm_prices, firm_produccion, firm_ventas)
        )

        # PASO 2
        firm_deuda_actual = np.sum(matriz_prestamos_firmas, axis=1)
        contratos_potenciales, demanda_credito_empresas = paso2_mercado_credito(
            firm_ids,
            firm_liquidez,
            firm_deuda_actual,
            factura_esperada_salarial,
            bancos_ids,
        )
        nuevos_prestamos_ib, contratos_finales_empresas, bancos_liquidez = (
            paso2_interbancario(
                bancos_ids,
                bancos_liquidez,
                bancos_patrimonio,
                bancos_depositos,
                bancos_deuda_acumulada,
                contratos_potenciales,
                matriz_interbancaria,
                tax_mode=p.TAX_MODE,
            )
        )

        # Actualizar balances
        firm_costo_financiero_iteracion = np.zeros(p.F)
        if len(contratos_finales_empresas) > 0:
            idx_f = contratos_finales_empresas[:, 0].astype(int)
            idx_b = contratos_finales_empresas[:, 1].astype(int)
            montos = contratos_finales_empresas[:, 2]
            tasas = contratos_finales_empresas[:, 3]
            intereses = montos * tasas
            total = montos + intereses
            np.add.at(firm_liquidez, idx_f, montos)
            np.add.at(firm_costo_financiero_iteracion, idx_f, intereses)
            for f, b, tot, intr in zip(idx_f, idx_b, total, intereses):
                matriz_prestamos_firmas[f, b] += tot
                matriz_intereses_firmas[f, b] += intr

        if len(nuevos_prestamos_ib) > 0:
            lenders = nuevos_prestamos_ib[:, 0].astype(int)
            borrowers = nuevos_prestamos_ib[:, 1].astype(int)
            amounts = nuevos_prestamos_ib[:, 2]
            total_rates = nuevos_prestamos_ib[:, 3]
            tax_rates = nuevos_prestamos_ib[:, 4]
            real_rates = total_rates - tax_rates
            interest_lender = amounts * real_rates
            total_lender = amounts + interest_lender
            tax_vals = amounts * tax_rates
            fondo_rescate_acumulado += np.sum(tax_vals)
            np.add.at(bancos_liquidez, borrowers, -tax_vals)
            for l, b, tot, intr in zip(
                lenders, borrowers, total_lender, interest_lender
            ):
                matriz_interbancaria[b, l] += tot
                matriz_intereses_ib[b, l] += intr

        # PASO 3
        (
            firm_produccion_real,
            firm_trabajadores_reales,
            firm_coste_salarial,
            hogares_ingresos_nomina,
            firm_liquidez,
            hogares_empleo_estado,
        ) = paso3_produccion_y_mercado_laboral(
            demanda_trabajo, firm_liquidez, hogares_empleo_estado, hogares_es_trabajador
        )
        hogares_liquidez += hogares_ingresos_nomina

        # PASO 4
        (
            firm_ventas_reales,
            firm_inventario_final,
            firm_ingresos,
            hogares_liquidez,
            hogares_gasto_total,
        ) = paso4_consumo(
            hogares_liquidez, nuevos_precios, firm_produccion_real, firm_inventario
        )
        firm_liquidez += firm_ingresos

        # PASO 5
        deuda_total_f = np.sum(matriz_prestamos_firmas, axis=1)
        obligacion_f = deuda_total_f * p.tau
        indices_quiebra_detectados = np.where((firm_liquidez - obligacion_f) < -1e-5)[0]

        (
            firm_liquidez,
            bancos_patrimonio,
            bancos_liquidez,
            bancos_activos,
            matriz_prestamos_firmas,
            matriz_interbancaria,
            hogares_liquidez,
            num_quiebras_firmas,
            num_quiebras_bancos,
        ) = paso5_resultados_y_quiebras(
            firm_liquidez,
            firm_ingresos,
            firm_coste_salarial,
            firm_costo_financiero_iteracion,
            matriz_prestamos_firmas,
            matriz_intereses_firmas,
            bancos_liquidez,
            bancos_patrimonio,
            matriz_interbancaria,
            matriz_intereses_ib,
            hogares_liquidez,
            np.arange(p.F),
            np.arange(p.F, p.F + p.B),
            fondo_rescate_acumulado,
            p.tau,
        )

        # PASO 6
        (
            firm_liquidez,
            bancos_liquidez,
            matriz_prestamos_firmas,
            matriz_intereses_firmas,
            matriz_interbancaria,
            matriz_intereses_ib,
            repago_f,
            repago_b,
        ) = paso6_repago_deuda(
            firm_liquidez,
            matriz_prestamos_firmas,
            matriz_intereses_firmas,
            bancos_liquidez,
            matriz_interbancaria,
            matriz_intereses_ib,
            p.tau,
        )

        # PASO 7
        firm_deuda_actual = np.sum(matriz_prestamos_firmas, axis=1)
        (firm_prices, firm_produccion, firm_ventas, firm_inventario, metricas) = (
            paso7_cierre_y_metricas(
                indices_quiebra_detectados,
                firm_ids,
                nuevos_precios,
                firm_produccion_real,
                firm_ventas_reales,
                firm_inventario_final,
                firm_liquidez,
                firm_deuda_actual,
                bancos_activos,
                fondo_rescate_acumulado,
                firm_trabajadores_reales,
                hogares_es_trabajador,
            )
        )

        historia["pib"].append(metricas["pib"])
        historia["desempleo"].append(metricas["desempleo"])
        historia["quiebras_firmas"].append(metricas["quiebras_firmas"])
        historia["quiebras_bancos"].append(p.B - metricas["bancos_vivos"])
        historia["deuda_privada"].append(metricas["deuda_total"])
        historia["fondo_rescate"].append(metricas["fondo_rescate"])

        if t % 50 == 0:
            print(
                f"Iter {t:3d} | PIB: {metricas['pib']:7.2f} | Bancos: {metricas['bancos_vivos']:2d} | "
                f"Fondo: {metricas['fondo_rescate']:5.2f}"
            )

    filename = f"output_{mode_name}_run{run_id}.txt"
    with open(filename, "w") as f:
        f.write("Step,PIB,Desempleo,QuiebrasF,BancosMuertos,FondoRescate\n")
        for t in range(T_STEPS):
            f.write(
                f"{t},{historia['pib'][t]},{historia['desempleo'][t]},"
                f"{historia['quiebras_firmas'][t]},{historia['quiebras_bancos'][t]},"
                f"{historia['fondo_rescate'][t]}\n"
            )
    print(f"Resultados guardados en '{filename}'")
    return historia


modes = ["NONE", "TOBIN", "SRT"]
monte_carlo_results = {m: [] for m in modes}

for m in modes:
    for r in range(N_RUNS):
        # print(f"--- Ejecutando {m} - Run {r+1}/{N_RUNS} ---") # Ya se imprime dentro de run_simulation
        sim_data = run_simulation(m, run_id=r)
        monte_carlo_results[m].append(sim_data)

print("\n=== RESULTADOS MONTE CARLO (Promedios Finales) ===")
for m in modes:
    runs = monte_carlo_results[m]
    # Promedio del PIB de los ultimos 50 pasos, promediado entre runs
    pib_finales = [np.mean(h["pib"][-50:]) for h in runs]
    muertos_finales = [h["quiebras_bancos"][-1] for h in runs]
    fondos_finales = [h["fondo_rescate"][-1] for h in runs]

    print(f"Modo {m:6s}:")
    print(
        f"  PIB (ult 50): Media {np.mean(pib_finales):.2f} | Std {np.std(pib_finales):.2f}"
    )
    print(
        f"  Bancos Muertos: Media {np.mean(muertos_finales):.2f} | Std {np.std(muertos_finales):.2f}"
    )
    print(
        f"  Fondo Rescate: Media {np.mean(fondos_finales):.2f} | Std {np.std(fondos_finales):.2f}"
    )
    print(
        f"  Fondo Rescate: Media {np.mean(fondos_finales):.2f} | Std {np.std(fondos_finales):.2f}"
    )
