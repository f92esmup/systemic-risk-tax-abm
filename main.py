# main.py
import numpy as np
import matplotlib.pyplot as plt
from parametros import Param as p

# Importar los pasos lógicos
from logica.paso1 import paso1
from logica.paso2 import paso2
from logica.paso3 import paso3
from logica.paso4 import paso4
from logica.paso5_6_7 import paso5


def ejecutar_simulacion(modo_impuesto="NINGUNO"):
    """
    Ejecuta una simulación completa del modelo CRISIS.

    Args:
        modo_impuesto (str): "NINGUNO", "TOBIN", "SRT"

    Returns:
        dict: Diccionario con históricos de datos para análisis.
    """
    print(f"--- Iniciando Simulación: Modo {modo_impuesto} ---")

    # =========================================================================
    # 1. INICIALIZACIÓN
    # =========================================================================
    np.random.seed(42)  # Reproducibilidad

    # --- Empresas (F) ---
    precios = np.random.uniform(p.PRECIO_INICIAL * 0.9, p.PRECIO_INICIAL * 1.1, p.F)
    produccion = np.random.uniform(
        p.PRODUCCION_INICIAL * 0.9, p.PRODUCCION_INICIAL * 1.1, p.F
    )
    ventas = produccion.copy()  # Asumimos equilibrio inicial

    liquidez_empresas = np.full(p.F, p.LIQUIDEZ_INICIAL_FIRMAS)
    equity_empresas = np.full(p.F, p.EQUITY_INICIAL_FIRMAS)
    deuda_empresas = np.zeros(p.F)  # Empiezan sin deuda para calentar motores
    tasa_empresas = np.full(p.F, p.R_BAR)
    inventario = np.zeros(p.F)

    # Asignación inicial de banco acreedor (aleatorio)
    banco_acreedor = np.random.randint(0, p.B, p.F)

    mask_renacidas = np.zeros(p.F, dtype=bool)  # Nadie acaba de quebrar en t=0

    # --- Bancos (B) ---
    liquidez_bancos = np.full(p.B, p.LIQUIDEZ_INICIAL_BANCOS)
    equity_bancos = np.full(p.B, p.EQUITY_INICIAL_BANCOS)

    # Matriz de Pasivos Interbancarios (B x B)
    # Inicialmente vacía o aleatoria muy escasa
    pasivos_interbancarios = np.zeros((p.B, p.B))
    tasas_interbancarias = np.full((p.B, p.B), p.R_BAR)

    # --- Hogares (H) ---
    depositos_hogares = np.full(p.H, p.DEPOSITOS_INICIALES_HOGARES)
    dividendos_per_capita = 0.0

    # --- Historial de Datos ---
    historia = {
        "PIB": [],  # Producción total real
        "Quiebras_F": [],  # Cantidad de empresas quebradas
        "Quiebras_B": [],  # Cantidad de bancos quebrados
        "Total_Deuda": [],  # Deuda total empresas
        "Volumen_IB": [],  # Volumen préstamos interbancarios
    }

    # =========================================================================
    # 2. BUCLE TEMPORAL (Time Loop)
    # =========================================================================
    for t in range(p.T):
        if t % 50 == 0:
            print(f"Paso {t}/{p.T}...")

        # ---------------------------------------------------------------------
        # Paso 1: Planificación (Precios y Producción Deseada)
        # ---------------------------------------------------------------------
        (
            nuevos_precios,
            demanda_laboral,
            produccion_necesaria,
            demanda_credito,
            factura_salarial,
            demanda_objetivo,
        ) = paso1(
            precios_prev=precios,
            produccion_prev=produccion,
            ventas_prev=ventas,
            liquidez_prev=liquidez_empresas,
            inventario_acumulado=inventario,
            mask_renacidas=mask_renacidas,
        )

        # Actualizamos precios para el periodo
        precios = nuevos_precios

        # ---------------------------------------------------------------------
        # Paso 2: Mercado de Crédito (Bancos y SRT)
        # ---------------------------------------------------------------------
        (
            nuevos_prestamos,
            tasas_elegidas,
            pasivos_interbancarios,
            liquidez_bancos,
            bancos_elegidos,
        ) = paso2(
            demanda_credito=demanda_credito,
            liquidez_bancos=liquidez_bancos,
            equity_bancos=equity_bancos,
            pasivos_interbancarios=pasivos_interbancarios,
            equity_empresas=equity_empresas,
            deuda_empresas=deuda_empresas,
            modo_impuesto=modo_impuesto,
        )

        # Actualizamos quién es el banco de cada empresa y su nueva tasa
        # Nota: Simplificación -> Si la empresa ya tenía deuda con banco X y pide más a Y,
        # idealmente se modelan múltiples préstamos. Aquí consolidamos al nuevo banco
        # o mantenemos el viejo si no hubo préstamo nuevo.
        mask_hubo_prestamo = nuevos_prestamos > 0
        banco_acreedor[mask_hubo_prestamo] = bancos_elegidos[mask_hubo_prestamo]
        tasa_empresas[mask_hubo_prestamo] = tasas_elegidas[mask_hubo_prestamo]

        # Acumulamos deuda nueva
        deuda_empresas += nuevos_prestamos

        # ---------------------------------------------------------------------
        # Paso 3: Producción Real (Labour Market)
        # ---------------------------------------------------------------------
        (
            produccion_real,
            oferta_total_bienes,
            empleo_real,
            factura_salarial_real,
            liquidez_empresas,
        ) = paso3(
            demanda_laboral_objetivo=demanda_laboral,
            liquidez_previa=liquidez_empresas,
            nuevos_prestamos=nuevos_prestamos,
            inventario_acumulado=inventario,
        )

        # Actualizamos la producción actual (Y)
        produccion = produccion_real

        # ---------------------------------------------------------------------
        # Paso 4: Consumo (Goods Market)
        # ---------------------------------------------------------------------
        (
            ventas_cantidad_real,
            ingresos_ventas,
            inventario_final,
            depositos_hogares,
            demanda_teorica,
            ingreso_salarial_per_capita,
        ) = paso4(
            precios_actuales=precios,
            oferta_total_bienes=oferta_total_bienes,
            factura_salarial_real=factura_salarial_real,
            depositos_hogares=depositos_hogares,
            dividendos_previos=dividendos_per_capita,
        )

        # Actualizamos inventarios y ventas para el siguiente t
        inventario = inventario_final
        ventas = ventas_cantidad_real

        # ---------------------------------------------------------------------
        # Paso 5: Contabilidad y Quiebras
        # ---------------------------------------------------------------------
        (
            liquidez_empresas,
            equity_empresas,
            deuda_empresas,
            mask_quiebra_F,
            liquidez_bancos,
            equity_bancos,
            mask_quiebra_B,
            pasivos_interbancarios,
            dividendos_per_capita,
        ) = paso5(
            liquidez_empresas=liquidez_empresas,
            ingresos_ventas=ingresos_ventas,
            deuda_empresas=deuda_empresas,
            tasa_empresas=tasa_empresas,
            equity_empresas=equity_empresas,
            banco_acreedor_empresa=banco_acreedor,
            liquidez_bancos=liquidez_bancos,
            equity_bancos=equity_bancos,
            pasivos_interbancarios=pasivos_interbancarios,
            tasas_interbancarias=tasas_interbancarias,
            depositos_hogares=depositos_hogares,
        )

        # Guardar estado 'renacidas' para el próximo paso 1
        mask_renacidas = mask_quiebra_F

        # ---------------------------------------------------------------------
        # Recolección de Datos
        # ---------------------------------------------------------------------
        historia["PIB"].append(np.sum(produccion_real))
        historia["Quiebras_F"].append(np.sum(mask_quiebra_F))
        historia["Quiebras_B"].append(np.sum(mask_quiebra_B))
        historia["Total_Deuda"].append(np.sum(deuda_empresas))
        historia["Volumen_IB"].append(np.sum(pasivos_interbancarios))

    return historia


# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================
if __name__ == "__main__":
    # 1. Correr escenario Base (Sin Impuesto)
    datos_base = ejecutar_simulacion("NINGUNO")

    # 2. Correr escenario SRT (Systemic Risk Tax)
    datos_srt = ejecutar_simulacion("SRT")

    # 3. Graficar Resultados Comparativos
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # PIB
    ax[0, 0].plot(datos_base["PIB"], label="Sin Impuesto", color="red", alpha=0.7)
    ax[0, 0].plot(datos_srt["PIB"], label="SRT", color="green", alpha=0.7)
    ax[0, 0].set_title("Producción Total (PIB)")
    ax[0, 0].legend()

    # Quiebras Bancarias (Cascadas)
    ax[0, 1].plot(datos_base["Quiebras_B"], label="Sin Impuesto", color="red")
    ax[0, 1].plot(datos_srt["Quiebras_B"], label="SRT", color="green")
    ax[0, 1].set_title("Quiebras Bancarias")

    # Deuda Total
    ax[1, 0].plot(datos_base["Total_Deuda"], label="Sin Impuesto", color="red")
    ax[1, 0].plot(datos_srt["Total_Deuda"], label="SRT", color="green")
    ax[1, 0].set_title("Deuda Corporativa Total")

    # Volumen Interbancario
    ax[1, 1].plot(datos_base["Volumen_IB"], label="Sin Impuesto", color="red")
    ax[1, 1].plot(datos_srt["Volumen_IB"], label="SRT", color="green")
    ax[1, 1].set_title("Volumen Mercado Interbancario")

    plt.tight_layout()
    plt.show()

    print("Simulación finalizada.")
