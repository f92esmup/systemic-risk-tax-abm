
import numpy as np
import pandas as pd
import time
from estado import EstadoEconomia
from logica import (
    paso1_planificacion_empresas, paso2_prestamos_bancarios, paso3_produccion, 
    paso4_consumo, paso5_repago_empresas, paso6_mercado_interbancario, 
    paso7_evolucion, ejecutar_contagio_interbancario
)
from parametros import *

class EjecutorSimulacion:
    def __init__(self, pasos=PASOS_SIMULACION):
        self.pasos = pasos
        self.resultados = {}

    def ejecutar_escenario(self, nombre_escenario, modo_impuesto):
        print(f"\n>>> Iniciando Escenario: {nombre_escenario} (Modo Impuesto: {modo_impuesto})")
        tiempo_inicio = time.time()
        
        # Inicializar Estado
        estado = EstadoEconomia()
        
        # Historial de métricas
        historial = {
            'paso': [],
            'produccion_total': [],
            'consumo_total': [],
            'defaults_empresas': [],
            'deuda_total': [],
            'riesgo_sistemico': [],
            'impuesto_recaudado': [],
            'patrimonio_bancos_total': []
        }
        
        for t in range(self.pasos):
            # --- Paso 1: Planificación ---
            estado = paso1_planificacion_empresas(estado)
            
            # --- Paso 2: Crédito ---
            estado = paso2_prestamos_bancarios(estado)
            
            # --- Paso 3: Producción ---
            estado = paso3_produccion(estado)
            
            # --- Paso 4: Consumo ---
            estado = paso4_consumo(estado)
            
            # --- Paso 5: Repago y Quiebras Empresas ---
            estado = paso5_repago_empresas(estado)
            
            # --- Paso 5b: Contagio Interbancario (Shock Externo) ---
            estado = ejecutar_contagio_interbancario(estado)
            
            # --- Paso 6: Interbancario y Tax ---
            estado = paso6_mercado_interbancario(estado, modo_impuesto=modo_impuesto)
            
            # --- Paso 7: Evolución ---
            estado = paso7_evolucion(estado)
            
            # --- Recolección de Datos ---
            historial['paso'].append(t)
            historial['produccion_total'].append(np.sum(estado.stock_empresas))
            
            # Usamos ventas_diarias si existe (lo añadimos en logica.py y estado.py)
            ventas = np.sum(estado.ventas_diarias_empresas) if hasattr(estado, 'ventas_diarias_empresas') else 0.0
            historial['consumo_total'].append(ventas)
            
            historial['defaults_empresas'].append(np.sum(estado.defaults_acumulados_empresas))
            historial['deuda_total'].append(np.sum(estado.prestamos_banco_empresa))
            historial['riesgo_sistemico'].append(estado.riesgo_sistemico_total)
            historial['impuesto_recaudado'].append(estado.impuesto_recaudado)
            historial['patrimonio_bancos_total'].append(np.sum(estado.patrimonio_bancos))
            
            if t % 10 == 0:
                print(f"   Paso {t}/{self.pasos} - SR: {estado.riesgo_sistemico_total:.2f} - Impuesto: {estado.impuesto_recaudado:.2f}")

        tiempo_transcurrido = time.time() - tiempo_inicio
        print(f"<<< Escenario {nombre_escenario} completado en {tiempo_transcurrido:.2f}s")
        
        df = pd.DataFrame(historial)
        self.resultados[nombre_escenario] = df
        return df

    def guardar_resultados(self, nombre_archivo="resultados_simulacion.csv"):
        # Concatenar todos los escenarios con columna Escenario
        todos_dfs = []
        for nombre, df in self.resultados.items():
            df['Escenario'] = nombre
            todos_dfs.append(df)
        
        if todos_dfs:
            df_final = pd.concat(todos_dfs)
            df_final.to_csv(nombre_archivo, index=False)
            print(f"Resultados guardados en {nombre_archivo}")
        else:
            print("No hay resultados para guardar.")

if __name__ == "__main__":
    # Configuración de simulación rápida
    sim = EjecutorSimulacion(pasos=PASOS_SIMULACION)
    
    # 1. Referencia (Sin Impuesto)
    sim.ejecutar_escenario("Referencia (Sin Impuesto)", modo_impuesto='NINGUNO')
    
    # 2. ITF (Impuesto Transacción Financiera)
    sim.ejecutar_escenario("ITF (Tasa Tobin)", modo_impuesto='ITF')
    
    # 3. IRS (Impuesto Riesgo Sistémico)
    sim.ejecutar_escenario("IRS (Impuesto Riesgo Sistémico)", modo_impuesto='IRS')
    
    sim.guardar_resultados()
