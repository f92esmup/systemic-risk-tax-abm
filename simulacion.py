
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


class RecolectorDatos:
    def __init__(self, nombre_archivo="relaciones_sistema.csv"):
        self.nombre_archivo = nombre_archivo
        self.datos_buffer = []
        # Escribir cabecera
        with open(self.nombre_archivo, "w") as f:
            f.write("Step,Source_Type,Source_ID,Target_Type,Target_ID,Weight,Relation_Type\n")

    def capturar_paso(self, estado, step):
        nuevas_filas = []

        # 1. Crédito: Banco -> Empresa (Weight=Monto Deuda)
        banco_idxs, empresa_idxs = np.where(estado.prestamos_banco_empresa > 1e-2)
        montos = estado.prestamos_banco_empresa[banco_idxs, empresa_idxs]
        for b, e, m in zip(banco_idxs, empresa_idxs, montos):
            nuevas_filas.append(f"{step},Banco,{b},Empresa,{e},{m:.2f},CREDITO")

        # 2. Interbancario: Banco -> Banco (Weight=Monto Deuda)
        b_src, b_tgt = np.where(estado.matriz_interbancaria > 1e-2)
        montos_ib = estado.matriz_interbancaria[b_src, b_tgt]
        for s, t, m in zip(b_src, b_tgt, montos_ib):
            nuevas_filas.append(f"{step},Banco,{s},Banco,{t},{m:.2f},INTERBANCARIO")

        # 3. Propiedad: Hogar -> Empresa
        for e_idx, h_idx in enumerate(estado.duenos_empresas):
            nuevas_filas.append(f"{step},Hogar,{h_idx},Empresa,{e_idx},1.0,PROPIEDAD")

        # 4. Propiedad: Hogar -> Banco
        for b_idx, h_idx in enumerate(estado.duenos_bancos):
            nuevas_filas.append(f"{step},Hogar,{h_idx},Banco,{b_idx},1.0,PROPIEDAD")
            
        # Write to file periodically or every step (handling buffer if needed, but direct append is safer for crash recovery)
        if nuevas_filas:
            with open(self.nombre_archivo, "a") as f:
                f.write("\n".join(nuevas_filas) + "\n")

class EjecutorSimulacion:
    def __init__(self, pasos=PASOS_SIMULACION):
        self.pasos = pasos
        self.resultados = {}
        # Recolector global (o por escenario, aquí lo haremos por escenario pero sobreescribiendo para simplificar la demo)

    def ejecutar_escenario(self, nombre_escenario, modo_impuesto):
        print(f"\n>>> Iniciando Escenario: {nombre_escenario} (Modo Impuesto: {modo_impuesto})")
        tiempo_inicio = time.time()
        
        # Inicializar Estado y Recolector (filename incluye el escenario para no pisar)
        estado = EstadoEconomia()
        nombre_safe = nombre_escenario.replace(" ", "_").replace("(", "").replace(")", "")
        recolector = RecolectorDatos(f"relaciones_{nombre_safe}.csv")
        
        # Historial de métricas
        historial = {
            'paso': [],
            'produccion_total': [],
            'consumo_total': [],
            'defaults_empresas': [],
            'deuda_total': [],
            'riesgo_sistemico': [],
            'impuesto_recaudado': [],
            'patrimonio_bancos_total': [],
            'fondo_rescate': []
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
            recolector.capturar_paso(estado, t)
            
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
            historial['fondo_rescate'].append(estado.fondo_rescate)
            
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
    # sim.ejecutar_escenario("ITF (Tasa Tobin)", modo_impuesto='ITF')
    
    # 3. IRS (Impuesto Riesgo Sistémico)
    sim.ejecutar_escenario("IRS (Impuesto Riesgo Sistémico)", modo_impuesto='IRS')
    
    sim.guardar_resultados()
