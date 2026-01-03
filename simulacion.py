
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

        # 1. Interbancario: Banco -> Banco (Weight=Monto Deuda) - PRIORIDAD 1 (Fig 2)
        b_src, b_tgt = np.where(estado.matriz_interbancaria > 1e-2)
        montos_ib = estado.matriz_interbancaria[b_src, b_tgt]
        for s, t, m in zip(b_src, b_tgt, montos_ib):
            nuevas_filas.append(f"{step},Banco,{s},Banco,{t},{m:.2f},INTERBANCARIO")

        # 2. Crédito: Banco -> Empresa (Weight=Monto Deuda) - PRIORIDAD 2 (Opcional si solo queremos Fig 2)
        # La mantenemos pero comentada si se requiere optimización extrema, el usuario dijo "fidelidad al paper",
        # la Fig 2 es RED INTERBANCARIA. Dejaremos Créditos para verificación macro pero no red.
        # banco_idxs, empresa_idxs = np.where(estado.prestamos_banco_empresa > 1e-2)
        # montos = estado.prestamos_banco_empresa[banco_idxs, empresa_idxs]
        # for b, e, m in zip(banco_idxs, empresa_idxs, montos):
        #     nuevas_filas.append(f"{step},Banco,{b},Empresa,{e},{m:.2f},CREDITO")
            
        # Write to file periodically or every step
        if nuevas_filas:
            with open(self.nombre_archivo, "a") as f:
                f.write("\n".join(nuevas_filas) + "\n")

class EjecutorSimulacion:
    def __init__(self, pasos=PASOS_SIMULACION):
        self.pasos = pasos
        self.resultados = {}

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
            'fondo_rescate': [],
            'tamano_cascada': [] # Nuevo: Fig 4b
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
            
            # --- Paso 5b: Contagio Interbancario (Shock Externo Previo a Mercado) ---
            estado, cascada_1 = ejecutar_contagio_interbancario(estado)
            
            # --- Paso 6: Interbancario y Tax ---
            # Ahora paso6 devuelve (estado, cascada_interna)
            estado, cascada_2 = paso6_mercado_interbancario(estado, modo_impuesto=modo_impuesto)
            
            # --- catch-up final if needed (paso6 already did it, but let's be safe or skip)
            # Como paso6 ya ejecutó contagio, no deberíamos necesitar llamarlo de nuevo inmediatamente
            # salvo que queramos ver si el tax causó algo extra (pero paso6 lo incluye).
            # Eliminamos la llamada redundante para evitar doble conteo o confusión.
            
            cascada_total_paso = cascada_1 + cascada_2
            
            # --- Paso 7: Evolución ---
            estado = paso7_evolucion(estado)
            
            # --- Recolección de Datos ---
            if t % 5 == 0: # Optimización: Guardar red cada 5 pasos
                recolector.capturar_paso(estado, t)
            
            historial['paso'].append(t)
            historial['produccion_total'].append(np.sum(estado.stock_empresas))
            
            ventas = np.sum(estado.ventas_diarias_empresas) if hasattr(estado, 'ventas_diarias_empresas') else 0.0
            historial['consumo_total'].append(ventas)
            
            historial['defaults_empresas'].append(np.sum(estado.defaults_acumulados_empresas))
            historial['deuda_total'].append(np.sum(estado.prestamos_banco_empresa))
            historial['riesgo_sistemico'].append(estado.riesgo_sistemico_total)
            historial['impuesto_recaudado'].append(estado.impuesto_recaudado)
            historial['patrimonio_bancos_total'].append(np.sum(estado.patrimonio_bancos))
            historial['fondo_rescate'].append(estado.fondo_rescate)
            historial['tamano_cascada'].append(cascada_total_paso)
            
            if t % 50 == 0:
                print(f"   Paso {t}/{self.pasos} - SR: {estado.riesgo_sistemico_total:.2f} - Cascada: {cascada_total_paso}")

        tiempo_transcurrido = time.time() - tiempo_inicio
        print(f"<<< Escenario {nombre_escenario} completado en {tiempo_transcurrido:.2f}s")
        
        # Guardar Snapshot Final de Riesgos Individuales (Fig 3b)
        # Calculamos DebtRank individual final
        # Necesitamos la funcion calcular_debtrank pero devuelve scalar.
        # Vamos a guardar el Out-Degree Ponderado como proxy o implementar DR Vectorial.
        # Por simplicidad y robustez: Guardamos Exposiciones Totales (Out-Strength) y Patrimonio.
        from logica import calcular_debtrank
        # Nota: calcular_debtrank en logica devuelve total, pero dentro calcula 'perdidas' vector. 
        # No podemos acceder facilmente, usamos proxy de NetworkX en visualizacion o guardamos raw data.
        # Mejor: Guardamos estado.matriz_interbancaria y estado.patrimonio_bancos en numpy.
        
        np.savez(f"snapshot_final_{nombre_safe}.npz", 
                 matriz_interbancaria=estado.matriz_interbancaria, 
                 patrimonio_bancos=estado.patrimonio_bancos)

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
