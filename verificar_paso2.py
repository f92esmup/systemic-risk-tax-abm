import numpy as np
from estado import EstadoEconomia
from logica import paso1_planificacion_empresas, paso2_prestamos_bancarios
from parametros import *

def verificar_paso2():
    print("--- Verificando Paso 2: Mercado de Crédito ---")
    
    # Inicializar estado y forzar necesidades
    estado = EstadoEconomia()
    
    # Configurar: 
    # Forzamos demanda
    estado.efectivo_empresas[:] = 0.0 # Todas necesitan crédito
    estado.efectivo_bancos[:] = 1e9 # Bancos infinitamente ricos para evitar credit crunch por ahora
    
    # Ejecutar pasos
    estado = paso1_planificacion_empresas(estado) # Genera demandas
    
    # Pre-check
    demanda_total = np.sum(estado.demanda_credito_empresas)
    print(f"Demanda Total Inicial: {demanda_total:.2f}")
    assert demanda_total > 0, "No hay demanda de crédito generada"
    
    # Ejecutar Paso 2
    estado = paso2_prestamos_bancarios(estado)
    
    # Verificaciones
    
    # 1. Tasa de Asignación 
    # Dado que los bancos tienen efectivo infinito, todas las empresas con tasa <= t_max deberían haber recibido crédito.
    otorgado = np.sum(estado.nuevos_prestamos_otorgados)
    print(f"Crédito Total Otorgado: {otorgado:.2f}")
    
    if otorgado < demanda_total:
         print("(Nota: Puede ser menor debido a la tasa máxima de interés)")

    assert otorgado > 0, "No se otorgaron préstamos a pesar de tener liquidez infinita"
    
    # 2. Verificación de Selección Racional
    for f in range(5): # Check primeras 5 empresas
        if estado.nuevos_prestamos_otorgados[f] > 0:
            banco = estado.eleccion_prestamista_empresa[f]
            tasa = estado.tasas_interes_prestamos[banco, f]
            print(f"Empresa {f} eligió Banco {banco} con tasa {tasa:.4f}")
            assert tasa >= TASA_REFINANCIACION, "La tasa no puede ser menor a la de refinanciación"
            
    # 3. Consistencia de Balances
    # El efectivo TOTAL de los bancos debió bajar
    efectivo_total_bancos_inicial = 1e9 * N_BANCOS
    assert np.sum(estado.efectivo_bancos) < efectivo_total_bancos_inicial, "El efectivo TOTAL de los bancos no disminuyó"
    
    # Conservación de flujos: Lo que bajó en bancos debió subir en empresas
    bajada_efectivo_bancos = efectivo_total_bancos_inicial - np.sum(estado.efectivo_bancos)
    subida_efectivo_empresas = np.sum(estado.efectivo_empresas) # Iniciaron en 0
    assert np.isclose(bajada_efectivo_bancos, subida_efectivo_empresas), \
        f"Fallo de conservación: Bancos bajaron {bajada_efectivo_bancos}, Empresas subieron {subida_efectivo_empresas}"
    
    print("✅ Verificación del Paso 2 completada.")

if __name__ == "__main__":
    verificar_paso2()
