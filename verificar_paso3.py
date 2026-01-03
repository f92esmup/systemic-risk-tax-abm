import numpy as np
from estado import EstadoEconomia
from logica import paso1_planificacion_empresas, paso2_prestamos_bancarios, paso3_produccion
from parametros import *

def verificar_paso3():
    print("--- Verificando Paso 3: Producción y Labor ---")
    
    estado = EstadoEconomia()
    
    # Configuración para provocar escasez masiva de trabajadores
    # Efectivo suficiente para que la restricción sea N_HOGARES, no dinero.
    estado.efectivo_empresas[:] = 1e6 
    estado.efectivo_bancos[:] = 1e9
    
    # Pre-pasos
    estado = paso1_planificacion_empresas(estado)
    estado = paso2_prestamos_bancarios(estado)
    
    # Capturar estado previo
    efectivo_inicial_empresas = np.copy(estado.efectivo_empresas)
    efectivo_inicial_hogares = np.copy(estado.efectivo_hogares)
    stock_inicial = np.copy(estado.stock_empresas)
    
    # Ejecutar Paso 3
    estado = paso3_produccion(estado)
    
    # Verificaciones
    
    # 1. Restricción de Población
    # El número total de empleados no puede superar N_HOGARES
    total_empleados = np.sum(estado.empleador_hogares != -1)
    print(f"Total Empleados: {total_empleados} / {N_HOGARES}")
    assert total_empleados <= N_HOGARES, "Se contrató más gente de la que existe"
    
    # Debería estar cerca del 100% de ocupación dadas las demandas altas
    assert total_empleados > N_HOGARES * 0.95, "El desempleo es sospechosamente alto dado el exceso de demanda"
    
    # 2. Conservación de Dinero (Salarios)
    # Lo que pagaron las empresas == Lo que recibieron los hogares
    delta_efectivo_empresas = np.sum(efectivo_inicial_empresas) - np.sum(estado.efectivo_empresas)
    delta_efectivo_hogares = np.sum(estado.efectivo_hogares) - np.sum(efectivo_inicial_hogares)
    
    print(f"Empresas pagaron: {delta_efectivo_empresas:.2f}")
    print(f"Hogares recibieron: {delta_efectivo_hogares:.2f}")
    
    assert np.isclose(delta_efectivo_empresas, delta_efectivo_hogares), "El dinero de salarios se perdió o creó en el camino"
    assert delta_efectivo_empresas > 0, "No se pagaron salarios"
    
    # 3. Producción Consistente
    # Aumento de stock = Total Empleados * alpha
    aumento_stock = np.sum(estado.stock_empresas) - np.sum(stock_inicial)
    produccion_esperada = total_empleados * PRODUCTIVIDAD_LABORAL
    
    print(f"Producción Real: {aumento_stock:.2f} (Esperada: {produccion_esperada:.2f})")
    assert np.isclose(aumento_stock, produccion_esperada), "La producción no coincide con la mano de obra contratada"
    
    print("✅ Verificación del Paso 3 completada.")

if __name__ == "__main__":
    verificar_paso3()
