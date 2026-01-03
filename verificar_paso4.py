import numpy as np
from estado import EstadoEconomia
from logica import paso4_consumo
from parametros import *

def verificar_paso4():
    print("--- Verificando Paso 4: Consumo ---")
    
    estado = EstadoEconomia()
    
    # Setup: Hogares con dinero, Empresas con stock y precios variados
    estado.efectivo_hogares[:] = 100.0 # Presupuesto = 100 * 0.8 = 80
    estado.stock_empresas[:] = 1000.0    # Stock abundante por defecto
    estado.efectivo_empresas[:] = 0.0
    
    # Precios heterogéneos para probar selección
    estado.precios_empresas[:] = 2.0
    estado.precios_empresas[0] = 1.0 # Empresa 0 barata -> Debería atraer más demanda
    
    # Caso especial: Empresa 1 sin stock para probar racionamiento
    estado.stock_empresas[1] = 0.0
    estado.precios_empresas[1] = 0.5 # Muy barata pero sin stock
    
    # Ejecutar Paso 4
    suma_inicial_efectivo_hogares = np.sum(estado.efectivo_hogares)
    suma_inicial_stock_empresas = np.sum(estado.stock_empresas)
    
    estado = paso4_consumo(estado)
    
    # Verificaciones
    
    # 1. Conservación de Dinero
    # Dinero gastado = Dinero recibido
    suma_final_efectivo_hogares = np.sum(estado.efectivo_hogares)
    ingresos_totales = np.sum(estado.efectivo_empresas)
    gasto = suma_inicial_efectivo_hogares - suma_final_efectivo_hogares
    
    print(f"Gasto Hogares: {gasto:.2f}")
    print(f"Ingreso Empresas: {ingresos_totales:.2f}")
    assert np.isclose(gasto, ingresos_totales), "Dinero desaparecido en la transacción"
    
    # 2. Selección de Precios (Empresa 0 vs Resto)
    print(f"Ingreso Empresa 0 (Barata, Stock OK): {estado.efectivo_empresas[0]:.2f}")
    print(f"Ingreso Empresa 1 (Muy Barata, Sin Stock): {estado.efectivo_empresas[1]:.2f}")
    
    assert estado.efectivo_empresas[1] < 1e-5, "Empresa 1 vendió productos sin tener stock!"
    assert estado.stock_empresas[1] >= 0, "Stock negativo en empresa 1"
    
    # 3. Reducción de Stock General
    # Stock final < Stock Inicial
    suma_final_stock_empresas = np.sum(estado.stock_empresas)
    delta_stock = suma_inicial_stock_empresas - suma_final_stock_empresas
    print(f"Stock Vendido Total: {delta_stock:.2f}")
    assert delta_stock > 0, "No se vendió nada"
    
    print("✅ Verificación del Paso 4 completada.")

if __name__ == "__main__":
    verificar_paso4()
