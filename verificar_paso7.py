import numpy as np
from estado import EstadoEconomia
from logica import paso7_evolucion
from parametros import *

def verificar_paso7():
    print("--- Verificando Paso 7: Evolución y Aprendizaje ---")
    
    estado = EstadoEconomia()
    
    # Setup
    demanda_inicial = 10.0
    precio_inicial = 2.0
    
    estado.demanda_esperada_empresas[:] = demanda_inicial
    estado.precios_empresas[:] = precio_inicial
    
    # Caso 1: Empresa 0 - Stock Sobrante (Sobreproducción)
    # Stock = 5.0. No vendió todo.
    # Esperamos: Expectativa Demanda Baje, Precio Baje.
    estado.stock_empresas[0] = 5.0
    
    # Caso 2: Empresa 1 - Stock Agotado (Subproducción)
    # Stock = 0.0. Vendió todo.
    # Esperamos: Expectativa Demanda Suba, Precio Suba.
    estado.stock_empresas[1] = 0.0
    
    # Ejecutar Step 7
    estado = paso7_evolucion(estado)
    
    # Verificaciones
    
    # Empresa 0
    nueva_demanda_0 = estado.demanda_esperada_empresas[0]
    nuevo_precio_0 = estado.precios_empresas[0]
    print(f"\n[Empresa 0 - Stock Sobrante (5.0)]")
    print(f"Demanda: {demanda_inicial} -> {nueva_demanda_0:.2f} (Esperado < {demanda_inicial})")
    print(f"Precio:  {precio_inicial} -> {nuevo_precio_0:.2f} (Esperado < {precio_inicial})")
    
    assert nueva_demanda_0 < demanda_inicial, "La demanda debería bajar por stock sobrante"
    assert nuevo_precio_0 < precio_inicial, "El precio debería bajar por stock sobrante"
    
    # Empresa 1
    nueva_demanda_1 = estado.demanda_esperada_empresas[1]
    nuevo_precio_1 = estado.precios_empresas[1]
    print(f"\n[Empresa 1 - Stock Agotado (0.0)]")
    print(f"Demanda: {demanda_inicial} -> {nueva_demanda_1:.2f} (Esperado > {demanda_inicial})")
    print(f"Precio:  {precio_inicial} -> {nuevo_precio_1:.2f} (Esperado > {precio_inicial})")
    
    assert nueva_demanda_1 > demanda_inicial, "La demanda debería subir por escasez"
    assert nuevo_precio_1 > precio_inicial, "El precio debería subir por escasez"
    
    print("\n✅ Verificación del Paso 7 completada.")

if __name__ == "__main__":
    verificar_paso7()
