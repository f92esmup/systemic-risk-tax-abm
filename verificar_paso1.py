import numpy as np
from estado import EstadoEconomia
from logica import paso1_planificacion_empresas
from parametros import *

def verificar_paso1():
    print("--- Verificando Paso 1: Demanda de Empresas ---")
    
    # Inicializar estado
    estado = EstadoEconomia()
    # Modificamos el efectivo de algunas empresas para forzar situaciones de crédito
    estado.efectivo_empresas[0] = 0.0 # Esta empresa necesitará crédito seguro
    estado.efectivo_empresas[1] = 1e6 # Esta empresa NO debería necesitar crédito
    
    # Ejecutar Paso 1
    estado = paso1_planificacion_empresas(estado)
    
    # Verificaciones Tensoriales
    
    # 1. No debe haber demandas negativas
    assert np.all(estado.demanda_esperada_empresas >= 0), "Error: Demanda negativa detectada"
    assert np.all(estado.demanda_trabajo_empresas >= 0), "Error: Demanda laboral negativa"
    
    # 2. Relación de Producción
    # N * alpha debe ser igual a Y (con margen de error flotante)
    chequeo_produccion = estado.demanda_trabajo_empresas * PRODUCTIVIDAD_LABORAL
    assert np.allclose(chequeo_produccion, estado.produccion_planeada_empresas), "Error: Función de producción inconsistente"
    
    # 3. Lógica de Crédito
    # Empresa 0 (sin efectivo) debe pedir crédito igual a su masa salarial
    assert estado.demanda_credito_empresas[0] == estado.masa_salarial_empresas[0], \
        f"Empresa 0 falló: Brecha crédito ({estado.demanda_credito_empresas[0]}) != Masa Salarial ({estado.masa_salarial_empresas[0]})"
    
    # Empresa 1 (mucho efectivo) no debe pedir crédito
    assert estado.demanda_credito_empresas[1] == 0.0, \
        f"Empresa 1 falló: Pidió crédito ({estado.demanda_credito_empresas[1]}) teniendo efectivo de sobra"
        
    print("✅ Verificación del Paso 1 completada con éxito.")
    print(f"Ejemplo - Empresa 0: Demanda {estado.demanda_esperada_empresas[0]:.2f}, Crédito: {estado.demanda_credito_empresas[0]:.2f}")
    print(f"Total Crédito Solicitado en el sistema: {np.sum(estado.demanda_credito_empresas):.2f}")

if __name__ == "__main__":
    verificar_paso1()
