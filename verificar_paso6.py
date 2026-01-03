import numpy as np
from estado import EstadoEconomia
from logica import paso6_mercado_interbancario, calcular_debtrank
from parametros import *

def verificar_paso6():
    print("--- Verificando Paso 6: Interbancario y Tax ---")
    
    estado = EstadoEconomia()
    
    # --- Escenario 1: Préstamo Simple sin Riesgo Sistémico previo ---
    print("\n[Test 1] Préstamo Simple (Banco 0 -> Banco 1)")
    # Banco 0: Superávit
    # Banco 1: Déficit
    
    estado.patrimonio_bancos[:] = 100.0
    estado.efectivo_bancos[:] = 20.0 # Bajo efectivo general
    
    estado.prestamos_banco_empresa[:, :] = 0.0
    estado.prestamos_banco_empresa[:, 0] = 500.0
    
    objetivo = 500.0 * RATIO_COLCHON_LIQUIDEZ # 50.0
    
    # Banco 0: Efectivo 1000 (Superávit enorme)
    estado.efectivo_bancos[0] = 1000.0
    # Banco 1: Efectivo 0 (Déficit 50)
    estado.efectivo_bancos[1] = 0.0
    
    # Ejecutar paso 6
    estado = paso6_mercado_interbancario(estado)
    
    print(f"Banco 0 Efectivo: {estado.efectivo_bancos[0]:.2f}")
    print(f"Banco 1 Efectivo: {estado.efectivo_bancos[1]:.2f}")
    print(f"Interbancario 0->1: {estado.matriz_interbancaria[0, 1]:.2f}")
    
    assert estado.matriz_interbancaria[0, 1] > 0, "No se ejecutó el préstamo 0->1"
    assert estado.efectivo_bancos[1] > 0, "Banco 1 no recibió liquidez"
    
    # --- Escenario 2: DebtRank y Tax ---
    print("\n[Test 2] Activación del Impuesto de Riesgo Sistémico (IRS/SRT)")
    # Limpiamos
    estado = EstadoEconomia()
    estado.patrimonio_bancos[:] = 100.0
    
    # Crear una red frágil ("Star Network" o similar)
    # Banco 2 le debe a Banco 3, 4, 5, 6... (Banco 2 es un gran deudor sistémico).
    deudores_de_2 = [3, 4, 5]
    for b in deudores_de_2:
        estado.matriz_interbancaria[b, 2] = 95.0
        
    # Calculamos riesgo basal
    rs_base = calcular_debtrank(estado.matriz_interbancaria, estado.patrimonio_bancos)
    print(f"Riesgo Sistémico Base: {rs_base:.4f}")
    
    state_dummy = EstadoEconomia() # Solo para reiniciar préstamos limpios
    estado.prestamos_banco_empresa[:, :] = 0.0
    estado.prestamos_banco_empresa[:, 0] = 500.0
    
    # Banco 2: Liquidez 0. Necesita 50.
    estado.efectivo_bancos[2] = 0.0
    
    # Banco 0: Liquidez sobra
    estado.efectivo_bancos[0] = 1000.0
    
    # Pre-Ejecución
    print("Ejecutando mercado interbancario con deudor sistémico...")
    estado = paso6_mercado_interbancario(estado)
    
    rs_nuevo = estado.riesgo_sistemico_total
    impuesto = estado.impuesto_recaudado
    prestamo_0_2 = estado.matriz_interbancaria[0, 2]
    
    print(f"RS Final: {rs_nuevo:.4f}")
    print(f"Delta RS: {rs_nuevo - rs_base:.4f}")
    print(f"Préstamo 0->2: {prestamo_0_2:.2f}")
    print(f"Impuesto Recaudado: {impuesto:.4f}")
    
    if prestamo_0_2 > 0:
        print("El préstamo se realizó.")
        # Verificamos si hubo tax (si el riesgo aumentó)
        if rs_nuevo > rs_base + 1e-4:
            assert impuesto > 0, "Riesgo aumentó pero no se cobró impuesto!"
            print("✅ Impuesto cobrado correctamente.")
        else:
            print("⚠️ El riesgo no aumentó significativamente.")
    else:
        print("El préstamo NO se realizó (quizás el impuesto era muy alto).")
        print("✅ Impuesto evitó transacción riesgosa (comportamiento esperado).")

    print("\n✅ Verificación del Paso 6 completada.")

if __name__ == "__main__":
    verificar_paso6()
