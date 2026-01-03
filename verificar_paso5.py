import numpy as np
from estado import EstadoEconomia
from logica import paso5_repago_empresas
from parametros import *

def verificar_paso5():
    print("--- Verificando Paso 5: Repago y Quiebras ---")
    
    estado = EstadoEconomia()
    
    # --- Configuración del Escenario ---
    
    # 1. Empresa Solvente (Empresa 0)
    # Debe 100, Tasa 5%. Amortización 5% (tau=0.05).
    # Debido = 100*0.05 + 100*0.05 = 5 + 5 = 10.
    # Efectivo = 20 (Suficiente)
    estado.prestamos_banco_empresa[0, 0] = 100.0
    estado.tasas_interes_prestamos[0, 0] = 0.05
    estado.efectivo_empresas[0] = 20.0
    
    # 2. Empresa Insolvente / Default (Empresa 1)
    # Debe 100. Debido = 10.
    # Efectivo = 5 (Insuficiente)
    # Resultado esperado: Paga 5. Banco pierde (100 - 5) = 95. Empresa reseteada.
    estado.prestamos_banco_empresa[0, 1] = 100.0
    estado.tasas_interes_prestamos[0, 1] = 0.05
    estado.efectivo_empresas[1] = 5.0
    
    # Estado inicial de bancos
    patrimonio_inicial_banco = estado.patrimonio_bancos[0]
    efectivo_inicial_banco = estado.efectivo_bancos[0]
    
    # Ejecutar Paso 5
    estado = paso5_repago_empresas(estado)
    
    # --- Verificaciones ---
    
    # 1. Caso Solvente (Empresa 0)
    print("\n--- Analizando Empresa 0 (Solvente) ---")
    # Deuda nueva = 100 - 5 (amortizado) = 95
    prestamo_0 = estado.prestamos_banco_empresa[0, 0]
    efectivo_0 = estado.efectivo_empresas[0]
    print(f"Deuda restante E0: {prestamo_0:.2f} (Esperado 95.00)")
    print(f"Efectivo restante E0: {efectivo_0:.2f} (Esperado 10.00)")
    
    assert np.isclose(prestamo_0, 95.0), f"Deuda E0 incorrecta: {prestamo_0}"
    assert np.isclose(efectivo_0, 10.0), f"Efectivo E0 incorrecto: {efectivo_0}"
    
    # 2. Caso Insolvente (Empresa 1)
    print("\n--- Analizando Empresa 1 (Default) ---")
    # Empresa debe ser reseteada
    prestamo_1 = estado.prestamos_banco_empresa[0, 1]
    efectivo_1 = estado.efectivo_empresas[1]
    defaults_acum = estado.defaults_acumulados_empresas[1]
    
    print(f"Deuda restante E1: {prestamo_1:.2f} (Esperado 0.00 - Reset)")
    print(f"Efectivo actual E1: {efectivo_1:.2f} (Esperado {EFECTIVO_INICIAL_EMPRESA} - Reset)")
    print(f"Defaults acumulados E1: {defaults_acum} (Esperado 1)")
    
    assert prestamo_1 == 0.0, "La deuda de la empresa quebrada no se borró"
    assert efectivo_1 == EFECTIVO_INICIAL_EMPRESA, "La empresa quebrada no se re-capitalizó"
    assert defaults_acum == 1, "No se contó el default"
    
    # 3. Balance del Banco 0
    print("\n--- Analizando Banco 0 ---")
    
    efectivo_final_banco = estado.efectivo_bancos[0]
    patrimonio_final_banco = estado.patrimonio_bancos[0]
    
    delta_efectivo = efectivo_final_banco - efectivo_inicial_banco
    delta_patrimonio = patrimonio_final_banco - patrimonio_inicial_banco
    
    print(f"Cambio Efectivo Banco: {delta_efectivo:.2f} (Esperado +15.00)")
    print(f"Cambio Patrimonio Banco: {delta_patrimonio:.2f} (Esperado -90.00)")
    print(f"Deuda Incobrable registrada: {estado.deuda_incobrable_bancos[0]:.2f} (Esperado 95.00)")
    
    assert np.isclose(delta_efectivo, 15.0), f"Flujo de caja bancario erróneo: {delta_efectivo}"
    assert np.isclose(delta_patrimonio, -90.0), f"Cambio patrimonial erróneo: {delta_patrimonio}"
    assert np.isclose(estado.deuda_incobrable_bancos[0], 95.0), "Registro de deuda incobrable incorrecto"
    
    print("\n✅ Verificación del Paso 5 completada con éxito.")

if __name__ == "__main__":
    verificar_paso5()
