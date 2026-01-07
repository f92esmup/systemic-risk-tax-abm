import numpy as np
from parametros import Param as p

def paso6_repago_deuda(
    firm_liquidez,
    matriz_prestamos_firmas,     # (F, B) Deuda Total (Principal + Intereses)
    matriz_intereses_firmas,     # (F, B) Stock de Intereses
    bancos_liquidez,
    matriz_interbancaria,        # (B, B) Deuda Interbancaria Total
    matriz_intereses_ib,         # (B, B) Stock Intereses Interbancarios
    tau
):
    """
    Ejecuta el Paso 6: Amortización de Deuda (Repayment).
    
    Lógica Exacta:
    1. Empresas pagan fracción 'tau' de su deuda a Bancos.
    2. Bancos pagan fracción 'tau' de su deuda interbancaria a otros Bancos.
    3. Si la liquidez es insuficiente, pagan todo lo que tienen (Partial Payment).
    4. Se actualizan los stocks de deuda e intereses proporcionalmente.
    """
    
    # --- A. REPAGO EMPRESAS -> BANCOS ---
    
    # 1. Calcular Monto a Pagar (Target)
    # Target = Deuda_Total * tau
    pago_objetivo_firmas = matriz_prestamos_firmas * tau
    
    # 2. Verificar Restricción de Liquidez por Empresa
    total_a_pagar_por_firma = np.sum(pago_objetivo_firmas, axis=1) # Vector (F,)
    
    # Ratio de cumplimiento (1.0 si puede pagar, <1.0 si no tiene caja)
    # Evitamos división por cero
    ratio_pago_firmas = np.divide(
        firm_liquidez, 
        total_a_pagar_por_firma, 
        out=np.ones_like(firm_liquidez), 
        where=total_a_pagar_por_firma > 1e-9
    )
    # No pueden pagar más del 100% de lo que deben, ni más de lo que tienen (ratio <= 1.0)
    ratio_pago_firmas = np.minimum(1.0, ratio_pago_firmas)
    
    # 3. Calcular Pago Real (Ajustado por liquidez)
    # Broadcasting: (F, B) * (F, 1)
    pago_real_firmas_matriz = pago_objetivo_firmas * ratio_pago_firmas[:, np.newaxis]
    
    # 4. Ejecutar Transferencias
    # Sale de Firmas
    total_pagado_por_firma = np.sum(pago_real_firmas_matriz, axis=1)
    firm_liquidez -= total_pagado_por_firma
    
    # Entra a Bancos
    total_recibido_por_banco = np.sum(pago_real_firmas_matriz, axis=0)
    bancos_liquidez += total_recibido_por_banco
    
    # 5. Actualizar Stocks de Deuda (Reducción)
    matriz_prestamos_firmas -= pago_real_firmas_matriz
    
    # Actualizar Stock de Intereses (Reducción Proporcional)
    # Si pagué el X% de la deuda total, reduzco el X% de los intereses asociados.
    # Payment_Fraction_Effective = Pago_Real / Deuda_Total
    # Pero simplificando: Reducción = Intereses * (Pago_Real / Deuda_Total)
    # O más simple: Si pagué 'tau * ratio', reduzco 'tau * ratio' del interés.
    reduccion_intereses = matriz_intereses_firmas * tau * ratio_pago_firmas[:, np.newaxis]
    matriz_intereses_firmas -= reduccion_intereses
    
    # Limpieza numérica (evitar -0.000001)
    matriz_prestamos_firmas = np.maximum(matriz_prestamos_firmas, 0)
    matriz_intereses_firmas = np.maximum(matriz_intereses_firmas, 0)


    # --- B. REPAGO INTERBANCARIO (BANCOS -> BANCOS) ---
    
    # 1. Calcular Monto a Pagar
    pago_objetivo_ib = matriz_interbancaria * tau
    
    # 2. Restricción Liquidez Banco Deudor (Filas)
    total_a_pagar_por_banco = np.sum(pago_objetivo_ib, axis=1) # Vector (B,)
    
    ratio_pago_bancos = np.divide(
        bancos_liquidez, 
        total_a_pagar_por_banco, 
        out=np.ones_like(bancos_liquidez), 
        where=total_a_pagar_por_banco > 1e-9
    )
    ratio_pago_bancos = np.minimum(1.0, ratio_pago_bancos)
    
    # 3. Pago Real
    pago_real_ib_matriz = pago_objetivo_ib * ratio_pago_bancos[:, np.newaxis]
    
    # 4. Transferencias
    # Sale del Deudor
    total_pagado_ib = np.sum(pago_real_ib_matriz, axis=1)
    bancos_liquidez -= total_pagado_ib
    
    # Entra al Acreedor (Columnas)
    total_recibido_ib = np.sum(pago_real_ib_matriz, axis=0)
    bancos_liquidez += total_recibido_ib
    
    # 5. Actualizar Stocks IB
    matriz_interbancaria -= pago_real_ib_matriz
    
    reduccion_intereses_ib = matriz_intereses_ib * tau * ratio_pago_bancos[:, np.newaxis]
    matriz_intereses_ib -= reduccion_intereses_ib
    
    matriz_interbancaria = np.maximum(matriz_interbancaria, 0)
    matriz_intereses_ib = np.maximum(matriz_intereses_ib, 0)
    
    return (
        firm_liquidez,
        bancos_liquidez,
        matriz_prestamos_firmas,
        matriz_intereses_firmas,
        matriz_interbancaria,
        matriz_intereses_ib,
        np.sum(pago_real_firmas_matriz), # Stats Total Repagado Firmas
        np.sum(pago_real_ib_matriz)      # Stats Total Repagado IB
    )
