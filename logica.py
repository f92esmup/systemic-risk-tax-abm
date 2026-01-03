import numpy as np
from parametros import *

def paso1_planificacion_empresas(estado):
    """
    Paso 1: Las empresas definen su demanda de trabajo y capital (crédito).
    
    Lógica Tensorial:
    1. Estimación de Demanda (D_i).
    2. Cálculo de Producción Planeada (Y_i).
    3. Cálculo de Trabajo Requerido (N_i).
    4. Cálculo de Masa Salarial (W_i).
    5. Cálculo de Necesidad de Crédito.
    """
    
    # 1. Estimación de Demanda
    # Inicialización: Demanda aleatoria uniforme.
    estado.demanda_esperada_empresas = np.random.uniform(5.0, 15.0, size=N_EMPRESAS)
    
    # 2. Producción Planeada 
    estado.produccion_planeada_empresas = estado.demanda_esperada_empresas.copy()
    
    # 3. Demanda de Trabajo
    # Y = alpha * N  =>  N = Y / alpha
    estado.demanda_trabajo_empresas = estado.produccion_planeada_empresas / PRODUCTIVIDAD_LABORAL
    
    # 4. Costo Salarial (Masa Salarial)
    estado.masa_salarial_empresas = estado.demanda_trabajo_empresas * TASA_SALARIAL
    
    # 5. Demanda de Crédito
    gap_liquidez = estado.masa_salarial_empresas - estado.efectivo_empresas
    estado.demanda_credito_empresas = np.maximum(0.0, gap_liquidez)
    
    return estado

def paso2_prestamos_bancarios(estado):
    """
    Paso 2: Mercado de Crédito y Asignación de Liquidez.
    """
    
    # Identificar empresas que necesitan crédito
    indices_empresas_necesitadas = np.where(estado.demanda_credito_empresas > 1e-5)[0]
    if len(indices_empresas_necesitadas) == 0:
        return estado
        
    # --- 1. Selección de Bancos Candidatos ---
    mascara_cotizaciones = np.zeros((N_EMPRESAS, N_BANCOS), dtype=bool)
    
    matriz_aleatoria = np.random.rand(len(indices_empresas_necesitadas), N_BANCOS)
    top_n_bancos = np.argsort(matriz_aleatoria, axis=1)[:, :N_SOLICITUDES_CREDITO]
    
    indices_fila = indices_empresas_necesitadas[:, np.newaxis]
    mascara_cotizaciones[indices_fila, top_n_bancos] = True
    
    # --- 2. Cálculo de Tasas de Interés ---
    deuda_empresa = np.sum(estado.prestamos_banco_empresa, axis=0) # (F,)
    
    recursos_empresa = estado.efectivo_empresas + valor_stock_empresa(estado)
    apalancamiento_empresa = np.divide(deuda_empresa, np.maximum(recursos_empresa, 1.0))
    
    especificidad_banco = np.random.uniform(0, 1, size=N_BANCOS)
    
    factor_riesgo_empresa = np.tanh(apalancamiento_empresa)
    
    matriz_riesgo = np.outer(especificidad_banco, factor_riesgo_empresa)
    tasas_ofrecidas = TASA_REFINANCIACION * (1.0 + matriz_riesgo)
    
    tasas_finales = np.where(mascara_cotizaciones.T, tasas_ofrecidas, np.inf)
    
    # --- 3. Selección de Mejor Oferta ---
    indices_mejor_banco = np.argmin(tasas_finales, axis=0) # (F,)
    tasas_mejor_banco = np.min(tasas_finales, axis=0)      # (F,)
    
    # --- 4. Asignación y Chequeo de Liquidez ---
    for f_idx in indices_empresas_necesitadas:
        banco_elegido = indices_mejor_banco[f_idx]
        tasa = tasas_mejor_banco[f_idx]
        
        demanda = estado.demanda_credito_empresas[f_idx]
        if tasa > TASA_INTERES_MAXIMA:
            demanda *= CONTRACCION_DEMANDA_CREDITO
            
        if estado.efectivo_bancos[banco_elegido] >= demanda:
            estado.efectivo_bancos[banco_elegido] -= demanda
            estado.efectivo_empresas[f_idx] += demanda
            estado.prestamos_banco_empresa[banco_elegido, f_idx] += demanda
            estado.tasas_interes_prestamos[banco_elegido, f_idx] = tasa
            
            estado.nuevos_prestamos_otorgados[f_idx] = demanda
            estado.eleccion_prestamista_empresa[f_idx] = banco_elegido
        else:
            # Credit Crunch
            pass

    return estado

def valor_stock_empresa(estado):
    return estado.stock_empresas * estado.precios_empresas

def ejecutar_mercado_interbancario_demanda(estado, deficit_liquidez, modo_impuesto='IRS'):
    """
    Ejecuta el mercado interbancario basado en la demanda de liquidez inmediata (déficit).
    Objetivo: Cubrir 'deficit_liquidez' para poder prestar a empresas.
    """
    indices_deficitarios = np.where(deficit_liquidez > 0)[0]
    if len(indices_deficitarios) == 0:
        return estado

    # Identificar Superavitarios (Bancos con efectivo disponible)
    # Nota: Usamos todo el efectivo disponible porque aun no se ha prestado a empresas.
    indices_superavitarios = np.where(estado.efectivo_bancos > 1e-4)[0]
    
    if len(indices_superavitarios) == 0:
        return estado # Nadie tiene dinero
        
    # Mezclar orden para evitar sesgos
    np.random.shuffle(indices_deficitarios)
    
    # Calcular DebtRank actual para comparaciones de impuestos
    rs_actual = calcular_debtrank(estado.matriz_interbancaria, estado.patrimonio_bancos)
    estado.riesgo_sistemico_total = rs_actual
    total_patrimonio_sistema = max(np.sum(estado.patrimonio_bancos), 1.0)
    
    N_COTIZACIONES = 5
    impuestos_recaudados = 0.0
    
    for idx_prestatario in indices_deficitarios:
        monto_necesario = deficit_liquidez[idx_prestatario]
        if monto_necesario < 1e-4: continue
        
        # Filtrar prestamistas con dinero
        prestamistas_validos = [p for p in indices_superavitarios if estado.efectivo_bancos[p] > 1e-4 and p != idx_prestatario]
        if not prestamistas_validos: break
        
        # Seleccionar candidatos (Cotizaciones)
        candidatos = np.random.choice(prestamistas_validos, min(len(prestamistas_validos), N_COTIZACIONES), replace=False)
        
        mejor_prestamista = -1
        mejor_tasa_efectiva = float('inf')
        mejor_dat = None # (monto, impuesto)
        
        # Calcular Probabilidad de Default del Prestatario (Aprox. por apalancamiento/fragilidad)
        # Leverage = Pasivos/Patrimonio? O Activos/Patrimonio? 
        # Usaremos Activos Totales / Patrimonio como proxy de riesgo
        activos = np.sum(estado.prestamos_banco_empresa[idx_prestatario]) + np.sum(estado.matriz_interbancaria[idx_prestatario])
        patrimonio = max(estado.patrimonio_bancos[idx_prestatario], 0.1)
        leverage = activos / patrimonio
        # Factor de escala 0.01 según Apéndice A.3
        prob_default_prestatario = 0.01 * np.tanh(leverage) 
        
        for idx_prestamista in candidatos:
            disponible = estado.efectivo_bancos[idx_prestamista]
            monto = min(monto_necesario, disponible)
            
            # --- CÁLCULO DE IMPUESTO ---
            impuesto = 0.0
            if modo_impuesto == 'IRS':
                # Simular impacto en DebtRank
                matriz_sim = estado.matriz_interbancaria.copy()
                matriz_sim[idx_prestamista, idx_prestatario] += monto
                nuevo_rs = calcular_debtrank(matriz_sim, estado.patrimonio_bancos)
                
                delta_rs = max(0.0, nuevo_rs - rs_actual)
                delta_norm = delta_rs / total_patrimonio_sistema
                
                # Formula Paper: Tax = Monto * Sensibilidad * Delta_Impacto * Prob_Default
                impuesto = monto * SENSIBILIDAD_IRS * delta_norm * prob_default_prestatario
                
            elif modo_impuesto == 'ITF':
                impuesto = monto * TASA_ITF
            
            # Costo para el prestatario
            costo_interes = monto * TASA_INTERBANCARIA
            costo_total = costo_interes + impuesto
            tasa_efectiva = costo_total / monto if monto > 0 else float('inf')
            
            if tasa_efectiva < mejor_tasa_efectiva:
                mejor_tasa_efectiva = tasa_efectiva
                mejor_prestamista = idx_prestamista
                mejor_dat = (monto, impuesto)
        
        # Ejecutar Transacción si es razonable
        if mejor_prestamista != -1 and mejor_tasa_efectiva < 0.50: # Cap alto pero razonable
            p_lender = mejor_prestamista
            monto, impuesto = mejor_dat
            
            # Verificar si prestatario puede pagar el impuesto upfront?
            # En realidad, el impuesto suele pagarlo el que inicia o ambos. 
            # Asumimos que sale del efectivo del prestatario (que recibe el prestamo).
            # Pero si no tiene cash (por eso pide), el impuesto reduce el neto recibido?
            # O el prestamista paga y lo cobra?
            # Paper: "The tax is levied on the transaction".
            # Simplificación: El prestatario paga el impuesto con parte del dinero recibido.
            
            estado.matriz_interbancaria[p_lender, idx_prestatario] += monto
            estado.efectivo_bancos[p_lender] -= monto
            
            # El prestatario recibe 'monto'
            estado.efectivo_bancos[idx_prestatario] += monto
            
            # Pagar impuesto
            if impuesto > 0:
                if estado.efectivo_bancos[idx_prestatario] >= impuesto:
                    estado.efectivo_bancos[idx_prestatario] -= impuesto
                    impuestos_recaudados += impuesto
                    estado.fondo_rescate += impuesto
                else:
                    # Si el impuesto consume todo el préstamo, revertimos (no sirve pedir)
                    # O cobramos lo que se pueda. Revertir es mas seguro para la logica.
                    estado.matriz_interbancaria[p_lender, idx_prestatario] -= monto
                    estado.efectivo_bancos[p_lender] += monto
                    estado.efectivo_bancos[idx_prestatario] -= monto
                    continue
            
            # Actualizar deficit restante
            deficit_liquidez[idx_prestatario] -= monto
            
            # Actualizar RS global
            rs_actual = calcular_debtrank(estado.matriz_interbancaria, estado.patrimonio_bancos)

    estado.impuesto_recaudado += impuestos_recaudados # Acumular si hay multiples rondas
    estado.riesgo_sistemico_total = rs_actual
    return estado

def paso2_y_mercado_interbancario_integrado(estado, modo_impuesto='IRS'):
    """
    Fusión de Paso 2 y Mercado Interbancario.
    Secuencia:
    1. Empresas solicitan crédito.
    2. Bancos calculan brecha (Demanda - Efectivo).
    3. Bancos piden en Interbancario para cubrir brecha.
    4. Bancos otorgan préstamos a empresas.
    """
    
    # --- 1. SOLICITUD DE CRÉDITO Y COTIZACIONES ---
    indices_empresas_necesitadas = np.where(estado.demanda_credito_empresas > 1e-5)[0]
    if len(indices_empresas_necesitadas) == 0:
        return estado

    # Selección de bancos (Copy-paste lógica de selección de paso2 original)
    mascara_cotizaciones = np.zeros((N_EMPRESAS, N_BANCOS), dtype=bool)
    matriz_aleatoria = np.random.rand(len(indices_empresas_necesitadas), N_BANCOS)
    top_n_bancos = np.argsort(matriz_aleatoria, axis=1)[:, :N_SOLICITUDES_CREDITO]
    indices_fila = indices_empresas_necesitadas[:, np.newaxis]
    mascara_cotizaciones[indices_fila, top_n_bancos] = True
    
    deuda_empresa = np.sum(estado.prestamos_banco_empresa, axis=0)
    recursos_empresa = estado.efectivo_empresas + valor_stock_empresa(estado)
    apalancamiento_empresa = np.divide(deuda_empresa, np.maximum(recursos_empresa, 1.0))
    
    factor_riesgo_empresa = np.tanh(apalancamiento_empresa)
    especificidad_banco = np.random.uniform(0, 1, size=N_BANCOS)
    matriz_riesgo = np.outer(especificidad_banco, factor_riesgo_empresa)
    tasas_ofrecidas = TASA_REFINANCIACION * (1.0 + matriz_riesgo)
    tasas_finales = np.where(mascara_cotizaciones.T, tasas_ofrecidas, np.inf)
    
    indices_mejor_banco = np.argmin(tasas_finales, axis=0) # (F,)
    tasas_mejor_banco = np.min(tasas_finales, axis=0)      # (F,)
    
    # --- 2. CALCULAR DEMANDA AGREGADA POR BANCO ---
    demanda_por_banco = np.zeros(N_BANCOS)
    
    # Mapear demanda de empresa a su banco elegido
    # Solo consideramos demandas válidas (tasa < MAX)
    demanda_valida_mask = tasas_mejor_banco < TASA_INTERES_MAXIMA
    
    # Iterar para sumar (podria vectorizarse pero loop es claro)
    empresas_potenciales = [] # Lista de tuplas (empresa, banco, monto, tasa)
    
    for f_idx in indices_empresas_necesitadas:
        if not demanda_valida_mask[f_idx]:
            # Demanda contraída
            demanda_red = estado.demanda_credito_empresas[f_idx] * CONTRACCION_DEMANDA_CREDITO
            # Aceptamos tasa alta? No, el paper dice que si > MAX, reducen demanda. 
            # Asumiremos que si reduce demanda, quizas acepta la tasa o busca otro banco?
            # Simplificación: Si tasa > MAX, reduce demanda y NO toma préstamo (o toma parcial?)
            # El codigo original decía: if tasa > MAX: demanda *= CONTRACTION.
            # Y luego chequeaba if banco tiene cash.
            # Asumimos que la empresa SÍ quiere el prestamo reducido.
            
            # Pero la tasa sigue siendo la ofrecida?
            pass 
        
        banco = indices_mejor_banco[f_idx]
        tasa = tasas_mejor_banco[f_idx]
        demanda = estado.demanda_credito_empresas[f_idx]
        
        if tasa > TASA_INTERES_MAXIMA:
             demanda *= CONTRACCION_DEMANDA_CREDITO
        
        demanda_por_banco[banco] += demanda
        empresas_potenciales.append((f_idx, banco, demanda, tasa))
        
    # --- 3. MERCADO INTERBANCARIO (Cubrir Déficits) ---
    legislacion_liquidez = demanda_por_banco # Necesitamos cubrir la demanda
    efectivo_actual = estado.efectivo_bancos.copy()
    
    déficit = np.maximum(0.0, legislacion_liquidez - efectivo_actual)
    
    # Llamar al interbancario para cubrir 'déficit'
    estado = ejecutar_mercado_interbancario_demanda(estado, déficit, modo_impuesto)
    
    # --- 4. OTORGAMIENTO DE CRÉDITO (Con nueva liquidez) ---
    for f_idx, banco, demanda, tasa in empresas_potenciales:
        if estado.efectivo_bancos[banco] >= demanda:
            estado.efectivo_bancos[banco] -= demanda
            estado.efectivo_empresas[f_idx] += demanda
            estado.prestamos_banco_empresa[banco, f_idx] += demanda
            estado.tasas_interes_prestamos[banco, f_idx] = tasa
            
            estado.nuevos_prestamos_otorgados[f_idx] = demanda
            estado.eleccion_prestamista_empresa[f_idx] = banco
        else:
            # Racionamiento / Credit Crunch Parcial
            monto_posible = estado.efectivo_bancos[banco]
            if monto_posible > 1e-4:
                estado.efectivo_bancos[banco] -= monto_posible
                estado.efectivo_empresas[f_idx] += monto_posible
                estado.prestamos_banco_empresa[banco, f_idx] += monto_posible
                estado.tasas_interes_prestamos[banco, f_idx] = tasa
                
                estado.nuevos_prestamos_otorgados[f_idx] = monto_posible
                estado.eleccion_prestamista_empresa[f_idx] = banco

    return estado

def paso3_produccion(estado):
    """
    Paso 3: Mercado Laboral y Producción.
    """
    
    # 1. Restricción Presupuestaria
    labor_max_pagable = np.floor(estado.efectivo_empresas / TASA_SALARIAL)
    
    # 2. Demanda Efectiva
    labor_deseada = np.minimum(estado.demanda_trabajo_empresas, labor_max_pagable)
    
    demanda_laboral_total = np.sum(labor_deseada)
    oferta_laboral_total = N_HOGARES
    
    # 3. Matching
    trabajadores_contratados = np.zeros(N_EMPRESAS, dtype=np.int32)
    
    if demanda_laboral_total <= oferta_laboral_total:
        seguro_deseado = np.maximum(0, labor_deseada)
        trabajadores_contratados = np.floor(seguro_deseado).astype(np.int32)
    else:
        ratio_racionamiento = oferta_laboral_total / demanda_laboral_total
        seguro_deseado = np.maximum(0, labor_deseada)
        trabajadores_contratados = np.floor(seguro_deseado * ratio_racionamiento).astype(np.int32)
        
        actual_contratado = np.sum(trabajadores_contratados)
        remanente = oferta_laboral_total - actual_contratado
        if remanente > 0:
            empresas_suertudas = np.random.choice(N_EMPRESAS, int(remanente), replace=True)
            np.add.at(trabajadores_contratados, empresas_suertudas, 1)
            
    # Actualizar hogares
    estado.empleador_hogares[:] = -1
    ids_empleadores = np.repeat(np.arange(N_EMPRESAS), trabajadores_contratados)
    conteo_asignado = min(len(ids_empleadores), N_HOGARES)
    ids_empleadores = ids_empleadores[:conteo_asignado]
    estado.empleador_hogares[:conteo_asignado] = ids_empleadores
    
    # 4. Pago de Salarios
    salarios_pagados = trabajadores_contratados * TASA_SALARIAL
    estado.efectivo_empresas -= salarios_pagados
    
    mascara_empleados = estado.empleador_hogares != -1
    estado.efectivo_hogares[mascara_empleados] += TASA_SALARIAL
    
    # 5. Producción Real
    produccion_real = trabajadores_contratados * PRODUCTIVIDAD_LABORAL
    estado.stock_empresas += produccion_real
    estado.stock_empresas = np.maximum(estado.stock_empresas, 0.0)
    
    return estado

def paso4_consumo(estado):
    """
    Paso 4: Consumo de los Hogares.
    """
    
    # 1. Presupuesto
    presupuestos_hogar = estado.efectivo_hogares * PROPENSION_CONSUMO
    
    # 2. Selección de Proveedores
    elecciones_oferta = np.random.randint(0, N_EMPRESAS, size=(N_HOGARES, N_SOLICITUDES_CONSUMO))
    precios_elegidos = estado.precios_empresas[elecciones_oferta]
    
    idx_mejor_local = np.argmin(precios_elegidos, axis=1)
    empresas_ganadoras = elecciones_oferta[np.arange(N_HOGARES), idx_mejor_local]
    precios_ganadores = precios_elegidos[np.arange(N_HOGARES), idx_mejor_local]
    
    # 3. Calcular Demanda Deseada
    cantidades_deseadas = np.zeros_like(presupuestos_hogar)
    precios_validos = precios_ganadores > 1e-9
    cantidades_deseadas[precios_validos] = presupuestos_hogar[precios_validos] / precios_ganadores[precios_validos]
    
    # 4. Agregación
    demanda_total_por_empresa = np.bincount(empresas_ganadoras, weights=cantidades_deseadas, minlength=N_EMPRESAS)
    
    # 5. Racionamiento
    stock_disponible = estado.stock_empresas
    factor_racionamiento = np.ones(N_EMPRESAS, dtype=np.float64)
    mascara_escasez = demanda_total_por_empresa > stock_disponible
    
    denominador = np.maximum(demanda_total_por_empresa, 1e-9)
    factor_racionamiento[mascara_escasez] = stock_disponible[mascara_escasez] / denominador[mascara_escasez]
    
    # 6. Ejecución
    racionamiento_hogar = factor_racionamiento[empresas_ganadoras]
    cantidades_reales = cantidades_deseadas * racionamiento_hogar
    gasto_real = cantidades_reales * precios_ganadores
    
    estado.efectivo_hogares -= gasto_real
    
    ingresos_empresa = np.bincount(empresas_ganadoras, weights=gasto_real, minlength=N_EMPRESAS)
    estado.efectivo_empresas += ingresos_empresa
    
    volumen_ventas = np.bincount(empresas_ganadoras, weights=cantidades_reales, minlength=N_EMPRESAS)
    estado.stock_empresas -= volumen_ventas
    estado.stock_empresas = np.maximum(estado.stock_empresas, 0.0)
    
    # Guardar ventas para el paso de evolución
    estado.ventas_diarias_empresas = volumen_ventas
    
    return estado

def paso5_repago_empresas(estado):
    """
    Paso 5: Servicio de Deuda, Repago y Quiebras.
    """
    
    # 1. Calcular Obligaciones
    principal_debido = estado.prestamos_banco_empresa * TASA_REEMBOLSO_DEUDA
    interes_debido = estado.prestamos_banco_empresa * estado.tasas_interes_prestamos
    total_deuda_matriz = principal_debido + interes_debido
    
    total_deuda_por_empresa = np.sum(total_deuda_matriz, axis=0)
    
    # 2. Identificar Solventes e Insolventes
    tiene_deuda = total_deuda_por_empresa > 1e-9
    mascara_default = np.zeros(N_EMPRESAS, dtype=bool)
    mascara_default[tiene_deuda] = estado.efectivo_empresas[tiene_deuda] < total_deuda_por_empresa[tiene_deuda]
    
    mascara_solventes = ~mascara_default & tiene_deuda
    
    # --- 3. Procesar Solventes ---
    if np.any(mascara_solventes):
        estado.efectivo_empresas[mascara_solventes] -= total_deuda_por_empresa[mascara_solventes]
        
        matriz_pagos = total_deuda_matriz[:, mascara_solventes]
        pagos_por_banco = np.sum(matriz_pagos, axis=1)
        estado.efectivo_bancos += pagos_por_banco
        
        matriz_amortizacion = principal_debido[:, mascara_solventes]
        estado.prestamos_banco_empresa[:, mascara_solventes] -= matriz_amortizacion
        
        matriz_ingreso_interes = interes_debido[:, mascara_solventes]
        ingreso_interes_banco = np.sum(matriz_ingreso_interes, axis=1)
        estado.patrimonio_bancos += ingreso_interes_banco
        
    # --- 4. Procesar Defaults ---
    estado.deuda_incobrable_bancos[:] = 0.0
    
    if np.any(mascara_default):
        efectivo_disponible = estado.efectivo_empresas[mascara_default]
        total_debido = total_deuda_por_empresa[mascara_default]
        
        tasa_recuperacion = efectivo_disponible / total_debido
        
        deuda_sub = total_deuda_matriz[:, mascara_default]
        matriz_recuperada = deuda_sub * tasa_recuperacion[np.newaxis, :]
        
        prestamos_sub = estado.prestamos_banco_empresa[:, mascara_default]
        matriz_cambio_patrimonio = matriz_recuperada - prestamos_sub
        
        total_recuperado_banco = np.sum(matriz_recuperada, axis=1)
        total_cambio_patrimonio = np.sum(matriz_cambio_patrimonio, axis=1)
        
        estado.efectivo_bancos += total_recuperado_banco
        estado.patrimonio_bancos += total_cambio_patrimonio
        
        estado.deuda_incobrable_bancos -= np.minimum(0, total_cambio_patrimonio)
        
        # --- REEMPLAZO DE EMPRESAS Y RESPONSABILIDAD PERSONAL ---
        estado.prestamos_banco_empresa[:, mascara_default] = 0.0
        estado.tasas_interes_prestamos[:, mascara_default] = 0.0
        
        # 1. Identificar dueños y cobrarles
        duenos_afectados = estado.duenos_empresas[mascara_default]
        # El dinero sale de los hogares
        estado.efectivo_hogares[duenos_afectados] -= EFECTIVO_INICIAL_EMPRESA
        
        # 2. Reiniciar empresa con ese dinero (Transferencia, no creación)
        estado.efectivo_empresas[mascara_default] = EFECTIVO_INICIAL_EMPRESA
        estado.stock_empresas[mascara_default] = 0.0
        estado.defaults_acumulados_empresas[mascara_default] += 1
        
        # 3. Asignar nuevos dueños (opcional, pero seguimos lógica de "start a new company")
        estado.duenos_empresas[mascara_default] = np.random.randint(0, N_HOGARES, size=np.sum(mascara_default))
        
    return estado

def calcular_debtrank(matriz_interbancaria, patrimonio_bancos):
    """
    Calcula el Riesgo Sistémico Total usando una aproximación de DebtRank.
    """
    patrimonio_seguro = np.maximum(patrimonio_bancos, 1e-2)
    
    # Matriz de Impacto: W_ij
    matriz_impacto = np.minimum(1.0, matriz_interbancaria / patrimonio_seguro[:, np.newaxis])
    
    total_rs = 0.0
    
    for k in range(N_BANCOS):
        h = np.zeros(N_BANCOS)
        h[k] = 1.0
        
        while True:
            estres_entrante = matriz_impacto @ h
            h_siguiente = np.minimum(1.0, h + estres_entrante)
            
            if np.allclose(h, h_siguiente, atol=1e-5):
                break
            h = h_siguiente
            
        perdidas = h * patrimonio_seguro
        perdidas[k] = 0.0
        
        dr_k = np.sum(perdidas)
        total_rs += dr_k
        
    return total_rs

def ejecutar_contagio_interbancario(estado):
    """
    Motor de Contagio Recursivo.
    Devuelve: (estado, numero_bancos_caidos)
    """
    bancos_defaultados = estado.patrimonio_bancos < 0
    
    nuevos_defaults = list(np.where(bancos_defaultados)[0])
    defaults_procesados = set(nuevos_defaults) # Optimización: set inicial
    nuevos_defaults_totales = set(nuevos_defaults) # Para contar total afectados en esta cascada
    
    # Cola de procesamiento
    cola_procesamiento = list(nuevos_defaults)

    while len(cola_procesamiento) > 0:
        defaulter_actual = cola_procesamiento.pop(0)
        
        indices_prestamistas = np.where(estado.matriz_interbancaria[:, defaulter_actual] > 0)[0]
        
        for prestamista in indices_prestamistas:
            exposicion = estado.matriz_interbancaria[prestamista, defaulter_actual]
            perdida = exposicion * LGD_INTERBANCARIO
            
            estado.patrimonio_bancos[prestamista] -= perdida
            estado.matriz_interbancaria[prestamista, defaulter_actual] = 0.0
            
            if estado.patrimonio_bancos[prestamista] < 0 and prestamista not in defaults_procesados:
                defaults_procesados.add(prestamista)
                nuevos_defaults_totales.add(prestamista)
                cola_procesamiento.append(prestamista)
                
    return estado, len(nuevos_defaults_totales)

def paso6_mercado_interbancario(estado, modo_impuesto='IRS'):
    """
    Paso 6: Mercado Interbancario Racional (Best Quote) con IRS (o SRT en inglés).
    """
    
    # 1. Calcular Liquidez
    activos_reales = np.sum(estado.prestamos_banco_empresa, axis=1)
    activos_interbancarios = np.sum(estado.matriz_interbancaria, axis=1)
    activos_totales = activos_reales + activos_interbancarios
    
    liquidez_objetivo = activos_totales * RATIO_COLCHON_LIQUIDEZ
    brecha_liquidez = liquidez_objetivo - estado.efectivo_bancos
    
    prestatarios = np.where(brecha_liquidez > 1.0)[0]
    prestamistas = np.where(brecha_liquidez < -1.0)[0]
    
    np.random.shuffle(prestatarios)
    
    rs_actual = calcular_debtrank(estado.matriz_interbancaria, estado.patrimonio_bancos)
    estado.riesgo_sistemico_total = rs_actual
    
    impuestos_recaudados = 0.0
    N_COTIZACIONES = 5 
    
    for idx_prestatario in prestatarios:
        necesario = brecha_liquidez[idx_prestatario]
        if necesario < 1.0: continue
        
        prestamistas_actuales = prestamistas[brecha_liquidez[prestamistas] < -1.0]
        if len(prestamistas_actuales) == 0: break
        
        if len(prestamistas_actuales) > N_COTIZACIONES:
            candidatos = np.random.choice(prestamistas_actuales, N_COTIZACIONES, replace=False)
        else:
            candidatos = prestamistas_actuales
            
        mejor_prestamista = -1
        mejor_costo_tasa = float('inf')
        mejor_impuesto = 0.0
        mejor_monto = 0.0
        
        total_patrimonio_sistema = max(np.sum(estado.patrimonio_bancos), 1.0)
        
        for idx_prestamista in candidatos:
            superavit = -brecha_liquidez[idx_prestamista]
            monto = min(necesario, superavit)
            
            impuesto = 0.0
            if modo_impuesto == 'IRS': # IRS = SRT
                matriz_sim = estado.matriz_interbancaria.copy()
                matriz_sim[idx_prestamista, idx_prestatario] += monto
                
                nuevo_rs_sim = calcular_debtrank(matriz_sim, estado.patrimonio_bancos)
                delta_rs = max(0.0, nuevo_rs_sim - rs_actual)
                
                delta_normalizado = delta_rs / total_patrimonio_sistema
                impuesto = monto * SENSIBILIDAD_IRS * delta_normalizado
                
            elif modo_impuesto == 'ITF': # ITF = FTT
                impuesto = monto * TASA_ITF
            
            costo_interes = monto * TASA_INTERBANCARIA
            costo_total_abs = costo_interes + impuesto
            tasa_efectiva = costo_total_abs / monto if monto > 0 else float('inf')
            
            if tasa_efectiva < mejor_costo_tasa:
                mejor_costo_tasa = tasa_efectiva
                mejor_prestamista = idx_prestamista
                mejor_impuesto = impuesto
                mejor_monto = monto
        
        if mejor_prestamista != -1 and mejor_costo_tasa < 0.20:
            idx_prestamista = mejor_prestamista
            monto = mejor_monto
            impuesto = mejor_impuesto
            
            estado.matriz_interbancaria[idx_prestamista, idx_prestatario] += monto
            estado.efectivo_bancos[idx_prestamista] -= monto
            estado.efectivo_bancos[idx_prestatario] += monto
            
            if estado.efectivo_bancos[idx_prestatario] >= impuesto:
                estado.efectivo_bancos[idx_prestatario] -= impuesto
                impuestos_recaudados += impuesto
                # Acumular en Fondo de Rescate (Stock-Flow Consistency)
                estado.fondo_rescate += impuesto
            else:
                estado.matriz_interbancaria[idx_prestamista, idx_prestatario] -= monto
                estado.efectivo_bancos[idx_prestamista] += monto
                estado.efectivo_bancos[idx_prestatario] -= monto
                continue
            
            brecha_liquidez[idx_prestatario] -= monto
            brecha_liquidez[idx_prestamista] += monto
            
            rs_actual = calcular_debtrank(estado.matriz_interbancaria, estado.patrimonio_bancos)
            
    estado.impuesto_recaudado = impuestos_recaudados
    estado.riesgo_sistemico_total = rs_actual
    
    estado, cascada_size = ejecutar_contagio_interbancario(estado)
    
    return estado, cascada_size

def paso7_evolucion(estado):
    """
    Paso 7: Actualización y Evolución (Aprendizaje).
    """
    stock_invendido = estado.stock_empresas
    
    # Ajuste de Cantidad basado en Ventas (disponibles en estado tras update)
    # Si tenemos ventas diarias guardadas:
    # Pero para no complicar, usaremos la heurística de stock que ya estaba, 
    # o la versión mejorada si quisiéramos.
    # Usaremos la lógica original traducida.
    
    ajuste = -VELOCIDAD_ADAPTACION * stock_invendido
    
    mascara_agotado = stock_invendido < 1e-4
    ajuste[mascara_agotado] = VELOCIDAD_ADAPTACION * estado.demanda_esperada_empresas[mascara_agotado] * 0.5
    
    estado.demanda_esperada_empresas += ajuste
    estado.demanda_esperada_empresas = np.maximum(estado.demanda_esperada_empresas, 1.0)
    
    # Ajuste de Precios
    estado.precios_empresas[~mascara_agotado] *= (1.0 - TASA_AJUSTE_PRECIO)
    estado.precios_empresas[mascara_agotado] *= (1.0 + TASA_AJUSTE_PRECIO)
    
    estado.precios_empresas = np.clip(estado.precios_empresas, 0.1, 100.0)
    
    return estado
