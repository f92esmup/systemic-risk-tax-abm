import numpy as np
from parametros import Param as p

def paso1(state, params):
    """
    PASO 1: Planificación de Producción y Precios (Adaptativo - Mark I Logic)
    
    Actualiza el precio y la producción objetivo de las empresas basándose en 
    el exceso de demanda/oferta del periodo anterior.
    
    Ref: Gualdi et al. (2014), Eq. (1)
         Poledna et al. (2016), Sec. III.A.2
    
    Args:
        state (dict): Diccionario conteniendo los arrays del sistema:
            - firms_production (F,) float64: Producción realizada en t-1
            - firms_demand (F,) float64: Demanda recibida en t-1
            - firms_prices (F,) float64: Precios en t-1
            - firms_wage (F,) float64: Salarios (W_BASE o variable)
        params (class): Clase Param con constantes (ALPHA, RANGOS, etc.)
        
    Returns:
        dict: Actualizaciones para el estado:
            - firms_target_production (F,)
            - firms_prices (F,)
            - firms_labor_demand (F,)
    """
    
    # Desempaquetar vectores para legibilidad (referencias, no copias costosas)
    Y_prev = state['firms_production']
    D_prev = state['firms_demand']
    P_prev = state['firms_prices']
    
    F = params.F
    
    # [FIX] Wage Update (Dynamic Wages - Paper 2 Extensión)
    # Se realiza antes de planificar producción t+1, usando resultados t-1.
    if 'firms_last_profit' in state and 'firms_wage' in state:
        W_prev = state['firms_wage']
        Profits_prev = state['firms_last_profit']
        
        # Calcular Desempleo (u) y Empleo (epsilon)
        # Asumimos que firms_labor_demand en state refleja los contratados t-1
        H_total = params.H
        Total_Employed = np.sum(state['firms_labor_demand'])
        u = max(0, 1.0 - (Total_Employed / H_total))
        epsilon = 1.0 - u
        
        # Aleatoriedad (xi) para salarios (gamma_w ~ 0.02 approx)
        gamma_w = 0.02
        xi_w = np.random.uniform(0, 1, F)
        
        # Mascaras de ajuste
        # Subir salario: Excess Demand (Y < D) y Beneficios > 0
        raise_wage_mask = (D_prev > Y_prev) & (Profits_prev > 0)
        
        # Bajar salario: Excess Supply (Y > D) y Beneficios < 0
        cut_wage_mask = (D_prev < Y_prev) & (Profits_prev < 0)
        
        # Aplicar Cambios
        # W(t+1) = W(t) * (1 + gamma * epsilon * xi) [Si demanda alta, sube más si hay pleno empleo]
        W_new = W_prev.copy()
        
        W_new[raise_wage_mask] *= (1 + gamma_w * epsilon * xi_w[raise_wage_mask])
        
        # W(t+1) = W(t) * (1 - gamma * u * xi) [Si oferta alta, baja más si hay mucho desempleo]
        W_new[cut_wage_mask] *= (1 - gamma_w * u * xi_w[cut_wage_mask])
        
        # Guardar
        state['firms_wage'] = np.maximum(W_new, params.W_BASE * 0.5) # Suelo salarial
        
    
    F = params.F
    
    # 1. Calcular Precio Promedio de Mercado (Weighted Average)
    # Se usa para comparar si el precio de la firma es alto o bajo
    total_sales = np.minimum(Y_prev, D_prev)
    denom = np.sum(total_sales)
    if denom > 0:
        P_avg = np.sum(P_prev * total_sales) / denom
    else:
        P_avg = np.mean(P_prev)
        
    # 2. Generar choques aleatorios idiosincráticos para ajuste (xi)
    # Gualdi Eq. 1 usa variables aleatorias uniformes U[0,1]
    xi = np.random.uniform(params.RANGO_AJUSTE_MIN, params.RANGO_AJUSTE_MAX, F)
    
    # 3. Lógica Vectorial de Ajuste (Eq. 1 Gualdi et al. 2014)
    # Definir máscaras booleanas para los 4 casos posibles
    # Fix: Incluir igualdad en excess_demand (Market Clearing es señal positiva)
    excess_demand = D_prev >= Y_prev
    excess_supply = ~excess_demand
    price_high    = P_prev >= P_avg
    price_low     = ~price_high
    
    # Inicializar nuevos vectores
    Y_target = Y_prev.copy()
    P_new = P_prev.copy()
    
    # Caso 1: Exceso Demanda & Precio Bajo -> Subir Precio
    # (El paper dice: si Y < D & P < P_bar => P(t+1) = P(t)(1 + gamma_p * xi))
    mask1 = excess_demand & price_low
    P_new[mask1] *= (1 + xi[mask1])
    
    # Caso 2: Exceso Demanda & Precio Alto -> Subir Producción
    # (El paper dice: si Y < D & P > P_bar => Y_target(t+1) = Y(t)(1 + gamma_y * xi))
    mask2 = excess_demand & price_high
    Y_target[mask2] *= (1 + xi[mask2])
    
    # Caso 3: Exceso Oferta & Precio Alto -> Bajar Precio
    # (El paper dice: si Y > D & P > P_bar => P(t+1) = P(t)(1 - gamma_p * xi))
    mask3 = excess_supply & price_high
    P_new[mask3] *= (1 - xi[mask3])
    
    # Caso 4: Exceso Oferta & Precio Bajo -> Bajar Producción
    # (El paper dice: si Y > D & P < P_bar => Y_target(t+1) = Y(t)(1 - gamma_y * xi))
    mask4 = excess_supply & price_low
    Y_target[mask4] *= (1 - xi[mask4])
    
    # 4. Asegurar cotas mínimas (Estabilidad Numérica)
    Y_target = np.maximum(Y_target, params.UMBRAL_INVENTARIO)
    P_new = np.maximum(P_new, P_avg * params.SUELO_PRECIO_RELATIVO)
    
    # 5. Calcular Demanda Laboral
    # Poledna et al. 2016: "firms define labour demand"
    # Asumiendo función de producción lineal Y = alpha * L => L = Y / alpha
    L_demand = np.ceil(Y_target / params.ALPHA).astype(int)
    
    return {
        'firms_target_production': Y_target,
        'firms_prices': P_new,
        'firms_labor_demand': L_demand
    }