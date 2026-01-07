# DebtRank:
from parametros import Param as p
import numpy as np


def calcular_impacto_sr(
    banco_deudor_id,
    banco_acreedor_id,
    monto_prestamo,
    matriz_interbancaria,
    bancos_patrimonio,
    bancos_depositos,
    bancos_deuda_acumulada,
):
    """
    Calcula el Impuesto de Riesgo Sistémico (SRT) para una transacción específica.
    Implementa la Ecuación 5 del paper comparando el escenario PRE y POST préstamo.

    Retorna:
        delta_el_euros (float): El incremento monetario en la Pérdida Esperada.
    """

    # --- ESCENARIO 1: PRE (Situación actual sin el nuevo préstamo) ---

    # 1. Probabilidad de Default (p_i) PRE
    pasivo_total_pre = bancos_depositos + bancos_deuda_acumulada
    # Evitamos div por cero
    patrimonio_seguro = np.maximum(bancos_patrimonio, 1e-9)

    leverage_pre = pasivo_total_pre / patrimonio_seguro
    # Fórmula Apéndice A.3: p = 0.01 * tanh(leverage)
    p_default_pre = 0.01 * np.tanh(leverage_pre)

    # 2. DebtRank y Valor Total PRE
    R_pre, V_total_pre = calcular_debtrank(matriz_interbancaria, bancos_patrimonio)

    # 3. Pérdida Sistémica Esperada (EL) PRE en Euros [Ec 1]
    # EL = Sum(p_i * R_i * V_total)
    EL_euros_pre = V_total_pre * np.sum(p_default_pre * R_pre)

    # --- ESCENARIO 2: POST (Situación hipotética con el préstamo) ---

    # 1. Actualizar Matriz Interbancaria (Copia temporal)
    matriz_post = matriz_interbancaria.copy()
    matriz_post[banco_deudor_id, banco_acreedor_id] += monto_prestamo

    # 2. Actualizar Probabilidad de Default POST
    # El deudor aumenta su pasivo, por tanto aumenta su apalancamiento y su riesgo.
    deuda_acumulada_post = bancos_deuda_acumulada.copy()
    deuda_acumulada_post[banco_deudor_id] += monto_prestamo

    pasivo_total_post = bancos_depositos + deuda_acumulada_post
    leverage_post = pasivo_total_post / patrimonio_seguro
    p_default_post = 0.01 * np.tanh(leverage_post)

    # 3. DebtRank y Valor Total POST
    R_post, V_total_post = calcular_debtrank(matriz_post, bancos_patrimonio)

    # 4. Pérdida Sistémica Esperada (EL) POST en Euros
    EL_euros_post = V_total_post * np.sum(p_default_post * R_post)

    # --- RESULTADO FINAL (El delta marginal) ---
    delta_el_euros = max(0.0, EL_euros_post - EL_euros_pre)

    return delta_el_euros


def calcular_debtrank(matriz_L, patrimonio):
    """
    Calcula el DebtRank de todos los bancos y la Pérdida Sistémica Esperada (EL).
    Basado en Apéndice D y Ecuación (5) del paper 1401.8026.

    Argumentos:
        matriz_L (np.array): Matriz (B, B) donde L[i, j] es la deuda de i hacia j.
                             (i es deudor, j es acreedor).
        patrimonio (np.array): Vector (B,) con el capital (equity) de cada banco.
        p_default_bancos (np.array): Vector (B,) con la probabilidad de default p_i.

    Retorna:
        EL_syst (float): Expected Systemic Loss total del sistema.
        R (np.array): Vector con el DebtRank R_i de cada banco.
    """
    B = p.B  # Número de bancos

    # 1. Total Liabilities y Valor Económico (v_i).Apéndice D, Ec D2
    # Total de deuda interbancaria de cada banco i.
    total_pasivos = np.sum(matriz_L, axis=1)
    V_total = np.sum(total_pasivos)

    if V_total == 0:
        return np.zeros(B), 0.0

    v = total_pasivos / V_total

    # 2. Matriz de Impacto W_ij. Apéndice D, Ec D1
    # Impacto de i (deudor) sobre j (prestamista o acredor).
    # W[i, j] = min(1, L[i,j] / C[j])
    # Nota: Si C[j] es <= 0, el impacto es máximo (1) ante cualquier pérdida.
    W = np.zeros((B, B))

    # Evitamos división por cero usando un epsilon o lógica condicional
    # Lo hacemos vectorizado:
    # L[i, j] / C[j]

    for j in range(B):
        if patrimonio[j] <= 0:
            # Si ya está quebrado, cualquier deuda es impacto total (o irrelevante según dinámica)
            # Asumimos 1.0 para consistencia matemática de la fórmula
            W[:, j] = 1.0
        else:
            W[:, j] = np.minimum(1.0, matriz_L[:, j] / patrimonio[j])

    # 3. Calcular DebtRank para cada nodo inicial s (source)
    R = np.zeros(B)

    # Algoritmo recursivo de DebtRank para cada posible shock inicial i
    for source in range(B):
        # Si el banco no tiene deuda, no puede causar contagio directo en esta red
        if total_pasivos[source] == 0:
            R[source] = 0
            continue

        # Inicialización de variables de estado Ec D3
        # h: nivel de distress [0, 1]
        # Estado {U (Undistressed), D (Distressed), I (Inactive)}
        # Usaremos: 0=U, 1=D, 2=I

        h = np.zeros(B)
        estado = np.zeros(B, dtype=int)  # 0: U

        # Choque inicial: El banco 'source' entra en default
        h[source] = 1.0
        estado[source] = 1  # D

        # Dinámica (hasta convergencia)
        for _ in range(B + 2):
            nodos_D = np.where(estado == 1)[0]
            if len(nodos_D) == 0:
                break

            h_prev = h.copy()

            # Actualización de h [Ec D3]
            # h_i(t) = min(1, h_i(t-1) + sum_j(W_ji * h_j(t-1)))
            # W_ji: Impacto de j sobre i.
            # OJO: W[j, i] es el impacto de j (deudor) sobre i (acreedor).
            # Si j está en distress, transmite a i.

            impacto_recibido = np.sum(
                W[nodos_D, :] * h_prev[nodos_D, np.newaxis], axis=0
            )
            h = np.minimum(1.0, h_prev + impacto_recibido)

            # Actualización de estados [Ec D4]
            estado[nodos_D] = 2  # D -> I

            # U -> D si h > 0 (usamos umbral numérico)
            nuevos_D = (estado == 0) & (h > 1e-10)
            estado[nuevos_D] = 1

        # DebtRank Final del nodo source [Ec D5]
        # R_i = Sum(h_j * v_j) - h_source * v_source
        # (El impacto en el sistema excluyendo la pérdida del propio source)
        loss_total = np.sum(h * v)
        R[source] = loss_total - (1.0 * v[source])

    return R, V_total
