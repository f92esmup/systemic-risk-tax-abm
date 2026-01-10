# logica/debtrank.py
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
    Calcula el SRT (Systemic Risk Tax) basado en el incremento marginal de EL.
    """
    # Evitar divisiones por cero o patrimonios negativos en cálculos de prob
    patrimonio_seguro = np.maximum(bancos_patrimonio, 1e-9)

    # --- ESCENARIO PRE ---
    pasivo_total_pre = bancos_depositos + bancos_deuda_acumulada
    leverage_pre = pasivo_total_pre / patrimonio_seguro
    p_default_pre = p.ALPHA_P * np.tanh(leverage_pre)  # p.ALPHA_P suele ser 0.01

    R_pre, V_pre = calcular_debtrank(matriz_interbancaria, bancos_patrimonio)
    EL_pre = V_pre * np.sum(p_default_pre * R_pre)

    # --- ESCENARIO POST ---
    matriz_post = matriz_interbancaria.copy()
    matriz_post[banco_deudor_id, banco_acreedor_id] += monto_prestamo

    deuda_acumulada_post = bancos_deuda_acumulada.copy()
    deuda_acumulada_post[banco_deudor_id] += monto_prestamo

    pasivo_total_post = bancos_depositos + deuda_acumulada_post
    leverage_post = pasivo_total_post / patrimonio_seguro
    p_default_post = p.ALPHA_P * np.tanh(leverage_post)

    R_post, V_post = calcular_debtrank(matriz_post, bancos_patrimonio)
    EL_post = V_post * np.sum(p_default_post * R_post)

    return max(0.0, EL_post - EL_pre)


def calcular_debtrank(matriz_L, patrimonio):
    """
    Implementación vectorizada de DebtRank.
    matriz_L[i, j]: Deuda de i hacia j.
    """
    B = len(patrimonio)

    # Total Pasivos (Interbancarios) por banco
    total_pasivos = np.sum(matriz_L, axis=1)
    V_total = np.sum(total_pasivos)

    if V_total < 1e-9:
        return np.zeros(B), 0.0

    v = total_pasivos / V_total

    # Matriz de Impacto W[i, j]: Impacto de i sobre j
    W = np.zeros((B, B))
    # Vectorización cuidadosa: dividimos columna j por patrimonio j
    # Si patrimonio <= 0, impacto es 1 (absorción total de daño)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = matriz_L / patrimonio[None, :]  # Broadcast correcto
        W = np.minimum(1.0, ratio)
        W[:, patrimonio <= 0] = 1.0  # Bancos quebrados transmiten todo

    # DebtRank Recursivo
    R = np.zeros(B)

    for source in range(B):
        # Si el nodo no debe nada, no contagia por canal interbancario directo
        if total_pasivos[source] == 0:
            continue

        h = np.zeros(B)
        h[source] = 1.0  # Shock inicial

        # Dinámica de contagio
        # Iteramos B veces (suficiente para propagar en red de B nodos)
        for _ in range(B):
            h_prev = h.copy()
            # W[j, i] no existe en numpy directo, usamos transposición o lógica:
            # Impacto que RECIBE i de todos los j: Sum(W[j, i] * h[j])
            # W[j, i] es impacto de j sobre i.
            # En matriz W definida arriba: W[deudor, acreedor].
            # h es vector de distress de deudores.

            impacto_recibido = np.dot(h_prev, W)
            # Explicación dot: h (1xB) dot W (BxB) -> (1xB)
            # h[j] * W[j, i] sumado sobre j. Correcto.

            h = np.minimum(1.0, h_prev + impacto_recibido)

            # Si no hay cambios, break (optimización)
            if np.allclose(h, h_prev):
                break

        # Cálculo final R_i (excluyendo la pérdida del propio source)
        R[source] = np.sum(h * v) - (1.0 * v[source])

    return R, V_total
