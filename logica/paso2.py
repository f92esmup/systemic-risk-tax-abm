import numpy as np
from parametros import Param as p


def paso2(
    demanda_credito,  # Vector (F,) Demanda monetaria de empresas
    liquidez_bancos,  # Vector (B,) Liquidez actual
    equity_bancos,  # Vector (B,) Capital
    pasivos_interbancarios,  # Matriz (B, B) - Filas: Deudor (Borrower), Col: Prestamista (Lender)
    equity_empresas,  # Vector (F,)
    deuda_empresas,  # Vector (F,)
    modo_impuesto="NINGUNO",  # "NINGUNO", "TOBIN", "SRT"
):
    """
    Paso 2: Mercado de Crédito y Formación de Red Interbancaria.

    1. Calcula fragilidad financiera.
    2. Establece tasas interbancarias (incluyendo Impuestos SRT/Tobin).
    3. Subasta de crédito Empresas-Bancos y ejecución de préstamos.

    Ref: [cite: 183, 221, 652]
    """
    F = p.F
    B = p.B

    # -------------------------------------------------------------------------
    # 1. Fragilidad Financiera y Solvencia (Creditworthiness)
    # -------------------------------------------------------------------------
    # Empresas: mu = tanh(Leverage) [cite: 629]
    # Evitamos división por cero con np.divide y where
    lev_empresas = np.divide(
        deuda_empresas,
        equity_empresas,
        out=np.zeros_like(deuda_empresas),
        where=equity_empresas != 0,
    )
    mu_empresas = np.tanh(lev_empresas)

    # Bancos: mu = tanh(Leverage) [cite: 648]
    # L_ij: Banco i debe a Banco j.
    total_pasivos_bancos = np.sum(
        pasivos_interbancarios, axis=1
    )  # Suma filas (lo que debe i)
    lev_bancos = np.divide(
        total_pasivos_bancos,
        equity_bancos,
        out=np.zeros_like(total_pasivos_bancos),
        where=equity_bancos != 0,
    )
    mu_bancos = np.tanh(lev_bancos)

    # Factores idiosincráticos (Aleatoriedad operativa) [cite: 628, 647]
    chi = np.random.uniform(0, 1, B)  # Especificidad banco (Mercado Empresas)
    
    # Especificidad banco (Mercado Interbancario)
    # Refactorizado de hardcode 0.1 a parametro RANGO_PSI
    psi = np.random.uniform(0, p.RANGO_PSI, B)  

    # -------------------------------------------------------------------------
    # 2. Ofertas Mercado Interbancario (Cálculo Matricial)
    # -------------------------------------------------------------------------
    # Tasa que Banco 'j' (Lender) ofrece a Banco 'i' (Borrower)
    # r_ji = r_bar * (1 + psi_j * mu_i) [cite: 645]
    # Matriz (B, B): Filas=Borrower (i), Cols=Lender (j)

    # Broadcast: mu_bancos (B,1) vs psi (1,B)
    r_ib_base = p.R_BAR * (1 + mu_bancos[:, None] * psi[None, :])

    # --- CÁLCULO DE IMPUESTOS ---
    tax_matrix = np.zeros((B, B))
    metadata_debug = {}

    if modo_impuesto == "TOBIN":
        # Tasa plana sobre la transacción (aprox como spread) [cite: 250, 673]
        # Refactorizado hardcode 0.002 -> p.TASA_TOBIN (Eq. A6)
        tax_matrix[:, :] = p.TASA_TOBIN

    elif modo_impuesto == "SRT":
        # Calcular contribución marginal al riesgo sistémico para cada par posible
        # Usamos el leverage como proxy de probabilidad de default p_i [cite: 663]
        # Refactorizado hardcode 0.01 -> p.FACTOR_PROB_DEFAULT 
        probs_default = p.FACTOR_PROB_DEFAULT * mu_bancos  # Proxy definido en Eq A4

        tax_matrix, delta_el_matrix = calcular_matriz_srt(
            pasivos_interbancarios,
            equity_bancos,
            probs_default,
            p.ZETA,  # Sensibilidad del impuesto (ej. 0.02 o 1.0)
        )
        metadata_debug["delta_el"] = delta_el_matrix

    # Tasas Interbancarias Totales [cite: 670]
    r_ib_total = r_ib_base + tax_matrix

    # Determinar mejor prestamista potencial para cada banco (si necesitara fondos)
    # Diagonal infinita (no prestarse a sí mismo)
    np.fill_diagonal(r_ib_total, np.inf)

    # Cada fila 'i' busca el mínimo en columnas 'j'
    mejores_lenders_idx = np.argmin(r_ib_total, axis=1)  # Vector (B,)
    costos_refinanciacion = r_ib_total[np.arange(B), mejores_lenders_idx]

    # -------------------------------------------------------------------------
    # 3. Subasta de Crédito a Empresas
    # -------------------------------------------------------------------------
    # Empresas contactan 'n' bancos aleatorios [cite: 216]
    bancos_contactados = np.random.randint(0, B, size=(F, p.N_BANCOS_CONTACTADOS))

    # Matriz de ofertas: (F, N_contactados)
    ofertas_tasas = np.zeros((F, p.N_BANCOS_CONTACTADOS))

    for k in range(p.N_BANCOS_CONTACTADOS):
        b_indices = bancos_contactados[:, k]

        # Tasa base para empresa f del banco b [cite: 626]
        r_base = p.R_BAR * (1 + chi[b_indices] * mu_empresas)

        # Spread de refinanciación [cite: 652]
        # Si Banco necesita fondos (Demanda > Liquidez), añade costo interbancario.
        # Aproximación ABM: El banco evalúa la demanda de ESTA empresa vs su liquidez total.

        liquidez_disp = liquidez_bancos[b_indices]
        demanda_f = demanda_credito

        # Déficit positivo solo si demanda > liquidez
        deficit = np.maximum(demanda_f - liquidez_disp, 0)

        # Ratio del préstamo que debe ser refinanciado
        ratio = np.divide(
            deficit, demanda_f, out=np.zeros_like(deficit), where=demanda_f != 0
        )

        # Sumar costo interbancario (incluyendo tax) ponderado
        costo_ib = costos_refinanciacion[b_indices]
        ofertas_tasas[:, k] = r_base + (ratio * costo_ib)

    # -------------------------------------------------------------------------
    # 4. Selección y Racionamiento
    # -------------------------------------------------------------------------
    # Elegir banco con tasa mínima [cite: 216]
    idx_min = np.argmin(ofertas_tasas, axis=1)

    tasas_finales = ofertas_tasas[np.arange(F), idx_min]
    bancos_elegidos = bancos_contactados[np.arange(F), idx_min]

    # Racionamiento de crédito [cite: 217]
    # Si tasa > r_max, empresa pide menos (phi * demanda)
    mask_racionado = tasas_finales > p.R_MAX
    monto_solicitado = demanda_credito.copy()
    monto_solicitado[mask_racionado] *= p.PHI  # ej. 0.8

    # -------------------------------------------------------------------------
    # 5. Ejecución de Préstamos (Secuencial)
    # -------------------------------------------------------------------------
    nuevos_prestamos = np.zeros(F)
    nueva_matriz_ib = pasivos_interbancarios.copy()
    liquidez_actual = liquidez_bancos.copy()

    # Aleatorizar orden de ejecución para no favorecer a index 0
    orden = np.random.permutation(F)
    
    # Tolerancia numérica para float comparisons
    EPSILON = 1e-9

    for f in orden:
        monto = monto_solicitado[f]
        # Optimization: Skip negligible loan requests
        if monto <= 1e-6:
            continue

        b = bancos_elegidos[f]

        # Caso 1: Tiene liquidez propia
        # Usamos EPSILON para robustez numérica
        if liquidez_actual[b] >= (monto - EPSILON):
            liquidez_actual[b] -= monto
            nuevos_prestamos[f] = monto

        # Caso 2: Necesita interbancario
        else:
            propio = np.maximum(liquidez_actual[b], 0.0)
            faltante = monto - propio

            lender = mejores_lenders_idx[b]

            # Verificar si el prestamista interbancario tiene fondos
            if liquidez_actual[lender] >= (faltante - EPSILON):
                # Transacción exitosa
                liquidez_actual[lender] -= faltante
                liquidez_actual[b] = 0.0  # Usa todo lo propio

                # Actualizar Red de Pasivos: Banco b ahora debe a Lender
                nueva_matriz_ib[b, lender] += faltante

                nuevos_prestamos[f] = monto
            else:
                # Credit Crunch: Préstamo denegado
                nuevos_prestamos[f] = 0.0

    return (
        nuevos_prestamos,
        tasas_finales,
        nueva_matriz_ib,
        liquidez_actual,
        bancos_elegidos,
        tax_matrix, # [Refactor] Exportamos matrix impositiva para contabilidad
        metadata_debug,
    )


# -------------------------------------------------------------------------
# UTILIDADES: DebtRank & SRT
# -------------------------------------------------------------------------


def calcular_matriz_srt(L, equity, probs_default, zeta):
    """
    Calcula el Impuesto SRT para cada link potencial.
    
    SRT_ij = Zeta * max(0, EL_syst(+loan) - EL_syst(base))
    
    Implementa Ecuación 5 del paper [cite: Eq. 5] y cálculo de Marginal Systemic Risk Contribution.
    
    Complejidad: O(B^3) debido a DebtRank anidado.

    Ref: [cite: 136-138, 151-153]
    """
    B = L.shape[0]
    tax_matrix = np.zeros((B, B))
    delta_el_matrix = np.zeros((B, B))

    # 1. Calcular DebtRank Base (R_i) y Valor Económico (v_i)
    # Valor Económico v_i = Activos Interbancarios prestados / Total Sistema
    # Ref: [cite: 947] v_i = L_i / sum(L_j)
    total_lending = np.sum(L, axis=0)  # Sumar col (lo que i ha prestado a otros)
    V_total = np.sum(total_lending)

    # Optimization: Early exit if no interbank liabilities exist
    if V_total == 0:
        return tax_matrix, delta_el_matrix  # Si no hay red, no hay riesgo sistémico

    v = total_lending / V_total

    # Pérdida Sistémica Esperada Base (EL_base)
    # EL = sum(p_i * V_total * R_i) [cite: 111, 138]
    # Ecuación 3 del paper (implícita)
    R_base = calcular_debtrank_vector(L, equity, v)
    EL_base = np.sum(probs_default * V_total * R_base)

    # 2. Cálculo Marginal (Perturbación)
    # Simulamos añadir un préstamo unitario para ver el gradiente del riesgo
    delta_loan = p.DELTA_LOAN_TEST

    # Nota: Bucle O(B^2) llamando a DebtRank O(B). Complejidad O(B^3).
    # Para B=20 es muy rápido.
    for borrower in range(B):
        for lender in range(B):
            if borrower == lender:
                continue

            # Red Hipotética
            L_hypo = L.copy()
            L_hypo[borrower, lender] += delta_loan

            # Recalcular métricas hipotéticas
            total_lending_hypo = np.sum(L_hypo, axis=0)
            V_total_hypo = np.sum(total_lending_hypo)
            v_hypo = total_lending_hypo / V_total_hypo

            R_hypo = calcular_debtrank_vector(L_hypo, equity, v_hypo)
            EL_hypo = np.sum(probs_default * V_total_hypo * R_hypo)

            # Contribución Marginal [cite: 137]
            delta_EL = EL_hypo - EL_base
            delta_el_matrix[borrower, lender] = delta_EL

            # SRT (Tasa)
            if delta_EL > 0:
                # El paper define SRT como valor monetario.
                # Para pasarlo a tasa de interés, dividimos por el monto del préstamo.
                # Tax Rate = (Zeta * Delta_EL) / Loan_Amount
                tax_value = zeta * delta_EL
                tax_rate = tax_value / delta_loan
                tax_matrix[borrower, lender] = tax_rate

    return tax_matrix, delta_el_matrix


def calcular_debtrank_vector(L, C, v):
    """
    Calcula DebtRank R_i para todos los nodos.
    Ref: Appendix D [cite: 936-977]

    L: Matriz Pasivos (Filas=Deudor, Cols=Acreedor)
    C: Capital
    v: Valor Económico
    """
    B = len(C)

    # Matriz de Impacto W_ij: Impacto de i (deudor) sobre j (acreedor)
    # Si i quiebra, j pierde L[i,j].
    # W_ij = min(1, L[i,j] / C[j]) [cite: 943]

    W = np.zeros((B, B))
    for i in range(B):  # Defaulter potencial
        for j in range(B):  # Victima
            if C[j] > 0:
                loss = L[i, j]  # i le debe a j
                W[i, j] = min(1.0, loss / C[j])
            else:
                # Si j ya está quebrado (C <= 0), impacto es máximo si hay deuda
                if L[i, j] > 0:
                    W[i, j] = 1.0

    # Algoritmo Recursivo [cite: 958]
    # Calculamos el impacto R_i para cada nodo i iniciando un distress h_i=1

    R = np.zeros(B)

    for i in range(B):
        # Estado inicial: i colapsa
        h_initial = np.zeros(B)
        h_initial[i] = 1.0  # [cite: 956]
        
        h = h_initial.copy()

        # Propagación (Iterar hasta convergencia o máximo B pasos)
        for t in range(B):
            h_prev = h.copy()
            
            # DebtRank Correcto: Fixed Point Iteration
            # h(t+1) = min(1, h_initial + sum(h(t) * W))
            # No acumulamos el flujo sobre el nivel previo, sino que recalculamos 
            # el nivel total de distress sostenido por la red.
            
            impact_from_network = np.dot(h, W)
            h = np.minimum(1.0, h_initial + impact_from_network)

            if np.allclose(h, h_prev):
                break

        # R_i = sum(h_j * v_j) - h_i * v_i (Excluyendo pérdida directa propia) [cite: 971]
        R[i] = np.sum(h * v) - (h[i] * v[i])

    return R
