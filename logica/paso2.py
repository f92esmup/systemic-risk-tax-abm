import numpy as np


def calcular_riesgo_sistemico_scalar(L, equity_banks, v_override=None):
    """
    Calcula el Riesgo Sistémico Total H(L) del sistema bancario usando algoritmo iterativo.
    Implementa la lógica de DebtRank (Poledna & Thurner 2016).

    Args:
        L (matrix): Matriz de Pasivos Interbancarios (B, B). L[i, j] es lo que i debe a j.
                    (Fila=Deudor, Columna=Acreedor/Prestamista)
        equity_banks (vector): Capital de los bancos (B,).
        v_override (vector, optional): Vector de importancia económica precalculado.

    Returns:
        H (float): Nivel de riesgo sistémico escalar (0 a 1).
        R (vector): Vector DebtRank (impacto de cada banco).
    """
    B = len(equity_banks)
    T_steps_debtrank = 100  # Máximo pasos de propagación

    # 1. Matriz de Impacto W_ij (Eq. D1 del Artículo 1)
    # W[j, i] debe ser el impacto sobre j (Acreedor) causado por el default de i (Deudor).
    # Fórmula Artículo: W_ji = min(1, A_ji / E_j) donde A_ji es lo que i debe a j.
    # En nuestra matriz L: L[i, j] es lo que i debe a j.
    # Por tanto: W[j, i] = min(1, L[i, j] / E_j)

    equity_safe = np.where(equity_banks <= 0, 1e-10, equity_banks)

    # División: L / E_j (donde E_j varía por columnas en L)
    # L_scaled[i, j] = L[i, j] / E_j
    L_scaled = L / equity_safe[np.newaxis, :]

    # W[j, i] es la transpuesta de L_scaled
    # W[j, i] = L_scaled[i, j] = L[i, j] / E_j
    W = np.minimum(1.0, L_scaled.T)

    # 2. Valor Económico v_i (Eq. D2 del Artículo 1)
    if v_override is not None:
        v = v_override
    else:
        # "Dado el total de pasivos interbancarios pendientes del banco i..."
        # v_i = Total pasivos de i / Total pasivos del sistema
        # L[i, :] son las deudas de i. Sum(L, axis=1) es el total pasivo de i.
        total_liabilities_per_bank = np.sum(L, axis=1)
        total_val = np.sum(total_liabilities_per_bank)

        if total_val > 0:
            v = total_liabilities_per_bank / total_val
        else:
            v = np.zeros(B)

    # 3. DebtRank Vectorial R (Iterativo)
    # R_i = Impacto económico total si banco i entra en default
    R = np.zeros(B)

    for i in range(B):
        # Estado inicial: banco i en default
        h = np.zeros(B)
        h[i] = 1.0  # PSI_i = 1 (Default inicial)
        last_h = np.zeros(B)

        # Propagación dinámica (solo el incremento de distress se propaga)
        for _ in range(T_steps_debtrank):
            diff = h - last_h
            if np.sum(diff) < 1e-6:
                break

            last_h = h.copy()
            # Impacto en j (Acreedor) = Sum_k W[j, k] * diff_k (Deudor)
            impact = W @ diff
            h = np.minimum(1.0, h + impact)

        # R_i = Suma ponderada del distress final en el sistema
        # Artículo 1 Ec D5 (aprox): R_i = Sum(h_j * v_j) - (contribución propia inicial excluida a veces, aquí completa)
        R[i] = np.sum(h * v)

    # 4. Riesgo Escalar H (Eq. 5 del Artículo 1)
    # H es el DebtRank promedio ponderado por importancia económica
    H = np.sum(R * v)

    return H, R


def paso2(state, params):
    """
    PASO 2: Mercado de Crédito (Firms-Banks) e Interbancario (Banks-Banks)

    1. Empresas piden crédito a N bancos aleatorios y eligen el mejor.
    2. Bancos evalúan liquidez.
    3. Mercado Interbancario con SRT Marginal basado en Expected Systemic Loss (EL).
    """

    F = params.F
    B = params.B

    # --- A. DESEMPAQUETAR ESTADO ---
    L_demand_F = state["firms_labor_demand"]
    W_wage = state.get("firms_wage", np.full(F, params.W_BASE))

    Liq_F = state["firms_liquidity"]
    Liq_B = state["banks_liquidity"]
    Equity_B = state["banks_equity"]

    L_FB = state["net_FB"]
    L_BB = state["net_BB"]

    # --- B. MERCADO DE CRÉDITO (FIRMS -> BANKS) ---

    # 1. Necesidad de Crédito
    payroll = L_demand_F * W_wage
    credit_needed_F = np.maximum(0, payroll - Liq_F)

    # 2. Fragilidad Financiera de Firmas
    total_debt_F = np.sum(L_FB, axis=1)
    fragility_F = np.zeros(F)
    mask_pos = (Liq_F + total_debt_F) > 0
    fragility_F[mask_pos] = total_debt_F[mask_pos] / (
        Liq_F[mask_pos] + total_debt_F[mask_pos] + 1e-9
    )
    mu_F = np.tanh(fragility_F)

    # 3. Muestreo de Bancos (Sin duplicados por firma)
    N_CONTACTS = params.N_BANCOS_CONTACTADOS

    # Vectorized random sampling without replacement (efficient)
    rng = np.random.default_rng()
    rand_matrix = rng.random((F, B))
    candidate_banks = np.argsort(rand_matrix, axis=1)[:, :N_CONTACTS]

    # Especificidad chi
    chi_matrix = np.random.uniform(0, 0.1, (F, N_CONTACTS))

    # Calcular Tasas Ofertadas
    rates_offered = params.R_BAR * (1 + chi_matrix * mu_F[:, np.newaxis])

    # Solo bancos solventes pueden ser contactados (Credit Market)
    solvent_banks_mask = Equity_B > 0
    # Si un banco candidato está quebrado, su tasa es infinita
    is_solvent_chosen = solvent_banks_mask[candidate_banks]
    rates_offered[~is_solvent_chosen] = 1e9

    # 4. Selección del Mejor Banco
    best_idx_local = np.argmin(rates_offered, axis=1)
    chosen_banks = candidate_banks[np.arange(F), best_idx_local]
    chosen_rates = rates_offered[np.arange(F), best_idx_local]

    # 5. Toma de Crédito con Elasticidad
    mask_high_rate = chosen_rates > params.R_MAX
    credit_taken_F = credit_needed_F.copy()
    credit_taken_F[mask_high_rate] *= params.PHI

    # Ejecutar Préstamos
    Liq_F += credit_taken_F
    np.add.at(L_FB, (np.arange(F), chosen_banks), credit_taken_F)

    withdrawals_per_bank = np.bincount(
        chosen_banks, weights=credit_taken_F, minlength=B
    )
    Liq_B -= withdrawals_per_bank

    # --- C. MERCADO INTERBANCARIO (BANKS -> BANKS) ---

    deficit_B_mask = Liq_B < 0
    # Solo bancos solventes pueden prestar (Surplus y Equity > 0)
    surplus_B_mask = (Liq_B > 0) & (Equity_B > 0)

    idxs_deficit = np.where(deficit_B_mask)[0]
    idxs_surplus = np.where(surplus_B_mask)[0]

    tax_matrix = np.zeros((B, B))
    transactions_list = []

    if len(idxs_deficit) > 0 and len(idxs_surplus) > 0:
        # Factor para probabilidad de default (Ec. 5 Artículo 1)
        factor_pd = getattr(params, "FACTOR_PROB_DEFAULT", 0.01)
        safe_equity = np.maximum(Equity_B, 1e-9)

        # Fragilidad del Borrower (i) para tasas
        leverage_B = np.zeros(B)
        total_liab_IB = np.sum(L_BB, axis=1)
        mask_eq = Equity_B > 0
        leverage_B[mask_eq] = total_liab_IB[mask_eq] / Equity_B[mask_eq]
        mu_B = np.tanh(leverage_B)

        # Psi (especificidad) del Lender
        psi_B = np.random.uniform(0, params.RANGO_PSI, B)

        modo = getattr(params, "MODO_IMPUESTO", "NINGUNO")

        # Iterar sobre Bancos con Déficit
        for i_global in idxs_deficit:
            amount_needed = abs(Liq_B[i_global])
            if amount_needed < 1e-9:
                continue

            # Muestreo aleatorio en el interbancario
            num_surplus = len(idxs_surplus)
            n_contacts_ib = min(num_surplus, params.N_BANCOS_CONTACTADOS)

            sampled_j_locals = np.random.choice(
                num_surplus, n_contacts_ib, replace=False
            )
            sampled_j_globals = idxs_surplus[sampled_j_locals]

            # Calcular Pérdida Sistémica Esperada (EL) BASE para el bucle actual
            # 1. Estado Actual de la Red (Recalculado por si hubo cambios)
            _, R_loop = calcular_riesgo_sistemico_scalar(L_BB, Equity_B)

            # 2. Vector v (Importancia Económica)
            total_liab_loop = np.sum(L_BB, axis=1)
            total_sys_loop = np.sum(total_liab_loop)

            # 3. Vector p_default (Probabilidad de Default)
            # Proxy: tanh(Apalancamiento)
            leverage_loop = total_liab_loop / safe_equity
            p_default_loop = factor_pd * np.tanh(leverage_loop)

            # 4. EL Actual
            EL_loop = np.sum(p_default_loop * R_loop)

            offers = []

            for j_global in sampled_j_globals:
                available = Liq_B[j_global]
                if available <= 1e-9:
                    continue

                # 1. Tasa Base
                r_offer = params.R_BAR * (1 + psi_B[j_global] * mu_B[i_global])

                # 2. Impuesto Marginal (SRT)
                tax_rate = 0.0

                # Simular Préstamo y calcular Delta EL
                # Usamos min(amount_needed, available) como proxy del préstamo real
                amount_test = min(amount_needed, available)
                if amount_test < 1e-9:
                    amount_test = 1e-9  # Prevenir div/0

                L_sim = L_BB.copy()
                L_sim[i_global, j_global] += amount_test

                # -- Estado Simulado --
                # a. DebtRank Nuevo
                _, R_new = calcular_riesgo_sistemico_scalar(L_sim, Equity_B)

                # b. v Nuevo
                total_liab_new = np.sum(L_sim, axis=1)
                total_sys_new = np.sum(total_liab_new)

                # c. p_default Nuevo (Solo cambia el deudor i_global)

                leverage_new = total_liab_new / safe_equity
                p_default_new = factor_pd * np.tanh(leverage_new)

                # d. EL Nuevo
                EL_new = np.sum(p_default_new * R_new)

                # Delta EL Monetario (Ec. 3 Artículo 1)
                # EL_monetary = EL_adimensional * V_total
                EL_monetary_loop = EL_loop * total_sys_loop
                EL_monetary_new = EL_new * total_sys_new

                delta_monetary = max(0, EL_monetary_new - EL_monetary_loop)

                if modo == "SRT":
                    # Dimensionalidad del Impuesto
                    # Tasa Impuesto = SRT (Monto) / Monto_Prestamo

                    # 2. Calcular Monto del Impuesto
                    # ZETA es 1.0 (Internalización Completa) -> Tax = Delta EL Monetario
                    tax_amount = params.ZETA * delta_monetary

                    # 3. Convertir a Tasa
                    tax_rate = tax_amount / amount_test

                elif modo == "TOBIN":
                    tax_rate = params.TASA_TOBIN

                offers.append(
                    {
                        "j_global": j_global,
                        "total_cost": r_offer + tax_rate,
                        "tax": tax_rate,  # Esto ahora es una tasa
                        "delta": delta_monetary,  # Guardar delta monetario para estadísticas
                    }
                )

            # Ordenar y ejecutar
            offers.sort(key=lambda x: x["total_cost"])

            took_total = 0
            for off in offers:
                j_glob = off["j_global"]
                available = Liq_B[j_glob]
                if available <= 0:
                    continue

                amount = min(amount_needed - took_total, available)

                L_BB[i_global, j_glob] += amount
                Liq_B[i_global] += amount
                Liq_B[j_glob] -= amount
                tax_matrix[i_global, j_glob] = off["tax"]

                if amount > 0:
                    transactions_list.append(
                        {
                            "borrower": int(i_global),
                            "lender": int(j_glob),
                            "amount": float(amount),
                            "marginal_sr": float(off["delta"]),
                            "tax": float(off["tax"]),
                        }
                    )

                took_total += amount
                if took_total >= amount_needed - 1e-9:
                    break

    # --- D. ACTUALIZAR SALARIOS ---
    payroll_possible = Liq_F
    L_hired_F = np.minimum(L_demand_F, np.floor(payroll_possible / W_wage))
    wages_paid = L_hired_F * W_wage
    Liq_F -= wages_paid

    return {
        "net_FB": L_FB,
        "net_BB": L_BB,
        "firms_liquidity": Liq_F,
        "banks_liquidity": Liq_B,
        "firms_labor_demand": L_hired_F,
        "wages_paid_vector": wages_paid,
        "tax_matrix": tax_matrix,
        "transactions": transactions_list,
        "new_rates_FB": chosen_rates,
        "bank_indices": chosen_banks,
    }


def calcular_debtrank_vector(L, equity_banks, v_sys=None):
    _, R = calcular_riesgo_sistemico_scalar(L, equity_banks, v_override=v_sys)
    return R
