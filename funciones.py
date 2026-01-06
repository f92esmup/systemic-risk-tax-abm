import numpy as np


def calcular_debtrank(L, C, v, steps=100):
    """
    Calculate DebtRank for each node in a financial network.
    Supports both single matrix (B, B) and batch of matrices (K, B, B).

    Args:
        L: Liability tensor. Shape (B, B) or (Batch, B, B). Rows=Borrowers.
        C: Capital/Equity. Shape (B,) or (Batch, B).
        v: Economic Value. Shape (B,). (Used for weighting R).
        steps: Max iterations.

    Returns:
        R: DebtRank values. Shape (Batch, B) or (B,).
    """
    # Ensure Batch dimensions for uniform handling
    if L.ndim == 2:
        L = L[np.newaxis, :, :]
        single_mode = True
    else:
        single_mode = False

    K, B, _ = L.shape

    if C.ndim == 1:
        C = np.tile(C, (K, 1))
    elif C.shape[0] != K and single_mode:
        C = C[np.newaxis, :]

    # 1. Compute Impact Matrix W (K, B, B)
    # W_ij = min(1, L_ij / E_j).
    # Broadcast C to (K, 1, B) to divide columns of L
    C_broad = C[:, np.newaxis, :]

    W = np.zeros_like(L)
    mask = C_broad > 0
    np.divide(L, C_broad, out=W, where=mask)
    W = np.minimum(1.0, W)

    # 2. State Initialization
    S = np.zeros((K, B, B), dtype=np.float64)
    idx = np.arange(B)
    S[:, idx, idx] = 1.0  # Initial shock
    
    # Delta stores the *new* distress to propagate
    Delta_S = S.copy()

    # 3. Recursive Dynamics (Delta Propagation)
    for _ in range(steps):
        # Only propagate the NEW distress (Delta)
        # New Impact = Delta_S @ W
        Impact = np.matmul(Delta_S, W)
        
        # Update Cumulative Stress
        S_next = np.minimum(1.0, S + Impact)
        
        # Calculate new Delta for next step
        Delta_S = S_next - S
        
        S = S_next
        
        # Convergence check: If no new distress, stop
        if np.all(Delta_S < 1e-5):
            break

    # 4. Calculate R_i
    v_norm = v / (np.sum(v) + 1e-10)
    v_broad = v_norm[np.newaxis, np.newaxis, :]
    Weighted_S = S * v_broad  # (K, B, B)
    Total_Impact = np.sum(Weighted_S, axis=2)
    self_impact = S[:, idx, idx] * v_norm[np.newaxis, :]

    R = Total_Impact - self_impact
    R = np.maximum(0.0, R)

    if single_mode:
        return R[0]
    else:
        return R


def calcular_impuesto_srt(
    L_current, proposed_loans_indices, proposed_amounts, C, v, p_default, zeta
):
    """
    Compute Systemic Risk Tax for a batch of proposed loans.
    """
    N_props = len(proposed_amounts)
    if N_props == 0:
        return np.array([])

    # 1. Baseline Systemic Loss
    R_base = calcular_debtrank(L_current, C, v)
    V_total = np.sum(v)
    EL_base = np.sum(p_default * R_base) * V_total

    # 2. Batch Hypothetical Networks
    L_batch = np.tile(L_current, (N_props, 1, 1))
    rows = proposed_loans_indices[:, 0]
    cols = proposed_loans_indices[:, 1]
    batch_indices = np.arange(N_props)
    L_batch[batch_indices, rows, cols] += proposed_amounts

    # 3. Compute DebtRank for Batch
    R_batch = calcular_debtrank(L_batch, C, v)  # (N_props, B)

    # 4. Expected Loss for Batch
    EL_new = np.sum(p_default[np.newaxis, :] * R_batch, axis=1) * V_total

    # 5. Marginal Contribution & Tax
    marginal_risk = EL_new - EL_base
    marginal_risk = np.maximum(0.0, marginal_risk)

    taxes = marginal_risk * zeta

    return taxes


# --- APPENDIX A: INTEREST RATE MECHANISM ---


def calcular_fragilidad_financiera(leverage, k_mu=10.0):
    """
    Eq A1/A2 helper: mu(l) = tanh(k_mu * leverage)
    """
    return np.tanh(k_mu * leverage)


def calcular_tasa_firma(r_bar, chi, fragility_firm):
    """
    Eq A1: r_if = r_bar * (1 + chi_i * mu(l_firm))

    Args:
        r_bar: float, benchmark rate
        chi: (Batch,) bank specificity
        fragility_firm: (Batch,) or scalar firm fragility

    Returns:
        rate: (Batch,)
    """
    return r_bar * (1 + chi * fragility_firm)


def calcular_tasa_interbancaria(r_bar, psi, fragility_borrower):
    """
    Eq A2: r_ij = r_bar * (1 + psi_i * mu(l_borrower))

    Args:
        r_bar: float
        psi: (Batch,) lender specificity
        fragility_borrower: (Batch,) or scalar borrower fragility

    Returns:
        rate: (Batch,)
    """
    return r_bar * (1 + psi * fragility_borrower)
