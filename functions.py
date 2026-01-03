import numpy as np

def compute_debtrank(L, C, v, initial_distress=None, steps=100):
    """
    Calculate DebtRank for each node in a financial network.
    Supports both single matrix (B, B) and batch of matrices (K, B, B).
    
    Args:
        L: Liability tensor. Shape (B, B) or (Batch, B, B). Rows=Borrowers.
        C: Capital/Equity. Shape (B,) or (Batch, B).
        v: Economic Value. Shape (B,). (Used for weighting R).
        initial_distress (optional): Shape compatible with batch.
        steps: Max iterations.
    
    Returns:
        R: DebtRank values. Shape (Batch, B) or (B,).
    """
    # Ensure Batch dimensions for uniform handling
    if L.ndim == 2:
        L = L[np.newaxis, :, :]
        # If C is 1D (B,), make it (1, B). If it's already (1, B) leave it.
        # But if passed as (Batch, B), C might be (Batch, B).
        # We need to handle both.
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
    # h (distress) shape (K, B) ? 
    # Actually standard DebtRank computes the impact OF node i on System.
    # To get R_i for ALL i, we need to simulate B scenarios? 
    # Or does the recursion h(t+1) = h(t) + h(t)@W allow computing all?
    # Yes, if S represents the matrix of impacts where S_ij is distress of j caused by i.
    # Initial S = Identity (diagonal).
    
    S = np.zeros((K, B, B), dtype=np.float64)
    idx = np.arange(B)
    S[:, idx, idx] = 1.0 # Each node initially distressed 1.0 in its own scenario
    
    # 3. Recursive Dynamics
    # S_new = min(1, S + S @ W)
    
    for _ in range(steps):
        S_next = S + np.matmul(S, W)
        S_next = np.minimum(1.0, S_next)
        
        # Check convergence (optional optimization)
        if np.allclose(S, S_next, atol=1e-5):
            S = S_next
            break
        S = S_next
        
    # 4. Calculate R_i
    # R_i = Sum_j (S_ij * v_j) - S_ii * v_i
    # v is relative economic importance.
    # v shape (B,).
    
    # Normalize v to sum 1? Usually yes.
    v_norm = v / (np.sum(v) + 1e-10)
    
    # Broadcast v to (1, 1, B) to match S (K, Rows=Source, Cols=Target)
    v_broad = v_norm[np.newaxis, np.newaxis, :]
    
    # Weighted Impact S_ij * v_j
    Weighted_S = S * v_broad # (K, B, B)
    
    # Total Distress caused by source i (Sum over j)
    # Result shape (K, B) -> R for each node i in simulation k
    Total_Impact = np.sum(Weighted_S, axis=2) 
    
    # Subtract initial impact (self)
    # S_ii * v_i. Since S_ii starts at 1 and roughly stays 1 (capped), 1*v_i.
    self_impact = S[:, idx, idx] * v_norm[np.newaxis, :]
    
    R = Total_Impact - self_impact
    R = np.maximum(0.0, R)
    
    if single_mode:
        return R[0]
    else:
        return R

def compute_srt_tax(L_current, proposed_loans_indices, proposed_amounts, C, v, p_default, zeta):
    """
    Compute Systemic Risk Tax for a batch of proposed loans.
    
    Args:
        L_current: Current Liability matrix (B, B).
        proposed_loans_indices: List or Array of tuples/rows [(borrower, lender)]. Shape (N_props, 2).
        proposed_amounts: Array of amounts. Shape (N_props,).
        C: Current Capital (B,).
        v: Economic Value (B,).
        p_default: Probability of default (B,).
        zeta: Tax sensitivity parameter.
        
    Returns:
        taxes: Vector of tax amounts (N_props,).
    """
    N_props = len(proposed_amounts)
    if N_props == 0:
        return np.array([])
        
    B = L_current.shape[0]
    
    # 1. Baseline Systemic Loss
    # Single computation
    R_base = compute_debtrank(L_current, C, v)
    # Expected Loss = Sum(p_i * R_i * V_total)
    # Let's say V_total is sum(v) or passed v is absolute. 
    # Assuming v IS the value (e.g. Total Assets), then R is fraction.
    # EL = Sum(p * R * v_total) or Sum(p * R_absolute).
    # If compute_debtrank returns fraction of system value (0..1), then:
    V_total = np.sum(v)
    EL_base = np.sum(p_default * R_base) * V_total
    
    # 2. Batch Hypothetical Networks
    # Create Batch (N_props, B, B)
    # Start with L_current repeated
    L_batch = np.tile(L_current, (N_props, 1, 1))
    
    # Add loans
    # proposed_loans_indices is (N_props, 2) -> (Borrower, Lender)
    rows = proposed_loans_indices[:, 0]
    cols = proposed_loans_indices[:, 1]
    
    # Advanced indexing to add amounts
    # L_batch[k, row[k], col[k]] += amount[k]
    batch_indices = np.arange(N_props)
    L_batch[batch_indices, rows, cols] += proposed_amounts
    
    # 3. Compute DebtRank for Batch
    # Capital C is constant for the moment of decision? 
    # (Actually tax reduces C, but we calc tax based on risk BEFORE tax payment typically).
    R_batch = compute_debtrank(L_batch, C, v) # (N_props, B)
    
    # 4. Expected Loss for Batch
    # EL_new = Sum(p * R_new) * V_total (per scenario)
    # p_default (B,). R_batch (N_props, B).
    EL_new = np.sum(p_default[np.newaxis, :] * R_batch, axis=1) * V_total
    
    # 5. Marginal Contribution & Tax
    marginal_risk = EL_new - EL_base
    marginal_risk = np.maximum(0.0, marginal_risk)
    
    taxes = marginal_risk * zeta
    
    return taxes
