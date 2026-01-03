import numpy as np

def calculate_debtrank(L, equity, initial_distress=None):
    """
    Calculate DebtRank for each node in a financial network.
    Supports both single matrix (B, B) and batch of matrices (K, B, B).
    
    DebtRank Algorithm (Recursion):
    1. Impact Matrix W_ij = min(1, L_ij / E_j). (Impact of i on j).
       Note: L_ij is Liability of i towards j. If i defaults, j loses assets.
       So distress flows i -> j.
       State Update: h_new = h + h @ W. (Accumulate distress from neighbors).
    
    Args:
        L: Liability Matrix. Shape (K, B, B) or (B, B). 
           Rows=Borrowers, Cols=Lenders.
        equity: Equity vector. Shape (K, B) or (B,).
        initial_distress (optional): Shape compatible with batch.
    
    Returns:
        R: DebtRank of each node. Shape (K, B) or (B,).
           R_i = Sum_j (h_final_j * v_j) - h_initial_i * v_i
    """
    # Ensure Batch dimensions
    if L.ndim == 2:
        L = L[np.newaxis, :, :]
        equity = equity[np.newaxis, :]
        single_mode = True
    else:
        single_mode = False
        
    K, B, _ = L.shape
    
    # 1. Compute Impact Matrix W
    # W_ij = min(1, L_ij / E_j)
    # L shape: (K, B, B). Equity shape: (K, B). Need to broadcast equity to (K, 1, B)
    # Avoid division by zero
    E_broad = equity[:, np.newaxis, :]
    W = np.zeros_like(L)
    mask = E_broad > 0
    
    # If E_j > 0: W_ij = L_ij / E_j
    # If E_j <= 0: Node already dead. Usually W_ij = 0 or 1 depending on model.
    # Assuming standard DR: Impact is capped at 1.
    np.divide(L, E_broad, out=W, where=mask)
    W = np.minimum(1.0, W)
    
    # 2. State Initialization
    # We want DebtRank of *every* node. This means running B simulations per batch item?
    # Or can we compute all "Source -> System" impacts in one go?
    # State S: Shape (K, B, B). 
    # S[k, i, j] = Distress of node j caused by source node i in scenario k.
    
    if initial_distress is None:
        # Identity matrix: Each node i starts with distress 1.0 on itself.
        S = np.zeros((K, B, B), dtype=np.float64)
        # Set diagonal to 1 for all K
        idx = np.arange(B)
        S[:, idx, idx] = 1.0
    else:
        # Custom distress
        pass # Not needed for now
        
    # 3. Recursive Dynamics
    # h(t+1) = min(1, h(t) + h(t) @ W)
    # We ignore the feedback onto the already distressed nodes to avoid double counting?
    # Poledna & Thurner (2016) Eq D3: h_i(t+1) = min(1, h_i(t) + sum W_ji h_j(t) ) ??
    # Let's stick to simple matrix accumulation: S_new = S + S @ W
    # The 'min(1, ...)' simulates cap on total loss.
    
    max_steps = B + 2 # Diameter of graph
    for _ in range(max_steps):
        S_next = S + np.matmul(S, W)
        S_next = np.minimum(1.0, S_next)
        
        # Check convergence
        if np.allclose(S, S_next, atol=1e-5):
            S = S_next
            break
        S = S_next
        
    # 4. Calculate DebtRank Values
    # R_i = sum_j (S_ij * v_j) - S_ii * v_i
    # Need economic value v_j. Usually v_j = InterbankAssets_j / TotalInterbankAssets.
    # Or simpler: Relative Equity? Or just Sum of impacts?
    
    # Paper Eq D1: R_i = sum_j h_j(T) v_j - h_i(0) v_i.
    # We need to compute v (relative importance).
    # Let's assume v based on Total Assets (or just Equity?).
    # Standard: v_i = \sum_k A_{ki} / \sum_{k,m} A_{km} ? 
    # Let's use uniform v for now or derived?
    # Actually, R is typically fraction of TOTAL SYSTEM LOSS.
    # v_j = Value of j.
    # Let's define Value as Total Assets or Equity.
    # Prompt doesn't specify v distinct from Equity.
    # "Variables de Estado... Capital(Equity)...".
    # Often v_j = E_j / sum(E).
    
    total_equity = np.sum(equity, axis=1, keepdims=True) # (K, 1)
    
    # Avoid zero division
    v = np.zeros_like(equity)
    mask_e = total_equity > 0
    np.divide(equity, total_equity, out=v, where=mask_e) # (K, B)
    
    # Broadcast v to allow multiplication with S (K, B, B)
    # S[k, i, j] is distress of j. We weight it by v[k, j].
    # v shape (K, B). Need (K, 1, B)
    v_broad = v[:, np.newaxis, :]
    
    # Weighted Distress
    Weighted_S = S * v_broad # (K, B, B)
    
    # Sum over j (cols) to get total impact of i
    # Total Distress caused by i
    Total_Impact = np.sum(Weighted_S, axis=2) # (K, B)
    
    # Subtract initial distress of self (to perform "net" impact) or keep "gross"?
    # Eq D1 includes subtraction.
    # Initial S_ii = 1.0. Weighted = 1.0 * v_i.
    # So R_i = Total_Impact_i - v_i.
    
    R = Total_Impact - v # (K, B)
    
    # Clip small negatives due to float precision
    R = np.maximum(0.0, R)
    
    if single_mode:
        return R[0]
    else:
        return R

def compute_expected_systemic_loss(debtrank_vector, p_default, total_value):
    """
    Compute Expected Systemic Loss (EL_syst).
    
    EL = sum_i (p_i * R_i * V_total)
    
    Args:
        debtrank_vector: (K, B) or (B,)
        p_default: Probability of default for each bank. (K, B) or (B,)
        total_value: Scalar or (K, 1). Total economic value of system.
        
    Returns:
        EL: Scalar or (K,).
    """
    # Expected Systemic Loss ratio
    # ESL_ratio = sum(p * R)
    
    term = debtrank_vector * p_default # Element-wise
    
    esl_ratio = np.sum(term, axis=-1) # Sum over banks
    
    return esl_ratio * total_value
