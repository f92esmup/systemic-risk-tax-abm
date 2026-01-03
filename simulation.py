import numpy as np
from parameters import Params

class CRISIS_Model:
    """
    Vectorized implementation of the Poledna & Thurner (2016) ABM model.
    Uses tensor-based state management to avoid explicit agent loops.
    """

    def __init__(self, seed=None):
        """
        Initialize the simulation model.
        
        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """
        Reset or initialize all state tensors and network topologies for a new simulation run.
        """
        # --- Dimensions ---
        B = Params.B
        F = Params.F
        H = Params.H
        
        # --- 1. Agents State (Tensors) ---
        # Banks State Matrix (B, Features)
        # Features: [Liquidity, Equity, Deposits, Leverage, BadDebt, DebtRank, ...]
        # For now, we will define indices for these columns to keep track
        self.IDX_BANK_LIQUIDITY = 0
        self.IDX_BANK_EQUITY = 1
        self.IDX_BANK_DEPOSITS = 2
        self.IDX_BANK_LEVERAGE = 3
        self.IDX_BANK_BAD_DEBT = 4
        self.IDX_BANK_DEBTRANK = 5
        self.N_BANK_FEATURES = 6
        
        self.banks_state = np.zeros((B, self.N_BANK_FEATURES), dtype=np.float64)
        
        # Firms State Matrix (F, Features)
        # Features: [Liquidity, Equity, Price, Demand, Production, Workers, SalariosPagados]
        # We need Current and Previous state for some memory variables (Price, Demand)
        # Strategy: Use extra columns for previous values or a separate previous_state tensor.
        # Given "memory of \tau" for prices/demand, separate columns might be cleaner if just t-1
        self.IDX_FIRM_LIQUIDITY = 0
        self.IDX_FIRM_EQUITY = 1
        self.IDX_FIRM_PRICE = 2
        self.IDX_FIRM_DEMAND = 3
        self.IDX_FIRM_PRODUCTION = 4
        self.IDX_FIRM_WORKERS = 5
        self.IDX_FIRM_WAGES_PAID = 6
        # Previous steps
        self.IDX_FIRM_PRICE_PREV = 7
        self.IDX_FIRM_DEMAND_PREV = 8
        self.N_FIRM_FEATURES = 9

        self.firms_state = np.zeros((F, self.N_FIRM_FEATURES), dtype=np.float64)

        # Households State Matrix (H, Features)
        # Features: [Deposits]
        self.IDX_HH_DEPOSITS = 0
        self.N_HH_FEATURES = 1
        
        self.households_state = np.zeros((H, self.N_HH_FEATURES), dtype=np.float64)

        # --- 2. Random Vectorized Initialization ---
        # Initialize Bank Equity/Assets uniformly for now (Ref: Appendix A often implies random init)
        # Adjust values based on calibration needs later
        # Example: Banks differ in size
        self.banks_state[:, self.IDX_BANK_EQUITY] = self.rng.uniform(10, 20, size=B)
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] = self.banks_state[:, self.IDX_BANK_EQUITY] * 0.1 # 10% cash
        
        # Initialize Firms
        self.firms_state[:, self.IDX_FIRM_EQUITY] = self.rng.uniform(5, 10, size=F)
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] = self.firms_state[:, self.IDX_FIRM_EQUITY] * 0.2
        self.firms_state[:, self.IDX_FIRM_PRICE] = 1.0
        self.firms_state[:, self.IDX_FIRM_PRICE_PREV] = 1.0


        # --- 3. Topology & Relationships ---
        
        # A. Network Adjacency Matrices (The Graphs)
        # L_bb: Interbank Liabilities (B, B). Rows=Borrowers, Cols=Lenders.
        # L_ij = Amount i owes to j.
        self.L_bb = np.zeros((B, B), dtype=np.float64)
        
        # L_fb: Firm Liabilities (F, B). Rows=Firms, Cols=Lenders.
        self.L_fb = np.zeros((F, B), dtype=np.float64)

        # B. Agent Relationships (Index Maps)
        # Households are randomly assigned to banks and firms initially.
        
        # Assign each household to one employer (Firm)
        # Shape (H,), values in [0, F-1]
        self.hh_employer_idx = self.rng.integers(0, F, size=H)
        
        # Assign each household to one bank for deposits
        # Shape (H,), values in [0, B-1]
        self.hh_bank_idx = self.rng.integers(0, B, size=H)

        # --- 4. History / Traceability ---
        # Lists to store copies of adjacency matrices at each step.
        # Using lists is efficient for appending; convert to array for analysis later if needed.
        self.history_L_bb = []
        self.history_L_fb = []

        # Record initial state
        self.record_history()

    def record_history(self):
        """Append current topology specific copies to history."""
        self.history_L_bb.append(self.L_bb.copy())
        self.history_L_fb.append(self.L_fb.copy())

    def step_firms_planning(self):
        """
        Phase 1: Firms Planning.
        - Update Prices based on market average.
        - Update Expected Demand (simple stochastic process for now).
        - Calculate Labor requirements.
        - Calculate Credit Demand if Wages > Liquidity.
        """
        # 1. Update Prices
        # Calculate market average price
        p_avg = np.mean(self.firms_state[:, self.IDX_FIRM_PRICE])
        
        # Vector of current prices
        prices = self.firms_state[:, self.IDX_FIRM_PRICE]
        
        # Price adjustment rule: p(t) = p(t-1) * (1 + beta * (p_avg - p)/p_avg + noise)
        # Using self.IDX_FIRM_PRICE_PREV to store history if needed, but for Markov process we can just update.
        # Let's save to PREV first
        self.firms_state[:, self.IDX_FIRM_PRICE_PREV] = prices
        
        noise = self.rng.normal(0, Params.PRICE_DRIFT_STD, size=Params.F)
        
        # Avoid division by zero if p_avg is 0 (unlikely but safe)
        if p_avg > 0:
            adjustment = Params.PRICE_ADJUSTMENT_SPEED * (p_avg - prices) / p_avg
            new_prices = prices + prices * (adjustment + noise)
        else:
            new_prices = prices * (1 + noise)
            
        # Constraint: Prices must be positive and non-zero
        new_prices = np.maximum(new_prices, 0.01)
        
        self.firms_state[:, self.IDX_FIRM_PRICE] = new_prices

        # 2. Update Demand Expectations
        # Save previous demand
        self.firms_state[:, self.IDX_FIRM_DEMAND_PREV] = self.firms_state[:, self.IDX_FIRM_DEMAND]
        
        # Simple random walk for demand expectation: D(t) = D(prev) * (1 + noise)
        # Initialize if demand is 0 (start of sim)
        current_demand = self.firms_state[:, self.IDX_FIRM_DEMAND]
        if np.all(current_demand == 0):
             current_demand = self.rng.uniform(10, 50, size=Params.F) # Random init
        
        demand_shock = self.rng.normal(0, 0.05, size=Params.F) # 5% volatility
        new_demand = current_demand * (1 + demand_shock)
        new_demand = np.maximum(new_demand, 0.0) # Non-negative
        
        self.firms_state[:, self.IDX_FIRM_DEMAND] = new_demand

        # 3. Calculate Resource Needs (Labor & Capital)
        # Labor Demand L = Y / Alpha (where Y = Demand)
        # We round up to ensure we meet demand? Or round nearest? 
        # "Trabajadores necesarios (entero)" -> np.ceil implies meeting full demand.
        labor_needed = np.ceil(new_demand / Params.alpha)
        
        self.firms_state[:, self.IDX_FIRM_WORKERS] = labor_needed
        self.firms_state[:, self.IDX_FIRM_PRODUCTION] = labor_needed * Params.alpha # Potential production

        # 4. Wage Bill and Credit Demand
        wage_bill = labor_needed * Params.WAGE
        self.firms_state[:, self.IDX_FIRM_WAGES_PAID] = wage_bill # Storing liability
        
        liquidity = self.firms_state[:, self.IDX_FIRM_LIQUIDITY]
        
        # Gap = Wages - Liquidity
        gap = wage_bill - liquidity
        
        # Credit Demand > 0 only if Gap > 0
        credit_demand = np.maximum(gap, 0.0)
        
        # Store temporary credit demand (could be a class attribute or return it)
        # We'll store it in a class attribute for the next step (Interbank/Credit Market)
        self.current_credit_demand = credit_demand
        
        return credit_demand

    def step_banking_market(self):
        """
        Phase 2 & 3: Credit Market (Firms-Banks) & Interbank Market (SRT).
        
        Part A: Firms request credit from Banks.
        Part B: Banks manage liquidity deficits via Interbank Market using DebtRank-based SRT.
        """
        import functions as fn # Import here to avoid circular dependencies if any
        
        # --- PART A: CREDIT MARKET (Firms -> Banks) ---
        
        # 1. Firms select N_SEARCH banks randomly
        # Shape (F, N_SEARCH)
        pool_indices = self.rng.integers(0, Params.B, size=(Params.F, Params.N_SEARCH))
        
        # 2. Banks offer rates
        # Rate r_if = r_bar + r_loan * exp(Leverage_f) ... Simplified: r_bar * (1 + Leverage)
        # Calculate Firm Leverage: L = Debt / Equity.
        # Current Debt is small/zero in initialization? Or use L_fb.
        # Simplified: Leverage = 1.0 for now or based on state.
        # Let's say Rate is mostly generic but slightly varying by bank (e.g. random noise or bank health).
        # Paper Eq A1: r_ib = r_b / (1 + c_b) ... 
        # Using a simplified vector approach: Rate = r_bar + Random + FirmRisk
        
        rates = np.full((Params.F, Params.N_SEARCH), Params.r_bar)
        # Add slight variation per bank (using pool_indices to fetch bank traits if needed)
        # Here we just assume competitive market with random spread
        rates += self.rng.uniform(0, 0.01, size=(Params.F, Params.N_SEARCH))
        
        # 3. Firms choose best bank (Lowest Rate)
        best_choice_idx = np.argmin(rates, axis=1) # Index 0..N_SEARCH-1
        # Get actual Bank ID
        # advanced indexing: rows 0..F, cols best_choice_idx
        chosen_bank_ids = pool_indices[np.arange(Params.F), best_choice_idx] # Shape (F,)
        chosen_rates = rates[np.arange(Params.F), best_choice_idx]
        
        # 4. Aggregate Credit Demand per Bank
        # self.current_credit_demand is (F,)
        # We need to sum this up for each bank.
        bank_inflows_demand = np.zeros(Params.B)
        if hasattr(self, 'current_credit_demand'):
            np.add.at(bank_inflows_demand, chosen_bank_ids, self.current_credit_demand)
            
            # Record these new Liabilities in L_fb?
            # Technically, loan is granted only if bank has funds.
            # But usually in Phase A, we assume preliminary agreement.
            # Let's add them tentatively. Real logic checks liquidity later.
            # For simplicity in Phase 3, we assume Banks accepted and now need to fund it.
            # L_fb: Rows=Firms, Cols=Banks.
            # Add new debt
            # Optimize: use vectors
             # self.L_fb[f, b] += amount
            # Iterating might be slow if many firms? 100 is fine.
            # Vectorized add?
            # np.add.at for 2D is harder. But we have (Firm_Index, Bank_Index).
            # We can flat index? Or just loop F (100 is small).
            # For strict vectorization:
            # Construct a sparse-like update or just loop 100. 100 is negligible.
            # Let's stick to vector principles where possible.
            # We can rely on just tracking the flow for now.
            pass
        else:
            # Should not happen if Step 1 ran
            pass

        # --- PART B: INTERBANK MARKET & SRT ---
        
        # 1. Identify Liquidity Gaps
        # Bank Liquidity
        bank_liquidity = self.banks_state[:, self.IDX_BANK_LIQUIDITY]
        
        # Gap = Needs - Current Cash
        gaps = bank_inflows_demand - bank_liquidity
        
        deficit_mask = gaps > 0
        surplus_mask = gaps < 0 # (Means -Gap is surplus)
        
        deficit_ids = np.where(deficit_mask)[0]
        surplus_ids = np.where(surplus_mask)[0]
        
        if len(deficit_ids) > 0 and len(surplus_ids) > 0:
            
            # Prepare Batch for DebtRank
            # We want to test matching every Deficit Bank (D) with every Surplus Bank (S).
            # But realistically, D needs to find *one* lender.
            # We need to compute SRT for all D-S pairs.
            
            # Hypothetical Pairs
            # Create grids
            # D_grid, S_grid = np.meshgrid(deficit_ids, surplus_ids, indexing='ij')
            # Pairs: (D, S). Total K = N_Def * N_Sur.
            
            pairs = []
            for d in deficit_ids:
                for s in surplus_ids:
                    pairs.append((d, s))
            
            K = len(pairs)
            
            if K > 0:
                # Construct L_batch
                # Copy current L_bb K times.
                # Shape (K, B, B)
                L_batch = np.tile(self.L_bb, (K, 1, 1))
                
                # Apply hypothetical loans
                # L[k, row=Borrower(d), col=Lender(s)] += Amount
                # Amount? Usually the needed amount (gap[d]), capped by surplus[s].
                amounts = np.zeros(K)
                
                for k, (d, s) in enumerate(pairs):
                    amount = min(gaps[d], -gaps[s]) # - because surplus is negative gap
                    L_batch[k, d, s] += amount
                    amounts[k] = amount
                
                # Equity for DR (Assumed constant for the moment of transaction? Or adjusted?)
                # Use current equity
                equity_vec = self.banks_state[:, self.IDX_BANK_EQUITY]
                equity_batch = np.tile(equity_vec, (K, 1))
                
                # --- CORE: VECTORIZED DEBTRANK ---
                # Calculate DR for all hypothetical networks
                # Returns (K, B)
                batch_dr = fn.calculate_debtrank(L_batch, equity_batch)
                
                # Compute Expected Systemic Loss (Systemic Risk)
                # Need p_default. Let's assume uniform small p or leverage dependent.
                p_def = np.full((K, Params.B), 0.05) # 5% default prob placeholder
                
                # Total System Equity as Value
                V_total = np.sum(equity_vec)
                
                # EL_syst for each scenario k
                # Shape (K,)
                EL_syst = fn.compute_expected_systemic_loss(batch_dr, p_def, V_total)
                
                # Marginal Contribution (SRT)
                # Delta = EL_syst(new) - EL_syst(current/old)
                # Calc baseline DR
                baseline_dr = fn.calculate_debtrank(self.L_bb, equity_vec)
                base_EL = fn.compute_expected_systemic_loss(baseline_dr, p_def[0], V_total)
                
                marginal_risk = EL_syst - base_EL
                marginal_risk = np.maximum(marginal_risk, 0)
                
                tax_costs = marginal_risk * Params.TAX_SRT_ZETA
                
                # Interest Costs
                # Rate * Amount.
                # Simple rate r_bar.
                interest_costs = amounts * Params.r_bar
                
                total_costs = interest_costs + tax_costs
                
                # --- MATCHING DECISION ---
                # Simple Greedy Optimization:
                # Sort pairs by lowest Total Cost per unit of loan? 
                # Or for each D, find best S.
                
                # Reshape costs to (N_Def, N_Sur) to find best S for each D
                cost_matrix = total_costs.reshape(len(deficit_ids), len(surplus_ids))
                
                # Optimal S for each D
                best_s_indices = np.argmin(cost_matrix, axis=1) # Indices into surplus_ids
                
                # Execute Transactions
                # Note: This ignores competition (if multiple D want same S and S runs out).
                # For Phase 3 demo, we execute simply.
                
                for i, d_idx in enumerate(deficit_ids):
                    s_idx = surplus_ids[best_s_indices[i]]
                    
                    # Check if S still has funds? (Skipped for Vector demo simplicity)
                    
                    amount = min(gaps[d_idx], -gaps[s_idx])
                    
                    # Update Real L_bb
                    self.L_bb[d_idx, s_idx] += amount
                    
                    # Accounting
                    # D gets Cash (+), S loses Cash (-)
                    self.banks_state[d_idx, self.IDX_BANK_LIQUIDITY] += amount
                    self.banks_state[s_idx, self.IDX_BANK_LIQUIDITY] -= amount
                    
                    # Record Tax Paid? (Reduce Equity of borrower appropriately?)
                    # self.banks_state[d_idx, self.IDX_BANK_EQUITY] -= tax[pair]...
        
        # Finally, update Firm-Bank L_fb based on credit granted
        # For simplicity, assuming all credit demands were met by the banks (via interbank or own funds)
        # We assume banks deliver the cash to firms.
        if hasattr(self, 'current_credit_demand'):
            # Update L_fb manually or via loop
            for f in range(Params.F):
                amt = self.current_credit_demand[f]
                if amt > 0:
                    bank = chosen_bank_ids[f]
                    self.L_fb[f, bank] += amt
                    # Firm gets cash
                    self.firms_state[f, self.IDX_FIRM_LIQUIDITY] += amt
                    # Bank loses cash (sent to firm)
                    self.banks_state[bank, self.IDX_BANK_LIQUIDITY] -= amt

    def run_step(self):
        """Execute one simulation step."""
        # 1. Firms Planning
        self.step_firms_planning()
        
        # 2. Banking Market
        self.step_banking_market()
        
