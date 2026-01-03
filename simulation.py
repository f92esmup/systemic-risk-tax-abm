import numpy as np
from parameters import Params

class CRISIS_Model:
    """
    Vectorized implementation of the Poledna & Thurner (2016) ABM model.
    Uses tensor-based state management to avoid explicit agent loops.
    """

    def __init__(self, seed=None, tax_mode='none', tax_param=0.0):
        self.rng = np.random.default_rng(seed)
        
        # Experiment Logic
        self.tax_mode = tax_mode.lower()
        self.tax_param = tax_param
        
        # Metrics for Analysis
        self.current_step_loss = 0.0 # Capital destroyed in cascades this step
        self.current_step_defaults = 0 # Number of bank defaults this step
        self.current_step_volume = 0.0 # Interbank volume this step
        
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
        # Initialize Bank Assets
        init_bank_assets = self.rng.uniform(Params.INIT_BANK_ASSETS[0], Params.INIT_BANK_ASSETS[1], size=B)
        # Equity = Assets * Capital Ratio
        self.banks_state[:, self.IDX_BANK_EQUITY] = init_bank_assets * Params.INIT_CAPITAL_RATIO
        # Liquidity = Assets (Assuming start with all liquid assets, no loans yet)
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] = init_bank_assets
        # Deposits = Assets - Equity
        self.banks_state[:, self.IDX_BANK_DEPOSITS] = init_bank_assets - self.banks_state[:, self.IDX_BANK_EQUITY]

        # Initialize Firms
        init_firm_assets = self.rng.uniform(Params.INIT_FIRM_ASSETS[0], Params.INIT_FIRM_ASSETS[1], size=F)
        self.firms_state[:, self.IDX_FIRM_EQUITY] = init_firm_assets
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] = init_firm_assets
        
        # Initialize Price to Marginal Cost (Breakeven)
        # MC = Wage / Alpha
        init_price = Params.WAGE / Params.alpha
        self.firms_state[:, self.IDX_FIRM_PRICE] = init_price
        self.firms_state[:, self.IDX_FIRM_PRICE_PREV] = init_price


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
        bank_liquidity = self.banks_state[:, self.IDX_BANK_LIQUIDITY]
        gaps = bank_inflows_demand - bank_liquidity
        
        deficit_ids = np.where(gaps > 0)[0]
        surplus_ids = np.where(gaps < 0)[0]
        
        if len(deficit_ids) > 0 and len(surplus_ids) > 0:
            
            # Generate all pairs (Deficit, Surplus)
            pairs_d, pairs_s = [], []
            for d in deficit_ids:
                for s in surplus_ids:
                    pairs_d.append(d)
                    pairs_s.append(s)
            
            pairs_d = np.array(pairs_d)
            pairs_s = np.array(pairs_s)
            
            if len(pairs_d) > 0:
                # Proposed Amounts: Min(Need, Available Surplus)
                amounts = np.minimum(gaps[pairs_d], -gaps[pairs_s])
                
                # Prepare args for compute_srt_tax
                proposed_indices = np.column_stack((pairs_d, pairs_s))
                
                # Parameters for function
                C = self.banks_state[:, self.IDX_BANK_EQUITY]
                # v = Total Assets (Liquidity + BankLoans + FirmLoans)
                # For now, approximate v as Equity or Assets?
                # User prompt mentioned: "v: Economic Value (Total Assets/Liabilities)"
                # Let's sum components we have.
                # Assets = Liquidity + Loans to Firms (L_fb sum axis 1 implicitly?) + Loans to Banks.
                # L_fb is (F,B) so LoansToFirms = Sum over F of L_fb[:, b].
                loans_to_firms = np.sum(self.L_fb, axis=0) # (B,)
                loans_to_banks = np.sum(self.L_bb, axis=1) # (B,) L_bb rows=borrower? No, rows=borrower means i owes j. 
                # So Assets of j = sum_i L_ij. (Col sum).
                # Wait. "Rows=Borrowers, Cols=Lenders."
                # So L_ij is amount i owes j. j is the lender (Asset Holder).
                # So Assets of Bank j = sum(L_ij over i).
                interbank_assets = np.sum(self.L_bb, axis=0)
                
                total_assets = bank_liquidity + loans_to_firms + interbank_assets
                v = total_assets
                
                # p_default: Simplified constant or function of Leverage
                p_def = np.full(Params.B, 0.05) 
                
                # TAX CALCULATION BASED ON MODE
                taxes = np.zeros(len(amounts))
                
                if self.tax_mode == 'srt':
                    # CALL SRT FUNCTION
                    # zeta = self.tax_param (e.g. 1.0)
                    taxes = fn.compute_srt_tax(
                        L_current=self.L_bb,
                        proposed_loans_indices=proposed_indices,
                        proposed_amounts=amounts,
                        C=C,
                        v=v,
                        p_default=p_def,
                        zeta=self.tax_param
                    )
                elif self.tax_mode == 'tobin':
                    # Flat rate tax
                    # rate = self.tax_param (e.g. 0.002)
                    taxes = amounts * self.tax_param
                else:
                    # 'none' or other -> 0 tax
                    pass
                
                # Interest Costs
                interest = amounts * Params.r_bar
                total_costs = interest + taxes
                
                # --- MATCHING DECISION ---
                # For each Deficit bank d, find S that minimizes total_cost[pair]
                # We need to reshape or iterate.
                
                # Unique deficits
                unique_d = np.unique(pairs_d)
                
                for d in unique_d:
                    # Indices in the pairs lists belonging to this d
                    mask = (pairs_d == d)
                    candidates_s = pairs_s[mask]
                    candidates_costs = total_costs[mask]
                    candidates_amounts = amounts[mask]
                    
                    if len(candidates_s) == 0: continue
                    
                    # Best match
                    best_idx_local = np.argmin(candidates_costs)
                    s_best = candidates_s[best_idx_local]
                    amount_best = candidates_amounts[best_idx_local]
                    
                    # Execute Transaction (Simplified: One per Deficit bank)
                    # Check consistency? (e.g. if S runs out of funds).
                    # We skip strict double-counting check for this phase.
                    
                    # Update L_bb
                    self.L_bb[d, s_best] += amount_best
                    
                    # Track Volume
                    self.current_step_volume += amount_best
                    
                    # Transfers
                    self.banks_state[d, self.IDX_BANK_LIQUIDITY] += amount_best
                    self.banks_state[s_best, self.IDX_BANK_LIQUIDITY] -= amount_best
                    
                    # Tax Payment? 
                    # If tax > 0, does the bank pay it to external sink?
                    # "Update Bank Equity (Deduct Tax)"
                    tax_val = taxes[mask][best_idx_local]
                    if tax_val > 0:
                        self.banks_state[d, self.IDX_BANK_EQUITY] -= tax_val

        # Finally, update Firm-Bank L_fb based on credit granted
        if hasattr(self, 'current_credit_demand'):
             for f in range(Params.F):
                amt = self.current_credit_demand[f]
                if amt > 0:
                    bank = chosen_bank_ids[f]
                    self.L_fb[f, bank] += amt
                    self.firms_state[f, self.IDX_FIRM_LIQUIDITY] += amt
                    self.banks_state[bank, self.IDX_BANK_LIQUIDITY] -= amt

    def step_real_economy(self):
        """
        Phase 4: Real Economy (Production, Wages, Consumption).
        
        A. Firms convert Labor -> Goods (Production) and pay Wages (Liquidity -> Households).
        B. Households consume Goods (Liquidity -> Firms).
        """
        # --- A. PRODUCTION & WAGES ---
        
        # 1. Production
        # Inventory = Workers * Alpha
        workers = self.firms_state[:, self.IDX_FIRM_WORKERS]
        new_production = workers * Params.alpha
        
        # Current logic: Are goods durable? Usually yes, inventory accumulates.
        # "Inventory_New = ..." imply adding to existing or just produced?
        # Let's assume accumulation.
        # Check if IDX_FIRM_PRODUCTION stores stock or flow. Ideally Stock.
        # "Producción" in state usually means current stock available for sale.
        self.firms_state[:, self.IDX_FIRM_PRODUCTION] += new_production
        
        # 2. Wage Payment
        # Wage Bill was calculated in Phase 1 (Planning) but let's recalc or use state
        # We stored it in IDX_FIRM_WAGES_PAID? 
        # Actually in `step_firms_planning`, we set `self.firms_state[:, self.IDX_FIRM_WAGES_PAID] = wage_bill`
        # as a liability record. Now we pay it.
        
        wage_bills = self.firms_state[:, self.IDX_FIRM_WAGES_PAID]
        firm_liquidity = self.firms_state[:, self.IDX_FIRM_LIQUIDITY]
        
        # Can they pay?
        # Ideally yes, if they got credit.
        # Actual Payment = Min(Liquidity, Bill). If less, workers get partial wages (or firm goes bankrupt/defaults later).
        # For this step, we assume they pay what they can.
        payments = np.minimum(wage_bills, firm_liquidity)
        
        # Deduct from Firms
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] -= payments
        
        # Add to Households
        # Vectorized transfer: Use the Employer Map
        # self.hh_employer_idx (H,) -> Index of firm
        # We need to know how much EACH household gets.
        # Assumption: All households working for Firm F get equal share of that firm's wage bill.
        # Workers count per firm:
        # We calculated `workers` vector (float).
        # But we have discrete households.
        # For the mapping to work, the number of households assigned to Firm F must match `workers`?
        # In this simple model (Table I parameters), H=1300, F=100. Avg 13 workers/firm.
        # `workers` computed in Phase 2 was "Labor Demand".
        # Does the firm HIRE/FIRE?
        # If Labor Demand != Current Employees, we need a Labor Market step (Matching).
        # Phase 2 calculated "Required Workers".
        # Phase 4 "Production" assumes they have them.
        # To simplify without a complex labor market matching:
        # We assume the `hh_employer_idx` is static or we just pay the demand distributed to *current* employees.
        # Or, logically, the firm pays `wage_bills` to the set of households `where hh_employer_idx == f`.
        # Let's calculate per-household wage for each firm.
        
        # Count actual employees per firm (Static topology for now)
        employee_counts = np.bincount(self.hh_employer_idx, minlength=Params.F)
        
        # Avoid division by zero
        wage_per_worker = np.zeros(Params.F)
        mask = employee_counts > 0
        np.divide(payments, employee_counts, out=wage_per_worker, where=mask)
        
        # Distribute to Households
        # Each HH gets wage_per_worker[their_employer]
        hh_income = wage_per_worker[self.hh_employer_idx]
        
        self.households_state[:, self.IDX_HH_DEPOSITS] += hh_income
        
        
        # --- B. CONSUMPTION MARKET ---
        
        # 1. Budget
        # B_h = Deposits * c
        hh_deposits = self.households_state[:, self.IDX_HH_DEPOSITS]
        budgets = hh_deposits * Params.c
        
        # 2. Firm Selection (Z-Search)
        # Sample Z firms per household
        # Shape (H, Z)
        z_indices = self.rng.integers(0, Params.F, size=(Params.H, Params.Z_CONSUMPTION))
        
        # Get Prices: (H, Z)
        prices_options = self.firms_state[z_indices, self.IDX_FIRM_PRICE]
        
        # Select min price
        winner_local_indices = np.argmin(prices_options, axis=1) # (H,) 0 or 1
        
        # Map back to global Firm Index
        # winner_global_idx[h] = z_indices[h, winner_local_indices[h]]
        # Advanced indexing
        winner_global_indices = z_indices[np.arange(Params.H), winner_local_indices]
        
        # 3. Aggregate Demand
        # Sum budgets destined for each firm
        demand_monetary = np.bincount(winner_global_indices, weights=budgets, minlength=Params.F)
        
        # 4. Sales & Rationing
        firm_prices = self.firms_state[:, self.IDX_FIRM_PRICE]
        firm_inventory = self.firms_state[:, self.IDX_FIRM_PRODUCTION]
        
        max_revenue = firm_inventory * firm_prices
        
        # Actual Revenue = Min(Demand, Max_Revenue)
        actual_revenue = np.minimum(demand_monetary, max_revenue)
        
        # Sales Quantity = Revenue / Price (Safe division)
        sales_qty = np.zeros(Params.F)
        price_mask = firm_prices > 0
        np.divide(actual_revenue, firm_prices, out=sales_qty, where=price_mask)
        
        # Rationing Magnitude (Scale Factor for Households)
        # If Demand > Max_Revenue, households spent less than `budgets`.
        # scale[f] = Actual_Rev / Demand[f]. 1.0 if full demand met.
        scale_factors = np.ones(Params.F)
        demand_mask = demand_monetary > 1e-9
        np.divide(actual_revenue, demand_monetary, out=scale_factors, where=demand_mask)
        
        # 5. Execution
        
        # Firms: Receive Money, Lose Inventory
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] += actual_revenue
        self.firms_state[:, self.IDX_FIRM_PRODUCTION] -= sales_qty
        
        # Households: Pay Money
        # Each household paid: Budget * Scale_Factor[Chosen_Firm]
        hh_scale = scale_factors[winner_global_indices]
        hh_expenditure = budgets * hh_scale
        
        self.households_state[:, self.IDX_HH_DEPOSITS] -= hh_expenditure

    def step_accounting(self):
        """
        Phase 5: Accounting, Bankruptcy, and Resets.
        
        A. Debt Repayment (Principal + Interest)
        B. Dividends
        C. Bankruptcies (Firms) & Defaults (Banks) with Contagion
        D. Variable Updates (Shift t -> t+1)
        """
        # Reset Step Metrics
        self.current_step_loss = 0.0
        self.current_step_defaults = 0
        # Reset Step Metrics
        self.current_step_loss = 0.0
        self.current_step_defaults = 0
        # self.current_step_volume = 0.0 # Removed to prevent overwriting banking market data
        
        
        # --- A. DEBT REPAYMENT ---
        # 1. Firms -> Banks (L_fb)
        # Payment = L_fb * DEBT_REPAYMENT_RATE
        repayment_firms = self.L_fb * Params.DEBT_REPAYMENT_RATE
        
        # Check if firms have liquidity? 
        # Ideally yes. If not, they pay what they have or default LATER in this function.
        # Simplification: They pay, potentially going negative on Liquidity (which triggers bankruptcy below).
        # OR: They assume partial payment. 
        # The prompt says: "Update: Firm_Liquidity -= sum(Payment, axis=1)". 
        # This implies standard accounting first, then check status.
        
        total_payment_by_firm = np.sum(repayment_firms, axis=1)
        total_receipt_by_bank = np.sum(repayment_firms, axis=0)
        
        self.L_fb -= repayment_firms
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] -= total_payment_by_firm
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] += total_receipt_by_bank
        
        # 2. Banks -> Banks (L_bb)
        # Payment = L_bb * DEBT_REPAYMENT_RATE
        repayment_interbank = self.L_bb * Params.DEBT_REPAYMENT_RATE
        
        total_payment_by_bank = np.sum(repayment_interbank, axis=1) # Row Sum = What I owe (Borrower)
        total_receipt_by_bank_ib = np.sum(repayment_interbank, axis=0) # Col Sum = What I receive (Lender)
        
        self.L_bb -= repayment_interbank
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] -= total_payment_by_bank
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] += total_receipt_by_bank_ib
        
        
        # --- B. DIVIDENDS ---
        # 1. Firms
        # Profit? Let's use Excess Liquidity threshold or just Positive Liquidity.
        # "Calculate Profits (Change in Equity or Proxy). Use max(0, Liquidity_Surplus)."
        # Let's use current positive liquidity as a proxy for distributable funds.
        firm_liq = self.firms_state[:, self.IDX_FIRM_LIQUIDITY]
        distributable_f = np.maximum(0, firm_liq) 
        # But paying ALL positive liquidity stops growth?
        # Maybe change in equity? For now, stick to prompt instruction: "max(0, Liquidity_Surplus)"
        # Assuming surplus above some operational need. Let's say above 0.
        
        dividends_f = distributable_f * Params.DIVIDEND_RATIO
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] -= dividends_f
        
        # Pay to Households
        # Using self.hh_employer_idx 
        # We need to distribute dividend_f[i] to all HH employed by i.
        # Count employees
        counts = np.bincount(self.hh_employer_idx, minlength=Params.F)
        per_capita_div = np.zeros(Params.F)
        mask_c = counts > 0
        np.divide(dividends_f, counts, out=per_capita_div, where=mask_c)
        
        # Add to HH
        self.households_state[:, self.IDX_HH_DEPOSITS] += per_capita_div[self.hh_employer_idx]
        
        # 2. Banks
        # Bank Equity is the main profit metric? Or Liquidity?
        # Usually Banks pay from Earnings.
        # Let's use Bank Equity for calculation basis but pay from Liquidity.
        bank_equity = self.banks_state[:, self.IDX_BANK_EQUITY]
        # Distributable only if Equity > Capital Target? (phi)
        # Simplification: Positive Equity.
        distributable_b = np.maximum(0, bank_equity)
        dividends_b = distributable_b * Params.DIVIDEND_RATIO
        
        # Check Liquidity constraint
        dividends_b = np.minimum(dividends_b, self.banks_state[:, self.IDX_BANK_LIQUIDITY])
        dividends_b = np.maximum(0, dividends_b)
        
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] -= dividends_b
        # Equity also drops when dividends paid
        self.banks_state[:, self.IDX_BANK_EQUITY] -= dividends_b
        
        # Pay to Households (Owners)
        # Using self.hh_bank_idx (Assuming this is "Primary Bank of Deposit" OR "Ownership").
        # Usually checking accounts != Stock ownership. 
        # But for model simplicity (closed system), let's assume HHs own their bank.
        counts_b = np.bincount(self.hh_bank_idx, minlength=Params.B)
        per_capita_div_b = np.zeros(Params.B)
        mask_cb = counts_b > 0
        np.divide(dividends_b, counts_b, out=per_capita_div_b, where=mask_cb)
        
        self.households_state[:, self.IDX_HH_DEPOSITS] += per_capita_div_b[self.hh_bank_idx]
        
        
        # --- C. BANKRUPTCIES & CASCADES ---
        
        # 1. Firm Bankruptcy
        # Condition: Liquidity < 0
        dead_firms_mask = self.firms_state[:, self.IDX_FIRM_LIQUIDITY] < 0
        dead_firms_indices = np.where(dead_firms_mask)[0]
        
        if len(dead_firms_indices) > 0:
            # Impact on Banks: Write-off loans
            # L_fb columns are banks. Rows are firms.
            # Sum of bad loans per bank
            bad_loans_per_bank = np.sum(self.L_fb[dead_firms_indices, :], axis=0) # (B,)
            
            # Reduce Bank Equity
            self.banks_state[:, self.IDX_BANK_EQUITY] -= bad_loans_per_bank
            # Reduce Bank Assignable Assets/Liquidity? No, loan was asset. Equity absorbs loss.
            
            # Reset Firms
            # Clean slate: New liquidity, new price, 0 debt
            # Reset Liquidity
            init_liq = self.rng.uniform(Params.INIT_FIRM_ASSETS[0], Params.INIT_FIRM_ASSETS[1], size=len(dead_firms_indices))
            self.firms_state[dead_firms_indices, self.IDX_FIRM_LIQUIDITY] = init_liq
            # Reset Price
            self.firms_state[dead_firms_indices, self.IDX_FIRM_PRICE] = 1.0
            # Reset Debt
            self.L_fb[dead_firms_indices, :] = 0.0
            
        
        # 2. Bank Default Cascade
        # Condition: Equity < 0
        
        dead_banks_prev = np.zeros(Params.B, dtype=bool)
        
        while True:
            # Current dead banks
            dead_banks_curr = self.banks_state[:, self.IDX_BANK_EQUITY] < 0
            
            # New defaults in this iteration
            # (Curr AND NOT Prev)
            new_defaults = dead_banks_curr & (~dead_banks_prev)
            new_default_indices = np.where(new_defaults)[0]
            
            count_new = len(new_default_indices)
            if count_new == 0:
                break # Convergence
            
            # Record Metrics
            self.current_step_defaults += count_new
                
            # Impact: Lenders lose assets
            losses = np.sum(self.L_bb[new_default_indices, :], axis=0)
            
            # Record Financial Loss (from Interbank defaults)
            # Actually, the loss occurs when we write off.
            # Total System Equity destruction = sum of negative equity? 
            # Or sum of loans defaulted?
            # Paper defines Loss as "Total Assets of defaulted banks"? Or "Losses absorbed by others"?
            # Let's track: Sum of L_bb defaulted. This is the shock size.
            loan_losses_value = np.sum(self.L_bb[new_default_indices, :])
            self.current_step_loss += loan_losses_value
            
            # Also Firms loans defaulting? 
            # If Bank A defaults, its loans to firms are assets of A. What happens to them?
            # In simple models, they are liquidated or lost. 
            # But the Loss metric usually refers to contagion losses.
            # Let's stick to interbank credit losses for "Cascades".
            
            # Update Equity of SURVIVING banks (actually all, doesn't matter for dead ones)
            self.banks_state[:, self.IDX_BANK_EQUITY] -= losses
            
            # Write off the debt (Asset gone)
            self.L_bb[new_default_indices, :] = 0.0
            
            # Update history mask
            dead_banks_prev = dead_banks_curr
            
        # Reset Dead Banks (Recapitalization / Bailout / Replacement)
        # To maintain N banks.
        final_dead_banks = np.where(dead_banks_prev)[0]
        if len(final_dead_banks) > 0:
            init_assets = self.rng.uniform(Params.INIT_BANK_ASSETS[0], Params.INIT_BANK_ASSETS[1], size=len(final_dead_banks))
            # Reset Equity
            # Assets = Equity + Liabilities. Assume 0 Liabilities initially.
            self.banks_state[final_dead_banks, self.IDX_BANK_EQUITY] = init_assets * Params.INIT_CAPITAL_RATIO # using Initial Capital Ratio
            self.banks_state[final_dead_banks, self.IDX_BANK_LIQUIDITY] = init_assets # Cash
            
            # Clear all connections involving dead banks
            # They already had their Borrowing rows cleared.
            # Need to clear their Lending columns (Columns j where j is dead)
            # Because a new bank starts with 0 loans granted.
            self.L_bb[:, final_dead_banks] = 0.0
            self.L_fb[:, final_dead_banks] = 0.0 # Clean firm loans too
        
        # --- D. VARIABLE UPDATES ---
        pass

    def run_step(self):
        """Execute one simulation step."""
        # Reset Per-Step Accumulators
        self.current_step_volume = 0.0
        
        # 1. Firms Planning
        self.step_firms_planning()
        
        # 2. Banking Market
        self.step_banking_market()
        
        # 3. Real Economy
        self.step_real_economy()
        
        # 4. Accounting & Reset
        self.step_accounting()
        
        # 5. Traceability
        self.record_history()
        
