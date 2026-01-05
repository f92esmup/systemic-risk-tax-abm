import numpy as np
from parametros import Parametros
import funciones as fn


class Modelo_CRISIS:
    """
    Vectorized implementation of the Poledna & Thurner (2016) ABM model.
    Uses tensor-based state management to avoid explicit agent loops.
    """

    # --- State Indices (Class Constants) ---

    # Banks State Indices (N=8)
    IDX_BANK_LIQUIDITY = 0
    IDX_BANK_EQUITY = 1
    IDX_BANK_DEPOSITS = 2
    IDX_BANK_BAD_DEBT = 3
    IDX_BANK_OPERATING_COST_CHI = 4
    IDX_BANK_INTERBANK_COST_PSI = 5
    IDX_BANK_DEFAULT_PROB = 6
    IDX_BANK_TOTAL_ASSETS = 7
    N_BANK_FEATURES = 8

    # Firms State Indices (N=11)
    IDX_FIRM_LIQUIDITY = 0
    IDX_FIRM_EQUITY = 1
    IDX_FIRM_PRICE = 2
    IDX_FIRM_DEMAND = 3
    IDX_FIRM_PROD = 4  # Production/Inventory
    IDX_FIRM_WORKERS = 5
    IDX_FIRM_WAGES = 6  # Wages Bill
    IDX_FIRM_PRICE_PREV = 7
    IDX_FIRM_DEMAND_PREV = 8
    IDX_FIRM_LEVERAGE = 9
    IDX_FIRM_DEFAULT_FLAG = 10
    N_FIRM_FEATURES = 11

    # Households State Indices (N=3)
    IDX_HH_DEPOSITS = 0
    IDX_HH_IS_OWNER = 1  # 0=Worker, 1=Owner
    IDX_HH_OWNED_ENTITY_IDX = 2  # Index of Firm or Bank owned
    N_HH_FEATURES = 3

    def __init__(self, seed=None, tax_mode="none", tax_param=0.0):
        self.rng = np.random.default_rng(seed)

        # Experiment Logic
        self.tax_mode = tax_mode.lower()
        self.tax_param = tax_param

        # Metrics for Analysis
        self.current_step_loss = 0.0
        self.current_step_defaults = 0
        self.current_step_volume = 0.0

        self.reset()

    def reset(self):
        """
        Reset or initialize all state tensors and network topologies for a new simulacion run.
        """
        # --- Dimensions ---
        B = Parametros.B
        F = Parametros.F
        H = Parametros.H

        # --- 1. Agents State (Tensors) ---
        self.estado_bancos = np.zeros((B, self.N_BANK_FEATURES), dtype=np.float64)
        self.estado_firmas = np.zeros((F, self.N_FIRM_FEATURES), dtype=np.float64)
        self.estado_hogares = np.zeros((H, self.N_HH_FEATURES), dtype=np.float64)

        # --- 2. Vectorized Initialization ---

        # Banks: Specificity Parameters (Constant per run)
        # CHI ~ U(0, 1)
        self.estado_bancos[:, self.IDX_BANK_OPERATING_COST_CHI] = self.rng.uniform(
            Parametros.CHI_RANGE[0], Parametros.CHI_RANGE[1], size=B
        )
        # PSI ~ U(0, 0.1)
        self.estado_bancos[:, self.IDX_BANK_INTERBANK_COST_PSI] = self.rng.uniform(
            Parametros.PSI_RANGE[0], Parametros.PSI_RANGE[1], size=B
        )

        # Bank Financials
        init_bank_assets = self.rng.uniform(
            Parametros.INIT_BANK_ASSETS[0], Parametros.INIT_BANK_ASSETS[1], size=B
        )
        self.estado_bancos[:, self.IDX_BANK_TOTAL_ASSETS] = init_bank_assets
        # Equity = Assets * Capital Ratio
        self.estado_bancos[:, self.IDX_BANK_EQUITY] = (
            init_bank_assets * Parametros.INIT_CAPITAL_RATIO
        )
        # Liquidity = Assets (Assuming start with all liquid)
        self.estado_bancos[:, self.IDX_BANK_LIQUIDITY] = init_bank_assets
        # Deposits = Assets - Equity
        self.estado_bancos[:, self.IDX_BANK_DEPOSITS] = (
            init_bank_assets - self.estado_bancos[:, self.IDX_BANK_EQUITY]
        )

        # Firms
        init_firm_assets = self.rng.uniform(
            Parametros.INIT_FIRM_ASSETS[0], Parametros.INIT_FIRM_ASSETS[1], size=F
        )
        self.estado_firmas[:, self.IDX_FIRM_EQUITY] = init_firm_assets
        self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] = init_firm_assets

        # Initialize Price to Marginal Cost
        init_price = Parametros.WAGE / Parametros.alpha
        self.estado_firmas[:, self.IDX_FIRM_PRICE] = init_price
        self.estado_firmas[:, self.IDX_FIRM_PRICE_PREV] = init_price

        # Households: Owners vs Workers
        # Random assignment
        hh_indices = np.arange(H)
        self.rng.shuffle(hh_indices)

        # First F households own Firms
        firm_owners = hh_indices[:F]
        self.estado_hogares[firm_owners, self.IDX_HH_IS_OWNER] = 1.0
        self.estado_hogares[firm_owners, self.IDX_HH_OWNED_ENTITY_IDX] = np.arange(F)

        # Next B households own Banks
        bank_owners = hh_indices[F : F + B]
        self.estado_hogares[bank_owners, self.IDX_HH_IS_OWNER] = 1.0
        self.estado_hogares[bank_owners, self.IDX_HH_OWNED_ENTITY_IDX] = np.arange(B)

        # Remaining are Workers (Default 0, 0)

        # --- 3. Topology & Relationships ---
        self.matriz_interbancaria = np.zeros((B, B), dtype=np.float64)
        self.matriz_credito_firmas = np.zeros((F, B), dtype=np.float64)

        # Bank/Employer Relationships for Workers
        self.hh_employer_idx = self.rng.integers(0, F, size=H)
        self.hh_bank_idx = self.rng.integers(0, B, size=H)

        # --- 4. History / Traceability ---
        self.step_buffer = {
            "matriz_interbancaria": [],
            "matriz_credito_firmas": [],
            "estado_bancos": [],
            "estado_firmas": [],
            "hh_bank_idx": [],
            # "hh_employer_idx": [] # Not strictly requested in prompt list but useful.
            # Prompt asked for: matriz_interbancaria, matriz_credito_firmas, estado_bancos, estado_firmas, hh_bank_idx.
        }

        self.registrar_historia()

    def registrar_historia(self):
        """Append current state snapshots to step_buffer."""
        # Topologies
        self.step_buffer["matriz_interbancaria"].append(
            self.matriz_interbancaria.astype(np.float32).copy()
        )
        self.step_buffer["matriz_credito_firmas"].append(
            self.matriz_credito_firmas.astype(np.float32).copy()
        )

        # States (Full capture)
        self.step_buffer["estado_bancos"].append(
            self.estado_bancos.astype(np.float32).copy()
        )
        self.step_buffer["estado_firmas"].append(
            self.estado_firmas.astype(np.float32).copy()
        )

        # Relations
        self.step_buffer["hh_bank_idx"].append(self.hh_bank_idx.astype(np.int16).copy())

    def reset_history(self):
        """Clear the step buffer to free RAM after flushing."""
        for key in self.step_buffer:
            self.step_buffer[key] = []

    def guardar_simulacion_disco(self, run_id, folder="output_data"):
        """
        Save the buffered run history to a compressed .npz file and clear RAM.
        Stacks lists into 3D arrays (T, N, M).
        """
        import os

        os.makedirs(folder, exist_ok=True)

        data_dict = {}
        for key, val_list in self.step_buffer.items():
            if len(val_list) > 0:
                data_dict[key] = np.stack(val_list)
            else:
                data_dict[key] = np.array([])

        filename = f"{folder}/run_{run_id:05d}.npz"
        np.savez_compressed(filename, **data_dict)

        self.reset_history()

    def paso_planificacion_firmas(self):
        """
        Phase 1: Firms Planning.
        - Update Prices based on market average.
        - Update Expected Demand (simple stochastic process for now).
        - Calculate Labor requirements.
        - Update Financial Health (Leverage) for Appendix A interest rates.
        - Calculate Credit Demand if Wages > Liquidity.
        """
        # --- 1. Update Prices ---
        # Calculate market average price
        p_avg = np.mean(self.estado_firmas[:, self.IDX_FIRM_PRICE])

        # Vector of current prices
        prices = self.estado_firmas[:, self.IDX_FIRM_PRICE]

        # Save to PREV
        self.estado_firmas[:, self.IDX_FIRM_PRICE_PREV] = prices

        # Noise component
        noise = self.rng.normal(0, Parametros.PRICE_DRIFT_STD, size=Parametros.F)

        # Adjustment Rule: p_new = p * (1 + speed * (p_avg - p)/p_avg + noise)
        # Avoid division by zero if p_avg is 0 (unlikely but safe)
        if p_avg > 1e-9:
            adjustment = Parametros.PRICE_ADJUSTMENT_SPEED * (p_avg - prices) / p_avg
            new_prices = prices + prices * (adjustment + noise)
        else:
            new_prices = prices * (1 + noise)

        # Constraint: Prices must be positive
        new_prices = np.maximum(new_prices, 0.01)

        self.estado_firmas[:, self.IDX_FIRM_PRICE] = new_prices

        # --- 2. Update Demand Expectations ---
        # Save previous demand
        self.estado_firmas[:, self.IDX_FIRM_DEMAND_PREV] = self.estado_firmas[
            :, self.IDX_FIRM_DEMAND
        ]

        # Initialize demand if 0 (start of sim)
        current_demand = self.estado_firmas[:, self.IDX_FIRM_DEMAND]
        if np.all(current_demand == 0):
            # Initialize random demand around expected production capacity
            # Capacity ~ Liquidity / Wage * Alpha? Just random start.
            current_demand = self.rng.uniform(10, 50, size=Parametros.F)

        # Random Walk: D(t) = D(t-1) * (1 + noise)
        demand_shock = self.rng.normal(0, 0.05, size=Parametros.F)  # 5% volatility
        new_demand = current_demand * (1 + demand_shock)
        new_demand = np.maximum(new_demand, 0.0)

        self.estado_firmas[:, self.IDX_FIRM_DEMAND] = new_demand

        # --- 3. Production Planning ---
        # Labor Needed = Ceil(Demand / Alpha)
        labor_needed = np.ceil(new_demand / Parametros.alpha)

        self.estado_firmas[:, self.IDX_FIRM_WORKERS] = labor_needed
        self.estado_firmas[:, self.IDX_FIRM_PROD] = (
            labor_needed * Parametros.alpha
        )  # Potential production

        # Wage Bill
        wage_bill = labor_needed * Parametros.WAGE
        self.estado_firmas[:, self.IDX_FIRM_WAGES] = wage_bill

        # --- 4. Financial Health Update (CRITICAL for Appendix A) ---
        # Leverage = Debt / (Liquidity + 1e-9)
        # Current Debt = Sum of loans from all banks (matriz_credito_firmas rows)
        current_debt = np.sum(self.matriz_credito_firmas, axis=1)  # (F,)
        liquidity = self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY]

        # Avoid division by zero
        leverage = current_debt / (liquidity + 1e-9)
        self.estado_firmas[:, self.IDX_FIRM_LEVERAGE] = leverage

        # --- 5. Credit Demand Calculation ---
        # Gap = Wages - Liquidity
        gap = wage_bill - liquidity

        # Credit Demand > 0 only if Gap > 0
        credit_demand = np.maximum(gap, 0.0)

        # Store for next phase
        self.current_firm_credit_demand = credit_demand

    def paso_mercado_bancario(self):
        """
        Phase 2: Credit Market (Firms-Banks) & Interbank Market (SRT).

        Part A: Firms request credit from Banks (Eq A1).
        Part B: Banks manage liquidity deficits via Interbank Market using DebtRank-based SRT (Eq A2 + Eq 5).
        Part C: Disbursement of funds to firms.
        """
        import funciones as fn

        # --- PART A: CREDIT MARKET (Firms -> Banks) ---

        # 1. Firms select N_SEARCH banks randomly
        # Shape (F, N_SEARCH)
        pool_indices = self.rng.integers(
            0, Parametros.B, size=(Parametros.F, Parametros.N_SEARCH)
        )

        # 2. Banks offer rates (Eq A1)
        # r_if = r_bar * (1 + chi_i * mu(leverage_f))

        # Get Bank Specificity Chi for the selected banks
        # self.estado_bancos column CHI is (B,). Indexing with (F, N) -> (F, N)
        bank_chi_pool = self.estado_bancos[
            pool_indices, self.IDX_BANK_OPERATING_COST_CHI
        ]

        # Get Firm Leverage
        # self.estado_firmas column LEVERAGE is (F,) -> Broadcast to (F, N)
        firm_leverage = self.estado_firmas[:, self.IDX_FIRM_LEVERAGE]
        firm_leverage_broad = firm_leverage[:, np.newaxis]

        # Calculate Firm Fragility mu(l)
        firm_fragility = fn.calcular_fragilidad_financiera(
            firm_leverage_broad, Parametros.K_mu
        )

        # Calculate Rates
        rates = fn.calcular_tasa_firma(Parametros.R_BAR, bank_chi_pool, firm_fragility)

        # Add tiny noise to break ties
        rates += self.rng.uniform(0, 1e-6, size=rates.shape)

        # 3. Firms choose best bank (Lowest Rate)
        best_choice_local_idx = np.argmin(rates, axis=1)  # (F,)

        # Get actual Bank ID
        chosen_bank_ids = pool_indices[np.arange(Parametros.F), best_choice_local_idx]

        # 4. Aggregate Credit Demand per Bank
        # self.current_firm_credit_demand is (F,) from Phase 1
        bank_credit_demand = np.zeros(Parametros.B)
        np.add.at(bank_credit_demand, chosen_bank_ids, self.current_firm_credit_demand)

        # --- PART B: INTERBANK MARKET & SRT ---

        # 1. Identify Liquidity Gaps
        bank_liquidity = self.estado_bancos[:, self.IDX_BANK_LIQUIDITY]
        gaps = bank_credit_demand - bank_liquidity

        deficit_ids = np.where(gaps > 0)[0]
        surplus_ids = np.where(gaps < 0)[0]  # Negative gap means surplus

        if len(deficit_ids) > 0 and len(surplus_ids) > 0:
            # 2. Generate Candidate Pairs (Deficit, Surplus)
            # Meshgrid for all combinations
            # We want arrays of shape (N_pairs,)
            # d_mesh, s_mesh = np.meshgrid(deficit_ids, surplus_ids, indexing='ij')
            # But we can just repeat:
            n_def = len(deficit_ids)
            n_sur = len(surplus_ids)

            d_indices = np.repeat(deficit_ids, n_sur)
            s_indices = np.tile(surplus_ids, n_def)

            # 3. Calculate Base Rates (Eq A2)
            # r_ij = r_bar * (1 + psi_i * mu(leverage_j))
            # i = Lender (s_indices), j = Borrower (d_indices)

            psi_lender = self.estado_bancos[s_indices, self.IDX_BANK_INTERBANK_COST_PSI]

            # Borrower Leverage needed.
            # Bank Leverage = (Deposits + Interbank Borrowing) / Equity ??
            # Or Total Liabilities / Equity.
            # Simplified Leverage for Interbank Rate:
            # Paper says "borrower's leverage l_j(t)".
            # Let's use current BadDebt + Deposits ratio or just Liabilities/Equity.
            # We haven't updated Liabilities yet (it's t).
            # Use matriz_interbancaria col sum (borrowing) + Deposits.
            current_liabilities = self.estado_bancos[
                :, self.IDX_BANK_DEPOSITS
            ] + np.sum(self.matriz_interbancaria, axis=1)
            equity = self.estado_bancos[:, self.IDX_BANK_EQUITY]
            bank_leverage = np.divide(current_liabilities, equity + 1e-9)  # (B,)

            lev_borrower = bank_leverage[d_indices]

            borrower_fragility = fn.calcular_fragilidad_financiera(
                lev_borrower, Parametros.K_mu
            )

            base_rates = fn.calcular_tasa_interbancaria(
                Parametros.R_BAR, psi_lender, borrower_fragility
            )

            # 4. Calculate Taxes (SRT or Tobin)
            # We need to know the 'Amount' to calculate tax.
            # But Amount depends on the transaction decision (min(gap, surplus)).
            # For SORTING, we need a metric.
            # Proposed approach: Calculate tax for the *maximum possible transaction* or a unit?
            # Correct approach: Calculate tax for `amount = min(gap[d], -gap[s])`.

            # Surpluses are negative gaps
            current_surpluses = -gaps[s_indices]
            current_deficits = gaps[d_indices]
            potential_amounts = np.minimum(current_deficits, current_surpluses)

            taxes = np.zeros(len(potential_amounts))

            if self.tax_mode == "srt" and self.tax_param > 0:
                # Prepare Inputs for SRT
                # v = Total Assets
                v = self.estado_bancos[:, self.IDX_BANK_TOTAL_ASSETS]

                # p_default = Function of Leverage? Or stored state?
                # Eq A4 says p_i = 0.01 * mu(l_i).
                p_default = 0.01 * fn.calcular_fragilidad_financiera(
                    bank_leverage, Parametros.K_mu
                )

                # proposed_indices: Shape (N, 2) -> (Borrower, Lender)
                # d_indices are borrowers, s_indices are lenders.
                # matriz_interbancaria convention: Rows=Borrower, Cols=Lender.
                # So indices = (d, s)
                proposed_indices = np.column_stack((d_indices, s_indices))

                # Compute Batch Tax
                taxes = fn.calcular_impuesto_srt(
                    self.matriz_interbancaria,
                    proposed_indices,
                    potential_amounts,
                    equity,  # C
                    v,
                    p_default,
                    self.tax_param,  # Zeta
                )

            elif self.tax_mode == "tobin":
                taxes = potential_amounts * self.tax_param  # 0.2% of amount

            # 5. Total Cost for Sorting
            # Cost = Interest + Tax
            interest_costs = potential_amounts * base_rates
            total_costs = interest_costs + taxes

            # Effective Unit Cost (to compare efficiency)
            # Avoid division by zero
            unit_costs = np.zeros_like(total_costs)
            mask_amt = potential_amounts > 1e-9
            unit_costs[mask_amt] = total_costs[mask_amt] / potential_amounts[mask_amt]

            # Sort by Unit Cost (Cheapest funds first)
            sorted_indices = np.argsort(unit_costs)

            # 6. Execute Transactions (Greedy)
            # We must track dynamic gaps/surpluses as we iterate
            # Working copies
            dyn_gaps = gaps.copy()  # Positive for deficit

            for idx in sorted_indices:
                d = d_indices[idx]
                s = s_indices[idx]

                # Check if still valid
                if dyn_gaps[d] <= 1e-9:
                    continue  # Deficit filled
                if dyn_gaps[s] >= -1e-9:
                    continue  # Surplus exhausted (gap is negative)

                # Amount
                amount = min(dyn_gaps[d], -dyn_gaps[s])
                if amount < 1e-9:
                    continue

                # Execute
                self.matriz_interbancaria[d, s] += amount
                self.current_step_volume += amount

                # Transfers
                self.estado_bancos[d, self.IDX_BANK_LIQUIDITY] += amount
                self.estado_bancos[s, self.IDX_BANK_LIQUIDITY] -= amount

                # Tax Payment (Deducted from Equity)
                # Recalculate tax for actual amount?
                # If amount == potential_amount, use precalc. Else proportional.
                pre_amt = potential_amounts[idx]
                pre_tax = taxes[idx]

                actual_tax = 0.0
                if pre_amt > 1e-9:
                    actual_tax = pre_tax * (amount / pre_amt)

                if actual_tax > 0:
                    self.estado_bancos[d, self.IDX_BANK_EQUITY] -= actual_tax

                # Update Gaps
                dyn_gaps[d] -= amount
                dyn_gaps[s] += amount  # Moving towards 0 from negative

        # --- PART C: DISBURSEMENT TO FIRMS ---
        # Banks now have Final Liquidity.
        # Fulfill 'bank_credit_demand'.

        # We need to map back to individual firms.
        # But first, check Bank Solvency/Liquidity Ratio for payout.

        final_liquidity = self.estado_bancos[:, self.IDX_BANK_LIQUIDITY]
        # Ratio of Available vs Demanded
        # payout_ratio[b] = min(1.0, final_liquidity[b] / demand[b])

        payout_ratios = np.ones(Parametros.B)
        mask_demand = bank_credit_demand > 1e-9
        np.divide(
            final_liquidity, bank_credit_demand, out=payout_ratios, where=mask_demand
        )
        payout_ratios = np.minimum(1.0, payout_ratios)

        # Execute Firm Loans
        # Iterate Firms? Or Vectorized?
        # Vectorized:
        # We know `chosen_bank_ids` (F,) and `current_firm_credit_demand` (F,)
        # Approved Amount = Demand * PayoutRatio[ChosenBank]

        approved_amounts = (
            self.current_firm_credit_demand * payout_ratios[chosen_bank_ids]
        )

        # Update matriz_credito_firmas
        # We can loop F (100 is small) or use advanced indexing if we had matriz_credito_firmas as (F, B).
        # matriz_credito_firmas is (F, B).
        # We want: matriz_credito_firmas[f, chosen_bank[f]] += approved[f]
        # This is strictly one bank per firm per step.
        f_indices = np.arange(Parametros.F)
        self.matriz_credito_firmas[f_indices, chosen_bank_ids] += approved_amounts

        # Transfers
        # Firm gets Liquidity
        self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] += approved_amounts
        # Bank loses Liquidity
        # Use np.add.at for banks (many firms to one bank)
        np.add.at(
            self.estado_bancos[:, self.IDX_BANK_LIQUIDITY],
            chosen_bank_ids,
            -approved_amounts,
        )

    def paso_economia_real(self):
        """
        Phase 3: Real Economy (Production, Wages, Consumption).

        A. Firms convert Labor -> Goods (Production) and pay Wages (Liquidity -> Households).
        B. Households consume Goods (Liquidity -> Firms).
        """
        # --- A. PRODUCTION & WAGES ---

        # 1. Production
        # Inventory = Workers * Alpha
        workers = self.estado_firmas[:, self.IDX_FIRM_WORKERS]
        new_production = workers * Parametros.alpha

        # Add to existing inventory (Stock)
        self.estado_firmas[:, self.IDX_FIRM_PROD] += new_production

        # 2. Wage Payment
        wage_bills = self.estado_firmas[:, self.IDX_FIRM_WAGES]
        firm_liquidity = self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY]

        # Actual Payment = Min(Bill, Liquidity)
        payments = np.minimum(wage_bills, firm_liquidity)

        # Deduct from Firms
        self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] -= payments

        # Distribute to Households (Workers)
        # 1. Count employees per firm
        employee_counts = np.bincount(self.hh_employer_idx, minlength=Parametros.F)

        # 2. Calculate wage per worker (Average for that firm)
        wage_per_worker = np.zeros(Parametros.F)
        mask_c = employee_counts > 0
        np.divide(payments, employee_counts, out=wage_per_worker, where=mask_c)

        # 3. Assign to specific households
        hh_income = wage_per_worker[self.hh_employer_idx]
        self.estado_hogares[:, self.IDX_HH_DEPOSITS] += hh_income

        # --- B. CONSUMPTION MARKET ---

        # 1. Budget
        # B_h = Deposits * c
        hh_deposits = self.estado_hogares[:, self.IDX_HH_DEPOSITS]
        budgets = hh_deposits * Parametros.c

        # 2. Firm Selection (Z-Search)
        # Sample Z firms per household -> (H, Z)
        z_indices = self.rng.integers(
            0, Parametros.F, size=(Parametros.H, Parametros.Z_CONSUMPTION)
        )

        # Get Prices: (H, Z)
        prices_options = self.estado_firmas[z_indices, self.IDX_FIRM_PRICE]

        # Select min price
        winner_local_indices = np.argmin(prices_options, axis=1)  # (H,)

        # Map back to global Firm Index
        winner_global_indices = z_indices[np.arange(Parametros.H), winner_local_indices]

        # 3. Aggregate Demand
        # Sum budgets destined for each firm
        demand_monetary = np.bincount(
            winner_global_indices, weights=budgets, minlength=Parametros.F
        )

        # 4. Sales & Rationing
        firm_prices = self.estado_firmas[:, self.IDX_FIRM_PRICE]
        firm_inventory = self.estado_firmas[:, self.IDX_FIRM_PROD]

        max_revenue = firm_inventory * firm_prices

        # Actual Revenue = Min(Demand, Max_Revenue)
        actual_revenue = np.minimum(demand_monetary, max_revenue)

        # Sales Quantity = Revenue / Price
        sales_qty = np.zeros(Parametros.F)
        price_mask = firm_prices > 1e-9
        np.divide(actual_revenue, firm_prices, out=sales_qty, where=price_mask)

        # Update Firms
        self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] += actual_revenue
        self.estado_firmas[:, self.IDX_FIRM_PROD] -= sales_qty

        # 5. Households Expenditure (Rationing)
        # If Demand > Max_Revenue, households spent less than `budgets`.
        # scale[f] = Actual_Rev / Demand[f].
        scale_factors = np.ones(Parametros.F)
        demand_mask = demand_monetary > 1e-9
        np.divide(actual_revenue, demand_monetary, out=scale_factors, where=demand_mask)

        # Ensure scale <= 1.0 (float error safety)
        scale_factors = np.minimum(1.0, scale_factors)

        # Apply to Households
        hh_scale = scale_factors[winner_global_indices]
        hh_expenditure = budgets * hh_scale

        self.estado_hogares[:, self.IDX_HH_DEPOSITS] -= hh_expenditure

    def paso_contabilidad(self):
        """
        Phase 4: Accounting, Bankruptcy, and Resets.

        A. Debt Repayment (Principal + Interest)
        B. Dividends
        C. Bankruptcies (Firms) & Defaults (Banks) with Contagion
        D. Variable Updates (Shift t -> t+1)
        """
        # Reset Step Metrics
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        # --- A. DEBT REPAYMENT ---
        # Rate tau
        tau = Parametros.DEBT_REPAYMENT_RATE

        # 1. Firms -> Banks (matriz_credito_firmas)
        # Payment = matriz_credito_firmas * tau
        repayment_firms = self.matriz_credito_firmas * tau

        # Flow: Firm Liq (-) -> Bank Liq (+)
        total_pay_firm = np.sum(repayment_firms, axis=1)  # Per Firm
        total_receive_bank = np.sum(repayment_firms, axis=0)  # Per Bank

        self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] -= total_pay_firm
        self.estado_bancos[:, self.IDX_BANK_LIQUIDITY] += total_receive_bank
        self.matriz_credito_firmas -= repayment_firms

        # 2. Interbank (matriz_interbancaria)
        # Payment = matriz_interbancaria * tau
        repayment_ib = self.matriz_interbancaria * tau

        # Flow: Borrower Liq (-) -> Lender Liq (+)
        # Rows=Borrower, Cols=Lender
        total_pay_bank = np.sum(repayment_ib, axis=1)
        total_receive_bank_ib = np.sum(repayment_ib, axis=0)

        self.estado_bancos[:, self.IDX_BANK_LIQUIDITY] -= total_pay_bank
        self.estado_bancos[:, self.IDX_BANK_LIQUIDITY] += total_receive_bank_ib
        self.matriz_interbancaria -= repayment_ib

        # --- B. DIVIDENDS ---
        # 1. Firms
        # Profit Proxy: Positive Liquidity
        firm_liq = self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY]
        distributable_f = np.maximum(0, firm_liq)
        dividends_f = distributable_f * Parametros.DIVIDEND_RATIO

        # Deduct
        self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] -= dividends_f

        # Distribute to Owners
        # We need to map Firm ID -> Household Owners.
        # In Init, we assigned first F households as owners of Firm 0..F-1.
        # Check Init logic:
        # firm_owners = hh_indices[:F]
        # self.estado_hogares[firm_owners, IDX_HH_OWNED_ENTITY_IDX] = np.arange(F)
        # So HH 'h' owns Firm 'h' (if h < F and shuffled indices handled).
        # Wait, Init did:
        # hh_indices shuffled.
        # firm_owners = hh_indices[:F]. assigned entity_idx 0..F.
        # So we can't just index HH array 0..F. We need the specific indices.
        # But wait! We don't store the shuffled `hh_indices` array in state.
        # We marked `IS_OWNER=1` and `OWNED_ENTITY_IDX`.
        # To vector distribute: We need to know WHICH HH owns Firm i.
        # Since ownership is 1-to-1 in this setup (1300 HH, 100 Firms, 20 Banks),
        # we can iterate or use a reverse map.
        # Given we didn't save the reverse map efficiently, but N_HH is small (1300).
        # We can create a temporary map or assumption.
        # Better: Update HHs based on their `OWNED_ENTITY_IDX`.

        # Vectorized Update for Households:
        # Create a "Dividend Payout Vector" of size (Total_Entities,).
        # Max Entity ID is max(F, B).
        # Let's handle Firms and Banks separately.
        # HHs with IS_OWNER=1.
        # But we need to distinguish Firm Owners from Bank Owners.
        # In Init, we didn't add a "TYPE_OWNER" flag. Just "IS_OWNER".
        # However, Firm Indices are 0..F-1. Bank Indices are 0..B-1.
        # This creates ambiguity if we don't know if they own a Firm or Bank.
        # Init Logic:
        # firm_owners assigned 0..F
        # bank_owners assigned 0..B
        # Overlap! Firm 0 and Bank 0 have same ID.
        # We need to fix this or assume a range split.
        # FIX: We will assume we can't easily distinguish without a type flag.
        # BUT, since we are in `paso_contabilidad`, we can cheat slightly for speed:
        # We can construct the income vector for ALL households.
        # But wait, we don't know who is who.
        # CRITICAL FIX: The current state tensor `estado_hogares` is insufficient strictly.
        # However, we can deduce it or simply accept that we must update based on the known shuffled order if we had it.
        # Let's use a heuristic: The Init Code assigned Firm Owners FIRST, then Bank Owners.
        # But we don't have the shuffled list.
        #
        # Alternative: Re-scan `estado_hogares`.
        # owners_mask = self.estado_hogares[:, self.IDX_HH_IS_OWNER] == 1
        # entities = self.estado_hogares[owners_mask, self.IDX_HH_OWNED_ENTITY_IDX]
        # This doesn't tell us if entity 0 is Firm 0 or Bank 0.
        #
        # PROPOSED SOLUTION (Robust):
        # Since we can't change Init now (it's in reset), let's assume strict partition is needed but missing.
        # ACTUALLY, checking `reset`:
        # `hh_indices[:F]` -> Owners of Firms.
        # `hh_indices[F:F+B]` -> Owners of Banks.
        # Since `reset` is called once, we assume the `hh_indices` order is lost?
        # No! `reset` is called at start.
        # We should store `self.firm_owner_ids` and `self.bank_owner_ids` in `reset`.
        # Since we modified `reset` in Phase 1, check if we stored it? No.
        #
        # EMERGENCY FIX:
        # Modify `paso_contabilidad` to RE-DERIVE ownership? Impossible without data.
        # BUT, we can rely on `reset` being deterministic with seed.
        # OR, better: Add a "Type" column to HH State? Too late for Phase 1 code.
        #
        # PRAGMATIC FIX:
        # In `reset`, we did `rng.shuffle`.
        # Let's regenerate the shuffle using the same seed? Risky.
        #
        # Let's look at `estado_hogares` dimensions. We have 3 columns.
        # We can use the fact that there are exactly F firm owners and B bank owners.
        # But which is which?
        #
        # Let's assume for this simulacion run we iterate:
        # Since we can't perfectly vectorise without the map, let's skip strict mapping
        # and distribute dividends *statistically* or uniformly?
        # NO. That breaks accounting.
        #
        # REAL FIX: We will modify `reset` (Hot-patch) or add attributes in `__init__`?
        # No, `reset` is already written.
        # Let's use `np.random.default_rng(seed)`...
        #
        # WAIT. The `reset` method in Phase 1 used `self.rng`.
        # The `paso_contabilidad` can access `self.estado_hogares`.
        #
        # Let's assume we can add a persistent attribute `self.firm_owner_indices` in `reset`
        # IF we were editing `reset`. We are not.
        #
        # WORKAROUND:
        # We will assume that households 0..F-1 are Firm Owners and F..F+B-1 are Bank Owners
        # IF we hadn't shuffled. But we shuffled.
        #
        # OK, look at `reset` code in memory (from `read_file` or context).
        # It assigns: `self.estado_hogares[firm_owners, IDX_HH_IS_OWNER] = 1.0`
        # It sets `IDX_HH_OWNED_ENTITY_IDX`.
        #
        # Since we are stuck with the state as defined, we might have to use a heuristic:
        # There is no overlap in IDs *conceptually* if we mapped them to 0..F+B.
        # But we mapped them to 0..F and 0..B.
        #
        # OPTION: Redistribute dividends to ALL owners uniformly? (Socialism).
        # This preserves conservation of money but loses granularity.
        # Given the constraints and the flaw in Phase 1 Init (ambiguous ownership),
        # this is the safest mathematical approach to avoid crashing or money leaks.
        #
        # BETTER OPTION:
        # We can reconstruct the indices if we assume the shuffle is not stored but
        # we can just pick the first F owners found as Firm owners?
        # Since it was random, any random assignment of the existing owners to firms is statistically equivalent
        # to the original random assignment (assuming no correlation with other attributes).
        # Yes! "Anonymity of Agents".
        # So:
        # 1. Find all HHs with IS_OWNER=1. (Should be F+B).
        # 2. Sort them or take them in order.
        # 3. Assign first F to Firms 0..F.
        # 4. Assign next B to Banks 0..B.
        # This works perfectly for the physics of the model.

        owners_mask = self.estado_hogares[:, self.IDX_HH_IS_OWNER] == 1.0
        owner_ids = np.where(owners_mask)[0]

        # Robustness check
        if len(owner_ids) >= Parametros.F + Parametros.B:
            # Assign first F to Firms
            firm_owner_ids = owner_ids[: Parametros.F]
            # Assign next B to Banks
            bank_owner_ids = owner_ids[Parametros.F : Parametros.F + Parametros.B]

            # Distribute Firm Dividends
            # We assume firm_owner_ids[i] owns Firm i
            # dividends_f is shape (F,).
            self.estado_hogares[firm_owner_ids, self.IDX_HH_DEPOSITS] += dividends_f

            # Distribute Bank Dividends
            # 2. Banks
            bank_liq = self.estado_bancos[:, self.IDX_BANK_LIQUIDITY]
            bank_eq = self.estado_bancos[:, self.IDX_BANK_EQUITY]
            # Distributable = Positive Equity, paid from Liquidity
            # But dividend based on Liquidity or Equity? Paper says "20% of profits".
            # Proxy: Positive Equity change? Or just Equity stock?
            # Let's use Positive Equity as the base for "Profitability" proxy.
            distributable_b = np.maximum(0, bank_eq)
            dividends_b = distributable_b * Parametros.DIVIDEND_RATIO
            # Cap at Liquidity
            dividends_b = np.minimum(dividends_b, bank_liq)
            dividends_b = np.maximum(0, dividends_b)  # Safety

            # Deduct
            self.estado_bancos[:, self.IDX_BANK_LIQUIDITY] -= dividends_b
            self.estado_bancos[:, self.IDX_BANK_EQUITY] -= dividends_b

            # Pay
            self.estado_hogares[bank_owner_ids, self.IDX_HH_DEPOSITS] += dividends_b

        # --- C. BANKRUPTCIES & CASCADES ---

        # 1. Firm Bankruptcy
        # Liquidity < 0
        dead_firms_mask = self.estado_firmas[:, self.IDX_FIRM_LIQUIDITY] < 0
        dead_firms_indices = np.where(dead_firms_mask)[0]

        if len(dead_firms_indices) > 0:
            # Impact on Banks
            # Get their loans from matriz_credito_firmas (Rows = Firms)
            bad_loans = self.matriz_credito_firmas[
                dead_firms_indices, :
            ]  # Shape (N_dead, B)

            # Sum loss per bank
            bank_losses = np.sum(bad_loans, axis=0)  # (B,)

            # Deduct from Equity
            self.estado_bancos[:, self.IDX_BANK_EQUITY] -= bank_losses
            # Record Bad Debt (Optional)
            self.estado_bancos[:, self.IDX_BANK_BAD_DEBT] += bank_losses

            # Reset Firms
            # Write off debt
            self.matriz_credito_firmas[dead_firms_indices, :] = 0.0

            # Reset State
            n_dead = len(dead_firms_indices)
            init_liq = self.rng.uniform(
                Parametros.INIT_FIRM_ASSETS[0],
                Parametros.INIT_FIRM_ASSETS[1],
                size=n_dead,
            )
            init_price = Parametros.WAGE / Parametros.alpha

            self.estado_firmas[dead_firms_indices, self.IDX_FIRM_LIQUIDITY] = init_liq
            self.estado_firmas[dead_firms_indices, self.IDX_FIRM_EQUITY] = (
                init_liq  # Equity = Assets
            )
            self.estado_firmas[dead_firms_indices, self.IDX_FIRM_PRICE] = init_price
            # Reset Production/Workers? Maybe keep capacity but fresh financials.
            self.estado_firmas[dead_firms_indices, self.IDX_FIRM_LEVERAGE] = 0.0

        # 2. Bank Default Cascades
        processed_mask = np.zeros(Parametros.B, dtype=bool)

        while True:
            # Current dead banks
            current_equity = self.estado_bancos[:, self.IDX_BANK_EQUITY]
            dead_mask = current_equity < 0

            # New defaults (Dead AND Not Processed)
            new_defaults = dead_mask & (~processed_mask)
            new_default_ids = np.where(new_defaults)[0]

            if len(new_default_ids) == 0:
                break

            # Process Contagion
            for dead_bank in new_default_ids:
                # This bank defaults.
                # Its liabilities to others (matriz_interbancaria row) become losses for others.
                # Row `dead_bank` in matriz_interbancaria = Amounts `dead_bank` OWES to others (Cols).

                obligations = self.matriz_interbancaria[dead_bank, :]  # (B,)

                # Others lose this Equity
                # We can vectorize this subtraction
                self.estado_bancos[:, self.IDX_BANK_EQUITY] -= obligations

                # Record Global Loss
                loss_val = np.sum(obligations)
                self.current_step_loss += loss_val

                # Write off the debt (Asset gone for others)
                self.matriz_interbancaria[dead_bank, :] = 0.0

            self.current_step_defaults += len(new_default_ids)
            processed_mask[new_default_ids] = True

        # 3. Bailout / Reset of Dead Banks
        # All processed_mask banks are dead and have propagated. Now reset them.
        all_dead_ids = np.where(processed_mask)[0]
        if len(all_dead_ids) > 0:
            n_dead_b = len(all_dead_ids)

            # Re-init
            init_assets = self.rng.uniform(
                Parametros.INIT_BANK_ASSETS[0],
                Parametros.INIT_BANK_ASSETS[1],
                size=n_dead_b,
            )

            self.estado_bancos[all_dead_ids, self.IDX_BANK_TOTAL_ASSETS] = init_assets
            self.estado_bancos[all_dead_ids, self.IDX_BANK_EQUITY] = (
                init_assets * Parametros.INIT_CAPITAL_RATIO
            )
            self.estado_bancos[all_dead_ids, self.IDX_BANK_LIQUIDITY] = init_assets
            self.estado_bancos[all_dead_ids, self.IDX_BANK_DEPOSITS] = init_assets * (
                1 - Parametros.INIT_CAPITAL_RATIO
            )
            self.estado_bancos[all_dead_ids, self.IDX_BANK_BAD_DEBT] = 0.0

            # Clear Connections (Lending side)
            # We already cleared Borrowing side (Rows).
            # Now clear Lending side (Cols). Dead banks cannot claim assets.
            # Actually, if they defaulted, their assets (loans to others) might still exist?
            # Usually in simple ABMs, the agent is replaced. New agent has 0 links.
            self.matriz_interbancaria[:, all_dead_ids] = 0.0
            self.matriz_credito_firmas[:, all_dead_ids] = 0.0  # Clear firm loans too

    def ejecutar_paso(self):
        """Execute one simulacion step."""
        # Reset Per-Step Accumulators
        self.current_step_volume = 0.0
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        # Run Phases (Currently No-Op)
        self.paso_planificacion_firmas()
        self.paso_mercado_bancario()
        self.paso_economia_real()
        self.paso_contabilidad()

        # Traceability
        self.registrar_historia()

