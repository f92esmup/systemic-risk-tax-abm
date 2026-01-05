import numpy as np
from parametros import Parametros
import funciones as fn


class Modelo_CRISIS:
    """
    Vectorized implementation of the Poledna & Thurner (2016) ABM model.
    Phase 1: Data Architecture & Initialization.
    """

    def __init__(self, seed=None, tax_mode="none", tax_param=0.0):
        self.rng = np.random.default_rng(seed)

        # Experiment Logic
        self.tax_mode = tax_mode.lower()
        self.tax_param = tax_param

        # Metrics for Analysis (Placeholders)
        self.current_step_loss = 0.0
        self.current_step_defaults = 0
        self.current_step_volume = 0.0

        self.reset()

    def reset(self):
        """
        Reset or initialize all state tensors and network topologies for a new simulation run.
        """
        # --- Dimensions ---
        B = Parametros.B
        F = Parametros.F
        H = Parametros.H

        # --- 1. Agents State (Tensors) ---
        self.estado_bancos = np.zeros((B, Parametros.N_BANK_FEATURES), dtype=np.float64)
        self.estado_firmas = np.zeros((F, Parametros.N_FIRM_FEATURES), dtype=np.float64)
        self.estado_hogares = np.zeros((H, Parametros.N_HH_FEATURES), dtype=np.float64)

        # --- 2. Vectorized Initialization ---

        # --- BANKS ---
        # Specificity Parameters (Constant per run)
        # CHI ~ U(0, 1)
        self.estado_bancos[:, Parametros.IDX_BANK_OPERATING_COST_CHI] = (
            self.rng.uniform(Parametros.CHI_RANGE[0], Parametros.CHI_RANGE[1], size=B)
        )
        # PSI ~ U(0, 0.1)
        self.estado_bancos[:, Parametros.IDX_BANK_INTERBANK_COST_PSI] = (
            self.rng.uniform(Parametros.PSI_RANGE[0], Parametros.PSI_RANGE[1], size=B)
        )

        # Financials
        init_bank_assets = self.rng.uniform(
            Parametros.INIT_BANK_ASSETS[0], Parametros.INIT_BANK_ASSETS[1], size=B
        )
        self.estado_bancos[:, Parametros.IDX_BANK_TOTAL_ASSETS] = init_bank_assets
        # Equity = Assets * Capital Ratio
        self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] = (
            init_bank_assets * Parametros.INIT_CAPITAL_RATIO
        )
        # Liquidity = Assets (Assuming start with all liquid)
        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] = init_bank_assets
        # Deposits = Assets - Equity
        self.estado_bancos[:, Parametros.IDX_BANK_DEPOSITS] = (
            init_bank_assets - self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
        )

        # --- FIRMS ---
        init_firm_assets = self.rng.uniform(
            Parametros.INIT_FIRM_ASSETS[0], Parametros.INIT_FIRM_ASSETS[1], size=F
        )
        self.estado_firmas[:, Parametros.IDX_FIRM_EQUITY] = init_firm_assets
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] = init_firm_assets

        # Initialize Price to Marginal Cost
        init_price = Parametros.WAGE / Parametros.alpha
        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE] = init_price
        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE_PREV] = init_price

        # --- HOUSEHOLDS (CRITICAL UPDATE) ---
        # 1. Shuffle Household Indices
        hh_indices = np.arange(H)
        self.rng.shuffle(hh_indices)

        # 2. Assign Owners
        # First F households -> Firm Owners
        firm_owners = hh_indices[:F]
        self.estado_hogares[firm_owners, Parametros.IDX_HH_IS_OWNER] = 1.0
        self.estado_hogares[firm_owners, Parametros.IDX_HH_OWNED_TYPE] = 1.0  # 1=Firm
        self.estado_hogares[firm_owners, Parametros.IDX_HH_OWNED_ENTITY_IDX] = (
            np.arange(F)
        )

        # Next B households -> Bank Owners
        bank_owners = hh_indices[F : F + B]
        self.estado_hogares[bank_owners, Parametros.IDX_HH_IS_OWNER] = 1.0
        self.estado_hogares[bank_owners, Parametros.IDX_HH_OWNED_TYPE] = 2.0  # 2=Bank
        self.estado_hogares[bank_owners, Parametros.IDX_HH_OWNED_ENTITY_IDX] = (
            np.arange(B)
        )

        # 3. Assign Employers (To ALL households, assuming full employment or potential)
        # Randomly assign an employer index (0..F-1)
        self.estado_hogares[:, Parametros.IDX_HH_EMPLOYER_IDX] = self.rng.integers(
            0, F, size=H
        )

        # --- 3. Topology & Relationships ---
        self.matriz_interbancaria = np.zeros((B, B), dtype=np.float64)
        self.matriz_credito_firmas = np.zeros((F, B), dtype=np.float64)

        # Note: hh_employer_idx is now in self.estado_hogares
        # hh_bank_idx (Choice of bank for deposits?) - Not strictly in state tensor, can be auxiliary.
        # Keeping it as auxiliary if needed for loop, but strictly state tensor has the core data.
        # We will keep hh_bank_idx auxiliary for consistency with previous logic if needed.
        self.hh_bank_idx = self.rng.integers(0, B, size=H)

        # --- 4. History / Traceability ---
        self.step_buffer = {
            "matriz_interbancaria": [],
            "matriz_credito_firmas": [],
            "estado_bancos": [],
            "estado_firmas": [],
            "estado_hogares": [],  # Full tensor capture
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
        self.step_buffer["estado_hogares"].append(
            self.estado_hogares.astype(np.float32).copy()
        )

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
        Phase 2: Firms Planning.
        - Update Prices based on market average.
        - Update Expected Demand (simple stochastic process).
        - Calculate Labor requirements.
        - Update Financial Health (Leverage) for Appendix A interest rates.
        - Calculate Credit Demand if Wages > Liquidity.
        """
        # --- 1. Update Prices ---
        prices = self.estado_firmas[:, Parametros.IDX_FIRM_PRICE]

        # Save to PREV
        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE_PREV] = prices.copy()

        # Calculate market average price
        p_avg = np.mean(prices)

        # Noise component
        noise = self.rng.normal(0, Parametros.PRICE_DRIFT_STD, size=Parametros.F)

        # Adjustment Rule: p_new = p * (1 + speed * (p_avg - p)/p_avg + noise)
        # Avoid division by zero if p_avg is 0
        if p_avg > 1e-9:
            adjustment = Parametros.PRICE_ADJUSTMENT_SPEED * (p_avg - prices) / p_avg
            new_prices = prices + prices * (adjustment + noise)
        else:
            new_prices = prices * (1 + noise)

        # Constraint: Prices must be positive
        new_prices = np.maximum(new_prices, 0.01)

        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE] = new_prices

        # --- 2. Update Demand Expectations ---
        # Save previous demand
        self.estado_firmas[:, Parametros.IDX_FIRM_DEMAND_PREV] = self.estado_firmas[
            :, Parametros.IDX_FIRM_DEMAND
        ].copy()

        # Initialize demand if 0 (start of sim)
        current_demand = self.estado_firmas[:, Parametros.IDX_FIRM_DEMAND]

        if np.all(current_demand == 0):
            current_demand = self.rng.uniform(10, 50, size=Parametros.F)

        # Random Walk: D(t) = D(t-1) * (1 + noise)
        # Using 5% volatility as a reasonable default for "xi"
        demand_shock = self.rng.normal(0, 0.05, size=Parametros.F)
        new_demand = current_demand * (1 + demand_shock)
        new_demand = np.maximum(new_demand, 0.0)

        self.estado_firmas[:, Parametros.IDX_FIRM_DEMAND] = new_demand

        # --- 3. Production Planning ---
        # Labor Needed = Ceil(Demand / Alpha)
        labor_needed = np.ceil(new_demand / Parametros.alpha)

        self.estado_firmas[:, Parametros.IDX_FIRM_WORKERS] = labor_needed
        self.estado_firmas[:, Parametros.IDX_FIRM_PROD] = (
            labor_needed * Parametros.alpha
        )

        # Wage Bill
        wage_bill = labor_needed * Parametros.WAGE
        self.estado_firmas[:, Parametros.IDX_FIRM_WAGES] = wage_bill

        # --- 4. Financial Health Update (Leverage) ---
        # Leverage = Debt / (Liquidity + epsilon)
        # Current Debt = Sum of loans from all banks (matriz_credito_firmas rows)
        current_debt = np.sum(self.matriz_credito_firmas, axis=1)  # (F,)
        liquidity = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]

        leverage = current_debt / (liquidity + 1e-9)
        self.estado_firmas[:, Parametros.IDX_FIRM_LEVERAGE] = leverage

        # --- 5. Credit Demand Calculation ---
        # Gap = Wages - Liquidity
        gap = wage_bill - liquidity

        # Credit Demand > 0 only if Gap > 0
        credit_demand = np.maximum(gap, 0.0)

        # Store for next phase
        self.current_firm_credit_demand = credit_demand

    def paso_mercado_bancario(self):
        """
        Phase 3: Credit Market (Firms-Banks) & Interbank Market (SRT).

        Part A: Firms request credit from Banks (Eq A1).
        Part B: Banks manage liquidity deficits via Interbank Market using DebtRank-based SRT (Eq A2 + Eq 5).
        Part C: Disbursement of funds to firms.
        """
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
            pool_indices, Parametros.IDX_BANK_OPERATING_COST_CHI
        ]

        # Get Firm Leverage
        # self.estado_firmas column LEVERAGE is (F,) -> Broadcast to (F, N)
        firm_leverage = self.estado_firmas[:, Parametros.IDX_FIRM_LEVERAGE]
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
        # self.current_firm_credit_demand is (F,) from Phase 2
        bank_credit_demand = np.zeros(Parametros.B)
        np.add.at(bank_credit_demand, chosen_bank_ids, self.current_firm_credit_demand)

        # --- PART B: INTERBANK MARKET & SRT ---

        # 1. Identify Liquidity Gaps
        bank_liquidity = self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
        gaps = bank_credit_demand - bank_liquidity

        deficit_ids = np.where(gaps > 0)[0]
        surplus_ids = np.where(gaps < 0)[0]  # Negative gap means surplus

        if len(deficit_ids) > 0 and len(surplus_ids) > 0:
            # 2. Generate Candidate Pairs (Deficit, Surplus)
            n_def = len(deficit_ids)
            n_sur = len(surplus_ids)

            # d_indices: [d1, d1, ..., d2, d2, ...]
            # s_indices: [s1, s2, ..., s1, s2, ...]
            d_indices = np.repeat(deficit_ids, n_sur)
            s_indices = np.tile(surplus_ids, n_def)

            # 3. Calculate Base Rates (Eq A2)
            # r_ij = r_bar * (1 + psi_i * mu(leverage_j))
            # i = Lender (s_indices), j = Borrower (d_indices)

            psi_lender = self.estado_bancos[
                s_indices, Parametros.IDX_BANK_INTERBANK_COST_PSI
            ]

            # Borrower Leverage needed.
            # Bank Leverage = (Deposits + Interbank Borrowing) / Equity
            # Use current state.
            current_ib_borrowing = np.sum(
                self.matriz_interbancaria, axis=1
            )  # Row sum = Borrowing
            current_deposits = self.estado_bancos[:, Parametros.IDX_BANK_DEPOSITS]
            equity = self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]

            # Avoid division by zero
            bank_leverage = (current_deposits + current_ib_borrowing) / (equity + 1e-9)

            lev_borrower = bank_leverage[d_indices]

            borrower_fragility = fn.calcular_fragilidad_financiera(
                lev_borrower, Parametros.K_mu
            )

            base_rates = fn.calcular_tasa_interbancaria(
                Parametros.R_BAR, psi_lender, borrower_fragility
            )

            # 4. Calculate Taxes (SRT or Tobin)
            # Calculate 'potential_amount' = min(deficit, surplus)
            # Surpluses are negative gaps, so take abs or negation
            current_surpluses = -gaps[s_indices]
            current_deficits = gaps[d_indices]
            potential_amounts = np.minimum(current_deficits, current_surpluses)

            taxes = np.zeros(len(potential_amounts))

            if self.tax_mode == "srt" and self.tax_param > 0:
                # Prepare Inputs for SRT
                v = self.estado_bancos[:, Parametros.IDX_BANK_TOTAL_ASSETS]

                # p_default = 0.01 * mu(l_i)
                p_default = 0.01 * fn.calcular_fragilidad_financiera(
                    bank_leverage, Parametros.K_mu
                )

                # proposed_indices: Shape (N, 2) -> (Borrower, Lender)
                # matriz_interbancaria convention: Rows=Borrower, Cols=Lender
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
                taxes = potential_amounts * self.tax_param  # Flat rate

            # 5. Total Cost for Sorting
            # Cost = Interest + Tax
            interest_costs = potential_amounts * base_rates
            total_costs = interest_costs + taxes

            # Effective Unit Cost (to compare efficiency)
            unit_costs = np.zeros_like(total_costs)
            mask_amt = potential_amounts > 1e-9
            unit_costs[mask_amt] = total_costs[mask_amt] / potential_amounts[mask_amt]

            # Sort by Unit Cost (Cheapest funds first)
            sorted_indices = np.argsort(unit_costs)

            # 6. Execute Transactions (Greedy)
            # Track dynamic gaps/surpluses
            dyn_gaps = gaps.copy()

            for idx in sorted_indices:
                d = d_indices[idx]
                s = s_indices[idx]

                # Check validity
                if dyn_gaps[d] <= 1e-9:
                    continue  # Deficit filled
                if dyn_gaps[s] >= -1e-9:
                    continue  # Surplus exhausted

                # Amount
                amount = min(dyn_gaps[d], -dyn_gaps[s])
                if amount < 1e-9:
                    continue

                # Execute Interbank Loan
                self.matriz_interbancaria[d, s] += amount
                self.current_step_volume += amount

                # Transfers
                self.estado_bancos[d, Parametros.IDX_BANK_LIQUIDITY] += amount
                self.estado_bancos[s, Parametros.IDX_BANK_LIQUIDITY] -= amount

                # Tax Payment (Deducted from Equity)
                # Proportional tax if amount < potential_amount
                pre_amt = potential_amounts[idx]
                pre_tax = taxes[idx]

                actual_tax = 0.0
                if pre_amt > 1e-9:
                    actual_tax = pre_tax * (amount / pre_amt)

                if actual_tax > 0:
                    self.estado_bancos[d, Parametros.IDX_BANK_EQUITY] -= actual_tax

                # Update Dynamic Gaps
                dyn_gaps[d] -= amount
                dyn_gaps[s] += amount

        # --- PART C: DISBURSEMENT TO FIRMS ---
        # Banks now have Final Liquidity.
        # Fulfill 'bank_credit_demand'.

        final_liquidity = self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]

        # Payout Ratio: min(1.0, Available / Demanded)
        payout_ratios = np.ones(Parametros.B)
        mask_demand = bank_credit_demand > 1e-9
        np.divide(
            final_liquidity, bank_credit_demand, out=payout_ratios, where=mask_demand
        )
        payout_ratios = np.minimum(1.0, payout_ratios)

        # Calculate Approved Amounts per Firm
        approved_amounts = (
            self.current_firm_credit_demand * payout_ratios[chosen_bank_ids]
        )

        # Update matriz_credito_firmas
        # One bank per firm per step (chosen_bank_ids)
        f_indices = np.arange(Parametros.F)
        self.matriz_credito_firmas[f_indices, chosen_bank_ids] += approved_amounts

        # Transfers
        # Firm gets Liquidity
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] += approved_amounts

        # Bank loses Liquidity
        np.add.at(
            self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY],
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
        workers = self.estado_firmas[:, Parametros.IDX_FIRM_WORKERS]
        new_production = workers * Parametros.alpha

        # Add to existing inventory (Stock)
        # Note: In Phase 2 we set IDX_FIRM_PROD to new_production.
        # Ideally, IDX_FIRM_PROD should accumulate if we want inventory dynamics.
        # But per Phase 2 logic, we overwrote it. Let's assume perishable goods or overwrite for now
        # as per previous logic, OR simpler: Phase 2 set the *capacity*?
        # Let's trust Phase 2 set IDX_FIRM_PROD correctly as the goods available now.
        # If we wanted accumulation: self.estado_firmas[:, ...] += new_production.
        # Given Phase 2: self.estado_firmas[:, IDX_FIRM_PROD] = labor * alpha.
        # We will proceed with that as the "Inventory for this period".

        # 2. Wage Payment
        wage_bills = self.estado_firmas[:, Parametros.IDX_FIRM_WAGES]
        firm_liquidity = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]

        # Actual Payment = Min(Bill, Liquidity)
        payments = np.minimum(wage_bills, firm_liquidity)

        # Deduct from Firms
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= payments

        # Distribute to Households (Workers)
        # 1. Count employees per firm
        # Household Employer Index
        hh_emp_idx = self.estado_hogares[:, Parametros.IDX_HH_EMPLOYER_IDX].astype(int)

        employee_counts = np.bincount(hh_emp_idx, minlength=Parametros.F)

        # 2. Calculate wage per worker (Average for that firm)
        wage_per_worker = np.zeros(Parametros.F)
        mask_c = employee_counts > 0
        np.divide(payments, employee_counts, out=wage_per_worker, where=mask_c)

        # 3. Assign to specific households
        hh_income = wage_per_worker[hh_emp_idx]
        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_income

        # --- B. CONSUMPTION MARKET ---

        # 1. Budget
        # B_h = Deposits * c
        hh_deposits = self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS]
        budgets = hh_deposits * Parametros.c

        # 2. Firm Selection (Z-Search)
        # Sample Z firms per household -> (H, Z)
        z_indices = self.rng.integers(
            0, Parametros.F, size=(Parametros.H, Parametros.Z_CONSUMPTION)
        )

        # Get Prices: (H, Z)
        prices_options = self.estado_firmas[z_indices, Parametros.IDX_FIRM_PRICE]

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
        firm_prices = self.estado_firmas[:, Parametros.IDX_FIRM_PRICE]
        firm_inventory = self.estado_firmas[:, Parametros.IDX_FIRM_PROD]

        max_revenue = firm_inventory * firm_prices

        # Actual Revenue = Min(Demand, Max_Revenue)
        actual_revenue = np.minimum(demand_monetary, max_revenue)

        # Sales Quantity = Revenue / Price
        sales_qty = np.zeros(Parametros.F)
        price_mask = firm_prices > 1e-9
        np.divide(actual_revenue, firm_prices, out=sales_qty, where=price_mask)

        # Update Firms
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] += actual_revenue
        self.estado_firmas[:, Parametros.IDX_FIRM_PROD] -= sales_qty

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

        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] -= hh_expenditure

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

        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= total_pay_firm
        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] += total_receive_bank
        self.matriz_credito_firmas -= repayment_firms

        # 2. Interbank (matriz_interbancaria)
        # Payment = matriz_interbancaria * tau
        repayment_ib = self.matriz_interbancaria * tau

        # Flow: Borrower Liq (-) -> Lender Liq (+)
        # Rows=Borrower, Cols=Lender
        total_pay_bank = np.sum(repayment_ib, axis=1)
        total_receive_bank_ib = np.sum(repayment_ib, axis=0)

        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] -= total_pay_bank
        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] += total_receive_bank_ib
        self.matriz_interbancaria -= repayment_ib

        # --- B. DIVIDENDS ---

        # 1. Firms
        # Profit Proxy: Positive Liquidity
        firm_liq = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]
        distributable_f = np.maximum(0, firm_liq)
        dividends_f = distributable_f * Parametros.DIVIDEND_RATIO

        # Deduct
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= dividends_f

        # Distribute to Owners (Vectorized)
        # Households with OWNED_TYPE == 1 (Firms)
        hh_is_firm_owner = self.estado_hogares[:, Parametros.IDX_HH_OWNED_TYPE] == 1
        # Get the ID of the firm they own
        owned_firm_idx = self.estado_hogares[
            hh_is_firm_owner, Parametros.IDX_HH_OWNED_ENTITY_IDX
        ].astype(int)

        # Add to deposits
        self.estado_hogares[hh_is_firm_owner, Parametros.IDX_HH_DEPOSITS] += (
            dividends_f[owned_firm_idx]
        )

        # 2. Banks
        bank_liq = self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
        bank_eq = self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]

        # Distributable = Positive Equity
        distributable_b = np.maximum(0, bank_eq)
        dividends_b = distributable_b * Parametros.DIVIDEND_RATIO

        # Cap at Liquidity
        dividends_b = np.minimum(dividends_b, bank_liq)
        dividends_b = np.maximum(0, dividends_b)

        # Deduct
        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] -= dividends_b
        self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= dividends_b

        # Distribute to Owners (Vectorized)
        # Households with OWNED_TYPE == 2 (Banks)
        hh_is_bank_owner = self.estado_hogares[:, Parametros.IDX_HH_OWNED_TYPE] == 2
        owned_bank_idx = self.estado_hogares[
            hh_is_bank_owner, Parametros.IDX_HH_OWNED_ENTITY_IDX
        ].astype(int)

        self.estado_hogares[hh_is_bank_owner, Parametros.IDX_HH_DEPOSITS] += (
            dividends_b[owned_bank_idx]
        )

        # --- C. BANKRUPTCIES & CASCADES ---

        # 1. Firm Bankruptcy
        # Liquidity < 0
        dead_firms_mask = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] < 0
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
            self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= bank_losses
            # Record Bad Debt (Optional)
            self.estado_bancos[:, Parametros.IDX_BANK_BAD_DEBT] += bank_losses

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

            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_LIQUIDITY] = (
                init_liq
            )
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_EQUITY] = (
                init_liq
            )
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_PRICE] = (
                init_price
            )
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_LEVERAGE] = 0.0
            # Reset Demand to 0 so it re-initializes
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_DEMAND] = 0.0

        # 2. Bank Default Cascades
        processed_mask = np.zeros(Parametros.B, dtype=bool)

        while True:
            # Current dead banks
            current_equity = self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
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
                # Row `dead_bank` = Amounts `dead_bank` OWES to others (Cols).

                obligations = self.matriz_interbancaria[dead_bank, :]  # (B,)

                # Others lose this Equity
                self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= obligations

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

            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_TOTAL_ASSETS] = (
                init_assets
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_EQUITY] = (
                init_assets * Parametros.INIT_CAPITAL_RATIO
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_LIQUIDITY] = (
                init_assets
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_DEPOSITS] = (
                init_assets * (1 - Parametros.INIT_CAPITAL_RATIO)
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_BAD_DEBT] = 0.0

            # Clear Connections (Lending side)
            # We already cleared Borrowing side (Rows).
            # Now clear Lending side (Cols). Dead banks cannot claim assets.
            # (Strictly speaking, liquidation value > 0, but simplified to 0 here for worst-case)
            self.matriz_interbancaria[:, all_dead_ids] = 0.0
            self.matriz_credito_firmas[:, all_dead_ids] = 0.0  # Clear firm loans too

    def ejecutar_paso(self):
        """Execute one simulacion step."""
        # Reset Per-Step Accumulators
        self.current_step_volume = 0.0
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        # Run Phases
        self.paso_planificacion_firmas()
        self.paso_mercado_bancario()
        self.paso_economia_real()
        self.paso_contabilidad()

        # Traceability
        self.registrar_historia()

