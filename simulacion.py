import numpy as np
from parametros import Parametros
import funciones as fn


class Modelo_CRISIS:
    """
    Vectorized implementation of the Poledna & Thurner (2016) ABM model.
    Phase 1: Data Architecture & Initialization (Full Graph Observability).
    Phase 2: Firms Planning & Labor Market Dynamics.
    Phase 3: Interbank Market & SRT Logic (Debug Mode).
    """

    def __init__(self, seed=None, tax_mode="none", tax_param=0.0):
        self.rng = np.random.default_rng(seed)

        # Experiment Logic
        self.tax_mode = tax_mode.lower()
        self.tax_param = tax_param

        # Metrics for Analysis
        self.current_step_loss = 0.0
        self.current_step_defaults = 0
        self.current_step_volume = 0.0
        self.current_firm_credit_demand = np.zeros(Parametros.F, dtype=np.float64)

        self.reset()

    def reset(self):
        """
        Reset or initialize all state tensors and network topologies for a new simulation run.
        Now includes persistent matrices for ALL interaction layers.
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
        self.estado_bancos[:, Parametros.IDX_BANK_OPERATING_COST_CHI] = (
            self.rng.uniform(Parametros.CHI_RANGE[0], Parametros.CHI_RANGE[1], size=B)
        )
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

        # --- 3. Topology & Relationships (Matrices) ---
        
        # A. Interbank Network (B x B) - Dynamic
        self.matriz_interbancaria = np.zeros((B, B), dtype=np.float64)
        
        # B. Credit Network (Firm -> Bank) (F x B) - Dynamic
        self.matriz_credito_firmas = np.zeros((F, B), dtype=np.float64)
        
        # C. Labor Network (Household -> Firm) (H x F) - Static (initially)
        # Each household has one employer.
        self.matriz_laboral = np.zeros((H, F), dtype=np.int8)
        employer_indices = self.rng.integers(0, F, size=H)
        self.matriz_laboral[np.arange(H), employer_indices] = 1

        # D. Deposit Network (Household -> Bank) (H x B) - Static (initially)
        # Each household has one bank.
        self.matriz_depositos = np.zeros((H, B), dtype=np.int8)
        bank_indices = self.rng.integers(0, B, size=H)
        self.matriz_depositos[np.arange(H), bank_indices] = 1

        # E. Ownership Networks (H x F, H x B) - Static
        self.matriz_propiedad_firmas = np.zeros((H, F), dtype=np.int8)
        self.matriz_propiedad_bancos = np.zeros((H, B), dtype=np.int8)

        hh_indices = np.arange(H)
        self.rng.shuffle(hh_indices)

        # First F households -> Firm Owners
        firm_owners = hh_indices[:F]
        self.matriz_propiedad_firmas[firm_owners, np.arange(F)] = 1

        # Next B households -> Bank Owners
        bank_owners = hh_indices[F : F + B]
        self.matriz_propiedad_bancos[bank_owners, np.arange(B)] = 1
        
        # F. Consumption Network (Household -> Firm) (H x F) - Dynamic per step
        # Stores the MONETARY VALUE of consumption.
        self.matriz_consumo = np.zeros((H, F), dtype=np.float64)

        # --- 4. History / Traceability ---
        self.step_buffer = {
            "matriz_interbancaria": [],
            "matriz_credito_firmas": [],
            "matriz_consumo": [],
            "matriz_laboral": [],
            "matriz_depositos": [],
            "matriz_propiedad_firmas": [],
            "matriz_propiedad_bancos": [],
            "estado_bancos": [],
            "estado_firmas": [],
            "estado_hogares": [],
        }

        self.registrar_historia()

    def registrar_historia(self):
        """Append current state snapshots to step_buffer."""
        # Topologies
        self.step_buffer["matriz_interbancaria"].append(self.matriz_interbancaria.astype(np.float32).copy())
        self.step_buffer["matriz_credito_firmas"].append(self.matriz_credito_firmas.astype(np.float32).copy())
        self.step_buffer["matriz_consumo"].append(self.matriz_consumo.astype(np.float32).copy())
        
        # These are technically static or semi-static, but for full reconstruction we save them.
        self.step_buffer["matriz_laboral"].append(self.matriz_laboral.astype(np.int8).copy())
        self.step_buffer["matriz_depositos"].append(self.matriz_depositos.astype(np.int8).copy())
        self.step_buffer["matriz_propiedad_firmas"].append(self.matriz_propiedad_firmas.astype(np.int8).copy())
        self.step_buffer["matriz_propiedad_bancos"].append(self.matriz_propiedad_bancos.astype(np.int8).copy())

        # States
        self.step_buffer["estado_bancos"].append(self.estado_bancos.astype(np.float32).copy())
        self.step_buffer["estado_firmas"].append(self.estado_firmas.astype(np.float32).copy())
        self.step_buffer["estado_hogares"].append(self.estado_hogares.astype(np.float32).copy())

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

    def actualizar_mercado_laboral(self):
        """
        Syncs the labor matrix (H x F) with the target worker count per firm.
        Fires workers if Overstaffed. Hires workers if Understaffed.
        """
        # 1. Current state
        current_workers = np.sum(self.matriz_laboral, axis=0) # (F,)
        target_workers = self.estado_firmas[:, Parametros.IDX_FIRM_WORKERS].astype(int)
        
        delta = target_workers - current_workers
        
        # Identify firms that need change
        
        # --- FIRING (delta < 0) ---
        firing_firms = np.where(delta < 0)[0]
        for f in firing_firms:
            n_fire = abs(delta[f])
            # Find current employees: (H,) boolean mask -> indices
            employee_indices = np.where(self.matriz_laboral[:, f] == 1)[0]
            
            if len(employee_indices) > 0:
                n_fire = min(n_fire, len(employee_indices))
                fired_indices = self.rng.choice(employee_indices, size=n_fire, replace=False)
                # Update Matrix
                self.matriz_laboral[fired_indices, f] = 0
                
        # --- HIRING (delta > 0) ---
        hiring_firms = np.where(delta > 0)[0]
        
        # Identify unemployed pool (Dynamic based on firings just happened)
        # Sum rows: if 0, unemployed.
        employment_status = np.sum(self.matriz_laboral, axis=1)
        unemployed_indices = np.where(employment_status == 0)[0]
        
        # Shuffle unemployed pool once
        self.rng.shuffle(unemployed_indices)
        pool_ptr = 0
        total_unemployed = len(unemployed_indices)
        
        for f in hiring_firms:
            n_hire = delta[f]
            
            # Check availability
            remaining_in_pool = total_unemployed - pool_ptr
            if remaining_in_pool <= 0:
                break # No more workers
            
            # Hire
            actual_hire = min(n_hire, remaining_in_pool)
            new_hires = unemployed_indices[pool_ptr : pool_ptr + actual_hire]
            
            # Update Matrix
            self.matriz_laboral[new_hires, f] = 1
            pool_ptr += actual_hire

        # Final Sync: Update Firm State to match actual Matrix (Rationing)
        real_workers = np.sum(self.matriz_laboral, axis=0)
        self.estado_firmas[:, Parametros.IDX_FIRM_WORKERS] = real_workers
        self.estado_firmas[:, Parametros.IDX_FIRM_PROD] = real_workers * Parametros.alpha
        self.estado_firmas[:, Parametros.IDX_FIRM_WAGES] = real_workers * Parametros.WAGE

    def paso_planificacion_firmas(self):
        """
        Phase 2: Firms Planning.
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
        if p_avg > 1e-9:
            adjustment = Parametros.PRICE_ADJUSTMENT_SPEED * (p_avg - prices) / p_avg
            new_prices = prices + prices * (adjustment + noise)
        else:
            new_prices = prices * (1 + noise)

        new_prices = np.maximum(new_prices, 0.01)
        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE] = new_prices

        # --- 2. Update Demand Expectations ---
        self.estado_firmas[:, Parametros.IDX_FIRM_DEMAND_PREV] = self.estado_firmas[
            :, Parametros.IDX_FIRM_DEMAND
        ].copy()

        current_demand = self.estado_firmas[:, Parametros.IDX_FIRM_DEMAND]
        if np.all(current_demand == 0):
            current_demand = self.rng.uniform(10, 50, size=Parametros.F)

        # Random Walk
        demand_shock = self.rng.normal(0, 0.05, size=Parametros.F)
        new_demand = current_demand * (1 + demand_shock)
        new_demand = np.maximum(new_demand, 0.0)

        self.estado_firmas[:, Parametros.IDX_FIRM_DEMAND] = new_demand

        # --- 3. Production Planning ---
        labor_needed = np.ceil(new_demand / Parametros.alpha)

        self.estado_firmas[:, Parametros.IDX_FIRM_WORKERS] = labor_needed
        # Initial estimate of production/wages (Target)
        self.estado_firmas[:, Parametros.IDX_FIRM_PROD] = (
            labor_needed * Parametros.alpha
        )
        wage_bill = labor_needed * Parametros.WAGE
        self.estado_firmas[:, Parametros.IDX_FIRM_WAGES] = wage_bill

        # --- 4. Labor Market Dynamics (Sync Matrix) ---
        # This updates Workers, Prod, and Wages to REALITY (Rationing applied)
        self.actualizar_mercado_laboral()
        
        # Re-fetch REAL wage bill for credit demand
        wage_bill = self.estado_firmas[:, Parametros.IDX_FIRM_WAGES]

        # --- 5. Financial Health Update (Leverage) ---
        current_debt = np.sum(self.matriz_credito_firmas, axis=1)  # (F,)
        liquidity = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]

        leverage = current_debt / (liquidity + 1e-9)
        self.estado_firmas[:, Parametros.IDX_FIRM_LEVERAGE] = leverage

        # --- 6. Credit Demand Calculation ---
        gap = wage_bill - liquidity
        credit_demand = np.maximum(gap, 0.0)

        self.current_firm_credit_demand = credit_demand

    def paso_mercado_bancario(self):
        """
        Phase 3: Credit Market (Firms-Banks) & Interbank Market (SRT).
        Includes DEBUG PRINTS for auditing flow.
        """
        # --- PART A: CREDIT MARKET (Firms -> Banks) ---

        pool_indices = self.rng.integers(
            0, Parametros.B, size=(Parametros.F, Parametros.N_SEARCH)
        )

        bank_chi_pool = self.estado_bancos[
            pool_indices, Parametros.IDX_BANK_OPERATING_COST_CHI
        ]

        firm_leverage = self.estado_firmas[:, Parametros.IDX_FIRM_LEVERAGE]
        firm_leverage_broad = firm_leverage[:, np.newaxis]

        firm_fragility = fn.calcular_fragilidad_financiera(
            firm_leverage_broad, Parametros.K_mu
        )

        rates = fn.calcular_tasa_firma(Parametros.R_BAR, bank_chi_pool, firm_fragility)
        rates += self.rng.uniform(0, 1e-6, size=rates.shape)

        best_choice_local_idx = np.argmin(rates, axis=1)  # (F,)
        chosen_bank_ids = pool_indices[np.arange(Parametros.F), best_choice_local_idx]

        bank_credit_demand = np.zeros(Parametros.B)
        np.add.at(bank_credit_demand, chosen_bank_ids, self.current_firm_credit_demand)

        # --- PART B: INTERBANK MARKET & SRT ---

        bank_liquidity = self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
        gaps = bank_credit_demand - bank_liquidity

        deficit_ids = np.where(gaps > 0)[0]
        surplus_ids = np.where(gaps < 0)[0]
        
        # print(f"DEBUG [Step]: Deficit Banks: {len(deficit_ids)}, Surplus Banks: {len(surplus_ids)}")

        total_taxes = 0.0

        if len(deficit_ids) > 0 and len(surplus_ids) > 0:
            n_def = len(deficit_ids)
            n_sur = len(surplus_ids)

            d_indices = np.repeat(deficit_ids, n_sur)
            s_indices = np.tile(surplus_ids, n_def)

            psi_lender = self.estado_bancos[
                s_indices, Parametros.IDX_BANK_INTERBANK_COST_PSI
            ]

            current_ib_borrowing = np.sum(self.matriz_interbancaria, axis=1)
            current_deposits = self.estado_bancos[:, Parametros.IDX_BANK_DEPOSITS]
            equity = self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]

            bank_leverage = (current_deposits + current_ib_borrowing) / (equity + 1e-9)
            lev_borrower = bank_leverage[d_indices]
            borrower_fragility = fn.calcular_fragilidad_financiera(
                lev_borrower, Parametros.K_mu
            )

            base_rates = fn.calcular_tasa_interbancaria(
                Parametros.R_BAR, psi_lender, borrower_fragility
            )

            current_surpluses = -gaps[s_indices]
            current_deficits = gaps[d_indices]
            potential_amounts = np.minimum(current_deficits, current_surpluses)

            taxes = np.zeros(len(potential_amounts))

            if self.tax_mode == "srt" and self.tax_param > 0:
                v = self.estado_bancos[:, Parametros.IDX_BANK_TOTAL_ASSETS]
                p_default = 0.01 * fn.calcular_fragilidad_financiera(
                    bank_leverage, Parametros.K_mu
                )
                proposed_indices = np.column_stack((d_indices, s_indices))
                taxes = fn.calcular_impuesto_srt(
                    self.matriz_interbancaria,
                    proposed_indices,
                    potential_amounts,
                    equity,
                    v,
                    p_default,
                    self.tax_param,
                )
            elif self.tax_mode == "tobin":
                taxes = potential_amounts * self.tax_param

            interest_costs = potential_amounts * base_rates
            total_costs = interest_costs + taxes

            unit_costs = np.zeros_like(total_costs)
            mask_amt = potential_amounts > 1e-9
            unit_costs[mask_amt] = total_costs[mask_amt] / potential_amounts[mask_amt]

            sorted_indices = np.argsort(unit_costs)
            dyn_gaps = gaps.copy()

            for idx in sorted_indices:
                d = d_indices[idx]
                s = s_indices[idx]

                if dyn_gaps[d] <= 1e-9 or dyn_gaps[s] >= -1e-9:
                    continue

                amount = min(dyn_gaps[d], -dyn_gaps[s])
                if amount < 1e-9:
                    continue

                self.matriz_interbancaria[d, s] += amount
                self.current_step_volume += amount

                self.estado_bancos[d, Parametros.IDX_BANK_LIQUIDITY] += amount
                self.estado_bancos[s, Parametros.IDX_BANK_LIQUIDITY] -= amount

                pre_amt = potential_amounts[idx]
                pre_tax = taxes[idx]
                actual_tax = 0.0
                if pre_amt > 1e-9:
                    actual_tax = pre_tax * (amount / pre_amt)
                
                if actual_tax > 0:
                    self.estado_bancos[d, Parametros.IDX_BANK_EQUITY] -= actual_tax
                    total_taxes += actual_tax

                dyn_gaps[d] -= amount
                dyn_gaps[s] += amount
        
        # print(f"DEBUG [Step]: Total Interbank Volume: {self.current_step_volume:.2f}, Taxes: {total_taxes:.4f}")

        # --- PART C: DISBURSEMENT TO FIRMS ---
        final_liquidity = self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
        payout_ratios = np.ones(Parametros.B)
        mask_demand = bank_credit_demand > 1e-9
        np.divide(
            final_liquidity, bank_credit_demand, out=payout_ratios, where=mask_demand
        )
        payout_ratios = np.minimum(1.0, payout_ratios)

        approved_amounts = (
            self.current_firm_credit_demand * payout_ratios[chosen_bank_ids]
        )

        f_indices = np.arange(Parametros.F)
        self.matriz_credito_firmas[f_indices, chosen_bank_ids] += approved_amounts

        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] += approved_amounts
        np.add.at(
            self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY],
            chosen_bank_ids,
            -approved_amounts,
        )

    def paso_economia_real(self):
        """
        Phase 3: Real Economy (Production, Wages, Consumption).
        UPDATED: Uses Matrix Architecture.
        """
        # --- A. PRODUCTION & WAGES ---

        # 1. Production
        # (Assuming perishable goods or instant conversion, kept simple as before)

        # 2. Wage Payment
        wage_bills = self.estado_firmas[:, Parametros.IDX_FIRM_WAGES]
        firm_liquidity = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]
        payments = np.minimum(wage_bills, firm_liquidity)

        # Deduct from Firms
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= payments

        # Distribute to Households (Workers) using MATRIZ_LABORAL
        # Count employees per firm: Sum columns of Matrix (H x F) -> (F,)
        employee_counts = np.sum(self.matriz_laboral, axis=0)

        # Wage per worker
        wage_per_worker = np.zeros(Parametros.F)
        mask_c = employee_counts > 0
        np.divide(payments, employee_counts, out=wage_per_worker, where=mask_c)

        # HH Income = Matriz_Laboral @ Wage_Per_Worker
        # (H x F) @ (F,) -> (H,)
        hh_income = self.matriz_laboral @ wage_per_worker
        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_income

        # --- B. CONSUMPTION MARKET ---

        # 1. Budget
        hh_deposits = self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS]
        budgets = hh_deposits * Parametros.c

        # 2. Firm Selection (Z-Search)
        z_indices = self.rng.integers(
            0, Parametros.F, size=(Parametros.H, Parametros.Z_CONSUMPTION)
        )
        prices_options = self.estado_firmas[z_indices, Parametros.IDX_FIRM_PRICE]
        winner_local_indices = np.argmin(prices_options, axis=1)
        winner_global_indices = z_indices[np.arange(Parametros.H), winner_local_indices]

        # 3. Aggregate Demand
        demand_monetary = np.bincount(
            winner_global_indices, weights=budgets, minlength=Parametros.F
        )

        # 4. Sales & Rationing
        firm_prices = self.estado_firmas[:, Parametros.IDX_FIRM_PRICE]
        firm_inventory = self.estado_firmas[:, Parametros.IDX_FIRM_PROD]
        max_revenue = firm_inventory * firm_prices
        actual_revenue = np.minimum(demand_monetary, max_revenue)

        # Sales Quantity
        sales_qty = np.zeros(Parametros.F)
        price_mask = firm_prices > 1e-9
        np.divide(actual_revenue, firm_prices, out=sales_qty, where=price_mask)

        # Update Firms
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] += actual_revenue
        self.estado_firmas[:, Parametros.IDX_FIRM_PROD] -= sales_qty

        # 5. Households Expenditure & RECORDING (Matriz Consumo)
        # Clear previous step consumption
        self.matriz_consumo.fill(0.0)

        scale_factors = np.ones(Parametros.F)
        demand_mask = demand_monetary > 1e-9
        np.divide(actual_revenue, demand_monetary, out=scale_factors, where=demand_mask)
        scale_factors = np.minimum(1.0, scale_factors)

        hh_scale = scale_factors[winner_global_indices]
        hh_expenditure = budgets * hh_scale

        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] -= hh_expenditure

        # Record in Matrix: Rows=Households, Cols=Firms
        # We use add.at for vectorized accumulation (though indices are unique per HH here)
        np.add.at(
            self.matriz_consumo,
            (np.arange(Parametros.H), winner_global_indices),
            hh_expenditure,
        )

    def paso_contabilidad(self):
        """
        Phase 4: Accounting, Bankruptcy, and Resets.
        UPDATED: Uses Matrix Architecture.
        """
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        # --- A. DEBT REPAYMENT ---
        tau = Parametros.DEBT_REPAYMENT_RATE

        # Firms -> Banks
        repayment_firms = self.matriz_credito_firmas * tau
        total_pay_firm = np.sum(repayment_firms, axis=1)
        total_receive_bank = np.sum(repayment_firms, axis=0)

        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= total_pay_firm
        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] += total_receive_bank
        self.matriz_credito_firmas -= repayment_firms

        # Interbank
        repayment_ib = self.matriz_interbancaria * tau
        total_pay_bank = np.sum(repayment_ib, axis=1)
        total_receive_bank_ib = np.sum(repayment_ib, axis=0)

        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] -= total_pay_bank
        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] += total_receive_bank_ib
        self.matriz_interbancaria -= repayment_ib

        # --- B. DIVIDENDS ---
        # 1. Firms
        firm_liq = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY]
        distributable_f = np.maximum(0, firm_liq)
        dividends_f = distributable_f * Parametros.DIVIDEND_RATIO
        self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] -= dividends_f

        # Distribute to Owners using MATRIX (H x F)
        # HH Income = Matriz_Propiedad @ Dividends (F,)
        hh_div_income_f = self.matriz_propiedad_firmas @ dividends_f
        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_div_income_f

        # 2. Banks
        bank_liq = self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
        bank_eq = self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
        distributable_b = np.maximum(0, bank_eq)
        dividends_b = distributable_b * Parametros.DIVIDEND_RATIO
        dividends_b = np.minimum(dividends_b, bank_liq)
        dividends_b = np.maximum(0, dividends_b)

        self.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY] -= dividends_b
        self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= dividends_b

        # Distribute using MATRIX (H x B)
        hh_div_income_b = self.matriz_propiedad_bancos @ dividends_b
        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] += hh_div_income_b

        # --- C. BANKRUPTCIES & CASCADES ---

        # 1. Firm Bankruptcy
        dead_firms_mask = self.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] < 0
        dead_firms_indices = np.where(dead_firms_mask)[0]

        if len(dead_firms_indices) > 0:
            bad_loans = self.matriz_credito_firmas[dead_firms_indices, :]
            bank_losses = np.sum(bad_loans, axis=0)
            self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= bank_losses
            self.estado_bancos[:, Parametros.IDX_BANK_BAD_DEBT] += bank_losses

            self.matriz_credito_firmas[dead_firms_indices, :] = 0.0

            # Reset Firms
            n_dead = len(dead_firms_indices)
            init_liq = self.rng.uniform(
                Parametros.INIT_FIRM_ASSETS[0],
                Parametros.INIT_FIRM_ASSETS[1],
                size=n_dead,
            )
            init_price = Parametros.WAGE / Parametros.alpha

            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_LIQUIDITY] = init_liq
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_EQUITY] = init_liq
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_PRICE] = init_price
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_LEVERAGE] = 0.0
            self.estado_firmas[dead_firms_indices, Parametros.IDX_FIRM_DEMAND] = 0.0
            
            # NOTE: We do NOT reset ownership matrices. Dead firms are just "restructured" 
            # and old owners keep shares (or we assume new equity injection comes from same owners).
            # This keeps the graph structure stable as per standard ABM simplification.

        # 2. Bank Default Cascades
        processed_mask = np.zeros(Parametros.B, dtype=bool)

        while True:
            current_equity = self.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
            dead_mask = current_equity < 0
            new_defaults = dead_mask & (~processed_mask)
            new_default_ids = np.where(new_defaults)[0]

            if len(new_default_ids) == 0:
                break

            for dead_bank in new_default_ids:
                obligations = self.matriz_interbancaria[dead_bank, :]
                self.estado_bancos[:, Parametros.IDX_BANK_EQUITY] -= obligations
                loss_val = np.sum(obligations)
                self.current_step_loss += loss_val
                self.matriz_interbancaria[dead_bank, :] = 0.0

            self.current_step_defaults += len(new_default_ids)
            processed_mask[new_default_ids] = True

        # 3. Bailout / Reset
        all_dead_ids = np.where(processed_mask)[0]
        if len(all_dead_ids) > 0:
            n_dead_b = len(all_dead_ids)
            init_assets = self.rng.uniform(
                Parametros.INIT_BANK_ASSETS[0],
                Parametros.INIT_BANK_ASSETS[1],
                size=n_dead_b,
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_TOTAL_ASSETS] = init_assets
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_EQUITY] = (
                init_assets * Parametros.INIT_CAPITAL_RATIO
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_LIQUIDITY] = init_assets
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_DEPOSITS] = (
                init_assets * (1 - Parametros.INIT_CAPITAL_RATIO)
            )
            self.estado_bancos[all_dead_ids, Parametros.IDX_BANK_BAD_DEBT] = 0.0

            self.matriz_interbancaria[:, all_dead_ids] = 0.0
            self.matriz_credito_firmas[:, all_dead_ids] = 0.0

    def ejecutar_paso(self):
        """Execute one simulacion step."""
        self.current_step_volume = 0.0
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        self.paso_planificacion_firmas()
        self.paso_mercado_bancario()
        self.paso_economia_real()
        self.paso_contabilidad()

        self.registrar_historia()