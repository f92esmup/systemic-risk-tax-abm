import numpy as np
from parametros import Parametros
import funciones as fn

def ejecutar_paso2(modelo):
    """
    Paso 2: Mercado Bancario (Liquidez y Préstamos).
    - Refactorizado para cumplir con Poledna & Thurner (2016).
    1. Bancos estiman costos de refinanciación (incluyendo Impuesto SRT).
    2. Firmas eligen banco (Market Friction: Lowest Rate).
    3. Mercado Interbancario Secuencial (Agent-based, no Global Opt).
    4. Desembolso y actualización.
    """
    
    # --- 1. PRE-CALCULATION & ESTIMATION ---
    # Banks estimate if they need liquidity. 
    # Simple heuristic: If Liquidity is low relative to Assets/Previous Demand, 
    # they price in the marginal cost of borrowing.
    
    # For simplicity, we assume all banks check "Market Conditions"
    # and if they were to borrow, what would be the tax.
    # We calculate a "Benchmark Tax" for each bank if it were to borrow from the system.
    
    bank_liquidity = modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
    equity = modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
    
    # Estimated Marginal Cost (Interbank Rate + Tax)
    estimated_marginal_cost = np.zeros(Parametros.B)
    
    # Identify likely deficit banks (Low Liquidity)
    # Heuristic: Banks with Liquidity < 10% of Assets might need cash
    # Or just calculate for all to be safe (vectorized, it's fast enough for B=100)
    
    # We need a "Representative Lender" to estimate tax. 
    # We use the bank with most liquidity as the hypothetical lender.
    rich_bank_idx = np.argmax(bank_liquidity)
    
    # Setup Batch for Estimation: All banks borrowing from `rich_bank_idx`
    borrowers = np.arange(Parametros.B)
    lenders = np.full(Parametros.B, rich_bank_idx)
    # Filter out self-loops
    valid_mask = borrowers != rich_bank_idx
    
    if np.any(valid_mask) and modelo.tax_mode == 'srt' and modelo.tax_param > 0:
        b_idxs = borrowers[valid_mask]
        l_idxs = lenders[valid_mask]
        
        # Hypothetical Amount: Small unit to test sensitivity (e.g. 1 unit or mean loan)
        amount = 1.0 
        
        # Current Metrics
        v = np.sum(modelo.matriz_interbancaria, axis=1)
        curr_L = modelo.matriz_interbancaria
        
        # Leverage/Fragility for Default Prob
        current_deposits = modelo.estado_bancos[:, Parametros.IDX_BANK_DEPOSITS]
        curr_ib_borrowing = np.sum(curr_L, axis=1)
        curr_leverage = (current_deposits + curr_ib_borrowing) / (equity + 1e-9)
        p_default = Parametros.DEFAULT_PROB_SCALING * fn.calcular_fragilidad_financiera(
            curr_leverage, Parametros.K_mu
        )
        
        prop_indices = np.column_stack((b_idxs, l_idxs))
        prop_amounts = np.full(len(b_idxs), amount)
        
        # Calculate Taxes
        est_taxes = fn.calcular_impuesto_srt(
            curr_L, prop_indices, prop_amounts,
            equity, v, p_default, modelo.tax_param
        )
        
        # Normalize to rate (Tax per unit)
        est_tax_rate = est_taxes / amount
        estimated_marginal_cost[b_idxs] = est_tax_rate

    # Add Base Interbank Rate estimate
    # r_ib = r_bar * (1 + psi * mu(borrower))
    # We use the borrower's own fragility
    current_deposits = modelo.estado_bancos[:, Parametros.IDX_BANK_DEPOSITS]
    curr_ib_borrowing = np.sum(modelo.matriz_interbancaria, axis=1)
    est_leverage = (current_deposits + curr_ib_borrowing) / (equity + 1e-9)
    est_fragility = fn.calcular_fragilidad_financiera(est_leverage, Parametros.K_mu)
    
    # Assume average lender Psi
    avg_psi = np.mean(modelo.estado_bancos[:, Parametros.IDX_BANK_INTERBANK_COST_PSI])
    est_ib_rate = fn.calcular_tasa_interbancaria(Parametros.R_BAR, avg_psi, est_fragility)
    
    estimated_marginal_cost += est_ib_rate

    # --- 2. FIRM SELECTS BANK ---

    pool_indices = modelo.rng.integers(
        0, Parametros.B, size=(Parametros.F, Parametros.N_SEARCH)
    )

    bank_chi_pool = modelo.estado_bancos[
        pool_indices, Parametros.IDX_BANK_OPERATING_COST_CHI
    ]
    
    # Get estimated costs for the pool
    bank_est_costs = estimated_marginal_cost[pool_indices]

    firm_leverage = modelo.estado_firmas[:, Parametros.IDX_FIRM_LEVERAGE]
    firm_leverage_broad = firm_leverage[:, np.newaxis]

    firm_fragility = fn.calcular_fragilidad_financiera(
        firm_leverage_broad, Parametros.K_mu
    )

    # Base Rate + Bank Markup (Est. Cost)
    rates = fn.calcular_tasa_firma(Parametros.R_BAR, bank_chi_pool, firm_fragility)
    
    # Logic: If bank has plenty of liquidity, it might not pass full marginal cost.
    # But for simplicity and safety, they price it in.
    rates += bank_est_costs
    
    # Noise/Friction
    rates += modelo.rng.uniform(0, 1e-6, size=rates.shape)

    best_choice_local_idx = np.argmin(rates, axis=1)  # (F,)
    chosen_bank_ids = pool_indices[np.arange(Parametros.F), best_choice_local_idx]

    # Register Demand
    bank_credit_demand = np.zeros(Parametros.B)
    np.add.at(bank_credit_demand, chosen_bank_ids, modelo.current_firm_credit_demand)

    # --- 3. SEQUENTIAL INTERBANK MARKET ---
    
    bank_liquidity = modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
    gaps = bank_credit_demand - bank_liquidity

    # Identify Initial Deficit/Surplus
    deficit_ids = np.where(gaps > 0)[0]
    surplus_ids = np.where(gaps < -1e-9)[0] # Strict surplus
    
    # Shuffle for random entry
    modelo.rng.shuffle(deficit_ids)
    
    total_taxes = 0.0
    bank_actual_refinancing_costs = np.zeros(Parametros.B) # Total cost (Interest + Tax)

    # Dynamic tracking
    current_gaps = gaps.copy()
    current_surpluses = np.zeros(Parametros.B)
    current_surpluses[surplus_ids] = -gaps[surplus_ids]

    for d in deficit_ids:
        needed = current_gaps[d]
        if needed <= 1e-9:
            continue
            
        # Bank d looks for lenders
        # Potential lenders: anyone with surplus > 0
        potential_lenders = np.where(current_surpluses > 1e-9)[0]
        if len(potential_lenders) == 0:
            break # No liquidity left in system
            
        # 3a. Calculate costs for ALL potential lenders
        # Rate = r_bar * (1 + psi_lender * mu(borrower))
        # Tax = SRT(d, lender, amount)
        
        # Vectorized calculation for this specific borrower 'd' against all 'lenders'
        lenders_psi = modelo.estado_bancos[potential_lenders, Parametros.IDX_BANK_INTERBANK_COST_PSI]
        
        # Borrower fragility (dynamic based on current state? stick to start of step for stability or update?)
        # Let's use current state
        curr_L = modelo.matriz_interbancaria
        curr_E = modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
        curr_v = np.sum(curr_L, axis=1)
        curr_deposits = modelo.estado_bancos[:, Parametros.IDX_BANK_DEPOSITS]
        curr_lev = (curr_deposits + np.sum(curr_L, axis=1)) / (curr_E + 1e-9)
        d_fragility = fn.calcular_fragilidad_financiera(curr_lev[d], Parametros.K_mu)
        
        base_rates = fn.calcular_tasa_interbancaria(Parametros.R_BAR, lenders_psi, d_fragility)
        
        # Max amount available from each lender
        available_amts = current_surpluses[potential_lenders]
        # Bank d wants 'needed', but can only take min(needed, available)
        # However, for rate calculation, the tax depends on amount.
        # We calculate tax for the FULL needed amount (or max available) to compare prices per unit.
        amounts_to_test = np.minimum(needed, available_amts)
        
        # Calculate Taxes
        taxes = np.zeros(len(potential_lenders))
        if modelo.tax_mode == 'srt' and modelo.tax_param > 0:
            p_default = Parametros.DEFAULT_PROB_SCALING * fn.calcular_fragilidad_financiera(
                curr_lev, Parametros.K_mu
            )
            # Batch: d -> all lenders
            prop_indices = np.column_stack((np.full(len(potential_lenders), d), potential_lenders))
            
            taxes = fn.calcular_impuesto_srt(
                curr_L, prop_indices, amounts_to_test,
                curr_E, curr_v, p_default, modelo.tax_param
            )
        elif modelo.tax_mode == 'tobin':
            taxes = amounts_to_test * modelo.tax_param
            
        # Calculate Unit Costs
        total_costs = (amounts_to_test * base_rates) + taxes
        unit_costs = np.divide(total_costs, amounts_to_test, out=np.zeros_like(total_costs), where=amounts_to_test>1e-9)
        
        # Sort lenders by Unit Cost
        sorted_lender_idxs = np.argsort(unit_costs)
        
        # 3b. Execute trades
        for idx in sorted_lender_idxs:
            lender = potential_lenders[idx]
            amount = amounts_to_test[idx] # This was min(needed, available)
            
            if amount < 1e-9:
                continue
                
            # Re-check actual tax (since we might have partially filled gap with previous lender?)
            # Actually, amounts_to_test was based on 'needed'. If 'needed' decreases, we should recalc.
            # But strictly, we fill sequentially.
            # 'amounts_to_test' was valid for the start of the loop.
            # We iterate:
            
            actual_amount = min(current_gaps[d], current_surpluses[lender])
            if actual_amount < 1e-9:
                continue

            # Calculate Final Tax for this transaction
            actual_tax = 0.0
            if modelo.tax_mode == 'srt' and modelo.tax_param > 0:
                # Need to recalc because 'curr_L' changed if we did multiple trades? 
                # Yes. Sequential updates matter for DebtRank.
                curr_L_dyn = modelo.matriz_interbancaria
                curr_E_dyn = modelo.estado_bancos[:, Parametros.IDX_BANK_EQUITY]
                curr_v_dyn = np.sum(curr_L_dyn, axis=1)
                
                # Update leverage/p_default? Maybe too expensive. Use base p_default from start of step
                # to save compute, but update L/v/E.
                
                prop_indices = np.array([[d, lender]], dtype=int)
                prop_amounts = np.array([actual_amount])
                
                # We use the initial p_default for stability or recompute?
                # Paper: "iterative". Let's use current.
                curr_lev_dyn = (curr_deposits + np.sum(curr_L_dyn, axis=1)) / (curr_E_dyn + 1e-9)
                p_def_dyn = Parametros.DEFAULT_PROB_SCALING * fn.calcular_fragilidad_financiera(curr_lev_dyn, Parametros.K_mu)

                tax_res = fn.calcular_impuesto_srt(
                    curr_L_dyn, prop_indices, prop_amounts,
                    curr_E_dyn, curr_v_dyn, p_def_dyn, modelo.tax_param
                )
                if len(tax_res) > 0:
                    actual_tax = tax_res[0]

            elif modelo.tax_mode == 'tobin':
                actual_tax = actual_amount * modelo.tax_param

            # Execute
            modelo.matriz_interbancaria[d, lender] += actual_amount
            modelo.current_step_volume += actual_amount
            
            rate_final = base_rates[idx] # Use the rate quoted (assuming it doesn't change wildly)
            
            # Update Rate Tracking (Avg)
            prev_amt = modelo.matriz_interbancaria[d, lender] - actual_amount
            prev_rate = modelo.matriz_tasas_interbancaria[d, lender]
            if (prev_amt + actual_amount) > 1e-9:
                new_avg = (prev_amt * prev_rate + actual_amount * rate_final) / (prev_amt + actual_amount)
                modelo.matriz_tasas_interbancaria[d, lender] = new_avg
            
            # Update Liquidity / Equity
            modelo.estado_bancos[d, Parametros.IDX_BANK_LIQUIDITY] += actual_amount
            modelo.estado_bancos[lender, Parametros.IDX_BANK_LIQUIDITY] -= actual_amount
            
            if actual_tax > 0:
                modelo.estado_bancos[d, Parametros.IDX_BANK_EQUITY] -= actual_tax
                modelo.estado_bancos[d, Parametros.IDX_BANK_LIQUIDITY] -= actual_tax
                total_taxes += actual_tax
                
            cost_transaction = (actual_amount * rate_final) + actual_tax
            bank_actual_refinancing_costs[d] += cost_transaction
            
            current_gaps[d] -= actual_amount
            current_surpluses[lender] -= actual_amount
            
            if current_gaps[d] <= 1e-9:
                break # Done with this borrower

    # --- 4. DISBURSEMENT TO FIRMS ---
    
    # Check if banks have enough liquidity now
    final_liquidity = modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY]
    payout_ratios = np.ones(Parametros.B)
    mask_demand = bank_credit_demand > 1e-9
    np.divide(
        final_liquidity, bank_credit_demand, out=payout_ratios, where=mask_demand
    )
    payout_ratios = np.minimum(1.0, payout_ratios)

    approved_amounts = (
        modelo.current_firm_credit_demand * payout_ratios[chosen_bank_ids]
    )

    f_indices = np.arange(Parametros.F)
    
    old_debt = modelo.matriz_credito_firmas[f_indices, chosen_bank_ids]
    old_rates = modelo.matriz_tasas_firmas[f_indices, chosen_bank_ids]
    
    # Final Rates: The rate the firm LOCKED IN (from step 2)
    # The bank assumes the risk of the estimate being wrong.
    # The paper says: "Banks add refinancing costs...".
    # Since we added 'bank_est_costs' in Step 2, 'rates' already contains the markup.
    # We do NOT add 'bank_actual_refinancing_costs' again, as that would be double counting 
    # or changing the deal after signature. 
    # The bank pays the actual tax, the firm pays the quoted rate.
    
    final_rates_chosen = rates[np.arange(Parametros.F), best_choice_local_idx]
    
    total_new_debt = old_debt + approved_amounts
    mask_pos = total_new_debt > 1e-9
    
    avg_rates = np.zeros_like(old_debt)
    if np.any(mask_pos):
        numerator = (old_debt[mask_pos] * old_rates[mask_pos]) + (approved_amounts[mask_pos] * final_rates_chosen[mask_pos])
        avg_rates[mask_pos] = numerator / total_new_debt[mask_pos]
        
    modelo.matriz_credito_firmas[f_indices, chosen_bank_ids] += approved_amounts
    modelo.matriz_tasas_firmas[f_indices, chosen_bank_ids] = avg_rates

    modelo.estado_firmas[:, Parametros.IDX_FIRM_LIQUIDITY] += approved_amounts
    np.add.at(
        modelo.estado_bancos[:, Parametros.IDX_BANK_LIQUIDITY],
        chosen_bank_ids,
        -approved_amounts,
    )