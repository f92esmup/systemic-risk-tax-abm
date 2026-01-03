class Params:
    """
    Parameters from Table I of Poledna & Thurner (2016)
    'Elimination of systemic risk in financial networks'
    """
    
    # --- System Dimensions ---
    T = 500             # Time steps
    B = 20              # Number of Banks
    F = 100             # Number of Firms
    H = 1300            # Number of Households
    
    # --- Economic Parameters (Table I) ---
    
    # Labor & Production
    WAGE = 1.0          # Wage rate (wb)
    alpha = 0.1         # Labor productivity
    
    # Consumption
    c = 0.8             # Marginal propensity to consume (High velocity!)
    Z_CONSUMPTION = 2   # Number of firms sampled by households (z)

    # Banking & Credit
    r_bar = 0.02        # Central bank baseline interest rate
    N_SEARCH = 5        # Number of banks a firm searches for credit (n)
    
    # Debt & Regulation
    DEBT_REPAYMENT_RATE = 0.05 # Rate of debt reimbursement (tau)
    
    # Note: 'phi' in paper is "Credit demand contraction" = 0.8.
    # It is NOT capital adequacy. We use a separate constant for init if needed.
    PHI_DEMAND_CONTRACTION = 0.8 
    
    # Dividends
    DIVIDEND_RATIO = 0.2 # Share of dividends (div)
    
    # Taxes
    TAX_TOBIN_RATE = 0.002 # 0.2%
    TAX_SRT_ZETA = 0.02    # Sensitivity for SRT (Main text uses 0.02)
    # TAX_SRT_ZETA = 1.0   # Use this for "Appendix B" strong mode
    
    # --- Algorithm / Simulation parameters (Not in Table I but required) ---
    
    # Price adjustment (Equation logic)
    PRICE_ADJUSTMENT_SPEED = 0.05   
    PRICE_DRIFT_STD = 0.01          

    # Initialization Distributions
    # To fix "Empty Volume", agents must start with LESS cash relative to needs.
    # If Liquidity is high, they don't borrow.
    # We set initial assets to be moderate.
    INIT_BANK_ASSETS = (100, 300) 
    INIT_FIRM_ASSETS = (5, 15)   # Firms start small
    
    # Capital Adequacy for Initialization (Not phi from Table I, but standard Basel)
    INIT_CAPITAL_RATIO = 0.10    # 10% Equity initially
    
    @classmethod
    def get_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}
