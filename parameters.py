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
    # Firms
    gamma = 0.5         # Input substitutability parameter
    alpha = 0.1         # Labor productivity (updated from 0.2 to match user prompt)
    
    WAGE = 1.0          # Fixed wage
    PRICE_ADJUSTMENT_SPEED = 0.05   # Price adjustment speed parameter (beta/lambda)
    PRICE_DRIFT_STD = 0.01          # Standard deviation for random price fluctuation

    # Banks
    r_bar = 0.02        # Central bank baseline interest rate (2%)
    phi = 0.2           # Bank capital adequacy target
    
    tau = 0.1           # Tax rate or similar
    
    # Households
    c = 0.2             # Marginal propensity to consume

    DIV_SHARE = 0.2     # Dividends share
    DIVIDEND_RATIO = 0.2 # Alias
    DEBT_REPAYMENT_RATE = 0.05 # Fraction of debt repaid per step
    
    # Paper specific tax rates
    TAX_TOBIN_RATE = 0.002 # 0.2%
    TAX_SRT_ZETA = 1.0     # Sensitivity for SRT
    
    # Initialization Distributions (ranges)
    INIT_BANK_ASSETS = (200, 800) 
    INIT_FIRM_ASSETS = (10, 50)
    
    # Graph Topology
    CONNECTION_PROB_BB = 0.2 
    N_SEARCH = 5        # Number of banks a firm searches for credit
    Z_CONSUMPTION = 2   # Number of firms a household samples for consumption
    
    # Households
    c = 0.8             # Marginal propensity to consume (Updated)
    
    @classmethod
    def get_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}
