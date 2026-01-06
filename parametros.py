class Parametros:
    """
    Parameters from Table I of Poledna & Thurner (2016)
    'Elimination of systemic risk in financial networks'
    """

    # --- System Dimensions ---
    T = 500  # Time steps
    B = 20  # Number of Banks (N^b)
    F = 100  # Number of Firms (N^f)
    H = 1300  # Number of Households (N^h)

    # --- Economic Parameters (Table I) ---

    # Labor & Production
    # CALIBRATION: Matches Table I in Poledna & Thurner (2016)
    WAGE = 1.0  # Wage rate (w_b)
    alpha = 0.1  # Labor productivity (alpha)

    # Consumption
    c = 0.8  # Marginal propensity to consume (c)
    Z_CONSUMPTION = 2  # Number of firms sampled by households (z)

    # Banking & Credit
    R_BAR = 0.02  # Central bank baseline interest rate (r_bar)
    N_SEARCH = 5  # Number of banks a firm searches for credit (n)

    # Interest Rate Mechanism (Appendix A)
    CHI_RANGE = (0.0, 1.0)  # Bank specificity cost (chi)
    PSI_RANGE = (0.0, 0.1)  # Interbank specificity cost (psi)
    K_mu = 10.0  # Slope for hyperbolic tangent function (K)

    # Debt & Regulation
    DEBT_REPAYMENT_RATE = 0.05  # Rate of debt reimbursement (tau)

    # Credit demand contraction (phi) - Not in Table I but standard in model logic
    PHI_DEMAND_CONTRACTION = 0.8

    # Dividends
    DIVIDEND_RATIO = 0.2  # Share of dividends (div)

    # Taxes
    TAX_TOBIN_RATE = 0.002  # 0.2%
    TAX_SRT_ZETA = 0.02  # Sensitivity for SRT (Main text uses 0.02)

    # --- Algorithm / Simulation Parameters ---

    # Price adjustment
    PRICE_ADJUSTMENT_SPEED = 0.05
    PRICE_DRIFT_STD = 0.01

    # Initialization Distributions
    # CALIBRATION: Supply ~30,000 to match calibrated Demand
    INIT_BANK_ASSETS = (1000, 2000)
    INIT_FIRM_ASSETS = (5, 15)

    # Capital Adequacy for Initialization
    INIT_CAPITAL_RATIO = 0.10  # 10% Equity initially

    # --- COLUMN INDICES (SCHEMA) ---

    # Households (N=3)
    IDX_HH_DEPOSITS = 0
    N_HH_FEATURES = 1

    # Firms (N=11)
    IDX_FIRM_LIQUIDITY = 0
    IDX_FIRM_EQUITY = 1
    IDX_FIRM_PRICE = 2
    IDX_FIRM_DEMAND = 3
    IDX_FIRM_PROD = 4
    IDX_FIRM_WORKERS = 5
    IDX_FIRM_WAGES = 6
    IDX_FIRM_PRICE_PREV = 7
    IDX_FIRM_DEMAND_PREV = 8
    IDX_FIRM_LEVERAGE = 9
    IDX_FIRM_DEFAULT_FLAG = 10
    N_FIRM_FEATURES = 11

    # Banks (N=8)
    IDX_BANK_LIQUIDITY = 0
    IDX_BANK_EQUITY = 1
    IDX_BANK_DEPOSITS = 2
    IDX_BANK_BAD_DEBT = 3
    IDX_BANK_OPERATING_COST_CHI = 4
    IDX_BANK_INTERBANK_COST_PSI = 5
    IDX_BANK_DEFAULT_PROB = 6
    IDX_BANK_TOTAL_ASSETS = 7
    N_BANK_FEATURES = 8

    @classmethod
    def get_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
