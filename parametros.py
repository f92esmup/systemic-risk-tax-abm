class Parametros:
    """
    Parameters from Table I of Poledna & Thurner (2016)
    'Elimination of systemic risk in financial networks'
    """

    # --- System Dimensions ---
    T = 500  # Time steps
    B = 20  # Number of Banks
    F = 100  # Number of Firms
    H = 1300  # Number of Households

    # --- Economic Parameters (Table I) ---

    # Labor & Production
    WAGE = 1.0  # Wage rate (wb)
    alpha = 0.1  # Labor productivity

    # Consumption
    c = 0.8  # Marginal propensity to consume
    Z_CONSUMPTION = 2  # Number of firms sampled by households (z)

    # Banking & Credit
    R_BAR = 0.02  # Central bank baseline interest rate (r_bar)
    N_SEARCH = 5  # Number of banks a firm searches for credit (n)

    # Interest Rate Mechanism (Appendix A)
    CHI_RANGE = (0.0, 1.0)  # Bank specificity for firm loans
    PSI_RANGE = (0.0, 0.1)  # Bank specificity for interbank loans
    K_mu = 10.0  # Slope for hyperbolic tangent function

    # Debt & Regulation
    DEBT_REPAYMENT_RATE = 0.05  # Rate of debt reimbursement (tau)

    # Credit demand contraction (phi)
    PHI_DEMAND_CONTRACTION = 0.8

    # Dividends
    DIVIDEND_RATIO = 0.2  # Share of dividends (div)

    # Taxes
    TAX_TOBIN_RATE = 0.002  # 0.2%
    TAX_SRT_ZETA = 0.02  # Sensitivity for SRT (Main text uses 0.02)

    # --- Algorithm / Simulation parametros ---

    # Price adjustment
    PRICE_ADJUSTMENT_SPEED = 0.05
    PRICE_DRIFT_STD = 0.01

    # Initialization Distributions
    INIT_BANK_ASSETS = (1000, 3000)
    INIT_FIRM_ASSETS = (5, 15)  # Firms start small

    # Capital Adequacy for Initialization
    INIT_CAPITAL_RATIO = 0.10  # 10% Equity initially

    # --- COLUMN INDICES (SCHEMA) ---

    # Households (N=5)
    IDX_HH_DEPOSITS = 0
    IDX_HH_IS_OWNER = 1  # 0=Worker, 1=Owner
    IDX_HH_OWNED_TYPE = 2  # 0=None, 1=Firm, 2=Bank
    IDX_HH_OWNED_ENTITY_IDX = 3  # Index of Firm or Bank
    IDX_HH_EMPLOYER_IDX = 4  # Index of Firm (Employer)
    N_HH_FEATURES = 5

    # Firms (N=11)
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

