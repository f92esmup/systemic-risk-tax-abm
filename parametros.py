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

    @classmethod
    def get_dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith("__")}
