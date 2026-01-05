import numpy as np
from parameters import Params
import functions as fn

class CRISIS_Model:
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
    IDX_FIRM_PROD = 4       # Production/Inventory
    IDX_FIRM_WORKERS = 5
    IDX_FIRM_WAGES = 6      # Wages Bill
    IDX_FIRM_PRICE_PREV = 7
    IDX_FIRM_DEMAND_PREV = 8
    IDX_FIRM_LEVERAGE = 9
    IDX_FIRM_DEFAULT_FLAG = 10
    N_FIRM_FEATURES = 11

    # Households State Indices (N=3)
    IDX_HH_DEPOSITS = 0
    IDX_HH_IS_OWNER = 1      # 0=Worker, 1=Owner
    IDX_HH_OWNED_ENTITY_IDX = 2 # Index of Firm or Bank owned
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
        Reset or initialize all state tensors and network topologies for a new simulation run.
        """
        # --- Dimensions ---
        B = Params.B
        F = Params.F
        H = Params.H

        # --- 1. Agents State (Tensors) ---
        self.banks_state = np.zeros((B, self.N_BANK_FEATURES), dtype=np.float64)
        self.firms_state = np.zeros((F, self.N_FIRM_FEATURES), dtype=np.float64)
        self.households_state = np.zeros((H, self.N_HH_FEATURES), dtype=np.float64)

        # --- 2. Vectorized Initialization ---
        
        # Banks: Specificity Parameters (Constant per run)
        # CHI ~ U(0, 1)
        self.banks_state[:, self.IDX_BANK_OPERATING_COST_CHI] = self.rng.uniform(
            Params.CHI_RANGE[0], Params.CHI_RANGE[1], size=B
        )
        # PSI ~ U(0, 0.1)
        self.banks_state[:, self.IDX_BANK_INTERBANK_COST_PSI] = self.rng.uniform(
            Params.PSI_RANGE[0], Params.PSI_RANGE[1], size=B
        )

        # Bank Financials
        init_bank_assets = self.rng.uniform(
            Params.INIT_BANK_ASSETS[0], Params.INIT_BANK_ASSETS[1], size=B
        )
        self.banks_state[:, self.IDX_BANK_TOTAL_ASSETS] = init_bank_assets
        # Equity = Assets * Capital Ratio
        self.banks_state[:, self.IDX_BANK_EQUITY] = (
            init_bank_assets * Params.INIT_CAPITAL_RATIO
        )
        # Liquidity = Assets (Assuming start with all liquid)
        self.banks_state[:, self.IDX_BANK_LIQUIDITY] = init_bank_assets
        # Deposits = Assets - Equity
        self.banks_state[:, self.IDX_BANK_DEPOSITS] = (
            init_bank_assets - self.banks_state[:, self.IDX_BANK_EQUITY]
        )

        # Firms
        init_firm_assets = self.rng.uniform(
            Params.INIT_FIRM_ASSETS[0], Params.INIT_FIRM_ASSETS[1], size=F
        )
        self.firms_state[:, self.IDX_FIRM_EQUITY] = init_firm_assets
        self.firms_state[:, self.IDX_FIRM_LIQUIDITY] = init_firm_assets

        # Initialize Price to Marginal Cost
        init_price = Params.WAGE / Params.alpha
        self.firms_state[:, self.IDX_FIRM_PRICE] = init_price
        self.firms_state[:, self.IDX_FIRM_PRICE_PREV] = init_price

        # Households: Owners vs Workers
        # Random assignment
        hh_indices = np.arange(H)
        self.rng.shuffle(hh_indices)

        # First F households own Firms
        firm_owners = hh_indices[:F]
        self.households_state[firm_owners, self.IDX_HH_IS_OWNER] = 1.0
        self.households_state[firm_owners, self.IDX_HH_OWNED_ENTITY_IDX] = np.arange(F)

        # Next B households own Banks
        bank_owners = hh_indices[F : F + B]
        self.households_state[bank_owners, self.IDX_HH_IS_OWNER] = 1.0
        self.households_state[bank_owners, self.IDX_HH_OWNED_ENTITY_IDX] = np.arange(B)

        # Remaining are Workers (Default 0, 0)
        
        # --- 3. Topology & Relationships ---
        self.L_bb = np.zeros((B, B), dtype=np.float64)
        self.L_fb = np.zeros((F, B), dtype=np.float64)

        # Bank/Employer Relationships for Workers
        self.hh_employer_idx = self.rng.integers(0, F, size=H)
        self.hh_bank_idx = self.rng.integers(0, B, size=H)

        # --- 4. History / Traceability ---
        self.step_buffer = {
            "L_bb": [],
            "L_fb": [],
            "banks_state": [],
            "firms_state": [],
            "hh_bank_idx": [],
            # "hh_employer_idx": [] # Not strictly requested in prompt list but useful. 
            # Prompt asked for: L_bb, L_fb, banks_state, firms_state, hh_bank_idx.
        }

        self.record_history()

    def record_history(self):
        """Append current state snapshots to step_buffer."""
        # Topologies
        self.step_buffer["L_bb"].append(self.L_bb.astype(np.float32).copy())
        self.step_buffer["L_fb"].append(self.L_fb.astype(np.float32).copy())
        
        # States (Full capture)
        self.step_buffer["banks_state"].append(self.banks_state.astype(np.float32).copy())
        self.step_buffer["firms_state"].append(self.firms_state.astype(np.float32).copy())
        
        # Relations
        self.step_buffer["hh_bank_idx"].append(self.hh_bank_idx.astype(np.int16).copy())

    def reset_history(self):
        """Clear the step buffer to free RAM after flushing."""
        for key in self.step_buffer:
            self.step_buffer[key] = []

    def save_run_to_disk(self, run_id, folder="output_data"):
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

    # --- Placeholder Step Logic (To be implemented in Phase 2) ---
    def step_firms_planning(self):
        pass

    def step_banking_market(self):
        pass

    def step_real_economy(self):
        pass

    def step_accounting(self):
        pass

    def run_step(self):
        """Execute one simulation step."""
        # Reset Per-Step Accumulators
        self.current_step_volume = 0.0
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        # Run Phases (Currently No-Op)
        self.step_firms_planning()
        self.step_banking_market()
        self.step_real_economy()
        self.step_accounting()

        # Traceability
        self.record_history()