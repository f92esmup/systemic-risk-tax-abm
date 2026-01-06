import numpy as np
from parametros import Parametros

# Import steps
from logica.paso1_planificacion import ejecutar_paso1
from logica.paso2_interbancario import ejecutar_paso2
from logica.paso3_produccion import ejecutar_paso3
from logica.paso4_consumo import ejecutar_paso4
from logica.paso5_dividendos_quiebras import ejecutar_paso5
from logica.paso6_repago_deuda import ejecutar_paso6
from logica.paso7_cierre import ejecutar_paso7


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

        # Households start with zero (receive wages in first step)
        self.estado_hogares[:, Parametros.IDX_HH_DEPOSITS] = 0.0

        # Initialize Price to Marginal Cost
        init_price = Parametros.WAGE / Parametros.alpha
        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE] = init_price
        self.estado_firmas[:, Parametros.IDX_FIRM_PRICE_PREV] = init_price

        # --- 3. Topology & Relationships (Matrices) ---

        # A. Interbank Network (B x B) - Dynamic
        self.matriz_interbancaria = np.zeros((B, B), dtype=np.float64)

        # B. Credit Network (Firm -> Bank) (F x B) - Dynamic
        self.matriz_credito_firmas = np.zeros((F, B), dtype=np.float64)

        # Interest Rate Matrices (Store agreed interest rates)
        self.matriz_tasas_interbancaria = np.zeros((B, B), dtype=np.float64)
        self.matriz_tasas_firmas = np.zeros((F, B), dtype=np.float64)

        # Profit Tracking (per step)
        self.current_step_profit_firms = np.zeros(F, dtype=np.float64)
        self.current_step_profit_bancos = np.zeros(B, dtype=np.float64)

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
        self.step_buffer["matriz_interbancaria"].append(
            self.matriz_interbancaria.astype(np.float32).copy()
        )
        self.step_buffer["matriz_credito_firmas"].append(
            self.matriz_credito_firmas.astype(np.float32).copy()
        )
        self.step_buffer["matriz_consumo"].append(
            self.matriz_consumo.astype(np.float32).copy()
        )

        # These are technically static or semi-static, but for full reconstruction we save them.
        self.step_buffer["matriz_laboral"].append(
            self.matriz_laboral.astype(np.int8).copy()
        )
        self.step_buffer["matriz_depositos"].append(
            self.matriz_depositos.astype(np.int8).copy()
        )
        self.step_buffer["matriz_propiedad_firmas"].append(
            self.matriz_propiedad_firmas.astype(np.int8).copy()
        )
        self.step_buffer["matriz_propiedad_bancos"].append(
            self.matriz_propiedad_bancos.astype(np.int8).copy()
        )

        # States
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

    def ejecutar_paso(self):
        """Execute one simulacion step."""
        self.current_step_volume = 0.0
        self.current_step_loss = 0.0
        self.current_step_defaults = 0

        ejecutar_paso1(self)
        ejecutar_paso2(self)
        ejecutar_paso3(self)
        ejecutar_paso4(self)
        ejecutar_paso5(self)  # Dividends & Bankruptcies (Step 5)
        ejecutar_paso6(self)  # Repayment (Step 6)
        ejecutar_paso7(self)

        self.registrar_historia()

