
import numpy as np
import pandas as pd
import time
from state import EconomyState
from logic import (
    step1_firms_planning, step2_banks_lending, step3_production, 
    step4_consumption, step5_firm_repayment, step6_interbank_market, 
    step7_evolution
)
from parameters import *

class SimulationRunner:
    def __init__(self, steps=SIMULATION_STEPS):
        self.steps = steps
        self.results = {}

    def run_scenario(self, scenario_name, tax_mode):
        print(f"\n>>> Iniciando Escenario: {scenario_name} (Tax Mode: {tax_mode})")
        start_time = time.time()
        
        # Inicializar Estado
        state = EconomyState()
        
        # Historial de métricas
        history = {
            'step': [],
            'total_production': [],
            'total_consumption': [],
            'firm_defaults': [],
            # 'bank_failures':Removed unused
            'total_debt': [],
            'systemic_risk': [],
            'collected_tax': [],
            'total_bank_equity': []
        }
        
        for t in range(self.steps):
            # --- Paso 1: Planificación ---
            state = step1_firms_planning(state)
            
            # --- Paso 2: Crédito ---
            state = step2_banks_lending(state)
            
            # --- Paso 3: Producción ---
            state = step3_production(state)
            
            # --- Paso 4: Consumo ---
            state = step4_consumption(state)
            
            # --- Paso 5: Repago y Quiebras Firmas ---
            state = step5_firm_repayment(state)
            
            # --- Paso 6: Interbancario y Tax ---
            state = step6_interbank_market(state, tax_mode=tax_mode)
            
            # --- Paso 7: Evolución ---
            state = step7_evolution(state)
            
            # --- Recolección de Datos ---
            history['step'].append(t)
            history['total_production'].append(np.sum(state.firm_stock)) # Stock como proxy de output neto
            history['total_consumption'].append(np.sum(state.firm_daily_sales) if hasattr(state, 'firm_daily_sales') else 0.0) # Placeholder
            
            # Contar defaults recientes (reseteado en step 5?) # No, cumulative aumenta. Deberíamos trackear marginal.
            # Usar firm_cumulative_default diff? 
            # Aproximación: Total Defaults acumulados
            history['firm_defaults'].append(np.sum(state.firm_cumulative_default))
            
            history['total_debt'].append(np.sum(state.bank_firm_loans))
            history['systemic_risk'].append(state.total_systemic_risk)
            history['collected_tax'].append(state.collected_tax)
            history['total_bank_equity'].append(np.sum(state.bank_equity))
            
            if t % 10 == 0:
                print(f"   Step {t}/{self.steps} - SR: {state.total_systemic_risk:.2f} - Tax: {state.collected_tax:.2f}")

        elapsed = time.time() - start_time
        print(f"<<< Escenario {scenario_name} completado en {elapsed:.2f}s")
        
        df = pd.DataFrame(history)
        self.results[scenario_name] = df
        return df

    def save_results(self, filename="simulation_results.csv"):
        # Concatenar todos los escenarios con columna Scenario
        all_dfs = []
        for name, df in self.results.items():
            df['Scenario'] = name
            all_dfs.append(df)
        
        if all_dfs:
            final_df = pd.concat(all_dfs)
            final_df.to_csv(filename, index=False)
            print(f"Resultados guardados en {filename}")
        else:
            print("No hay resultados para guardar.")

if __name__ == "__main__":
    # Configuración de simulación rápida
    # Ejecutamos los 3 escenarios clave del paper
    
    sim = SimulationRunner(steps=SIMULATION_STEPS)
    
    # 1. Benchmark (Sin Tax)
    sim.run_scenario("Benchmark (No Tax)", tax_mode='NONE')
    
    # 2. FTT (Financial Transaction Tax)
    sim.run_scenario("FTT (Tobin Tax)", tax_mode='FTT')
    
    # 3. SRT (Systemic Risk Tax - Paper Proposal)
    sim.run_scenario("SRT (Systemic Risk Tax)", tax_mode='SRT')
    
    sim.save_results()
