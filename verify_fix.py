import pandas as pd
import numpy as np
import os
import shutil
from parametros import Param
from main import ejecutar_simulacion

def verify_fix():
    print("--- Verifying Fixes ---")
    
    # 1. Configure Short Run
    Param.T = 200
    run_id = "test_verification"
    output_dir = f"outputdata/run_{run_id}"
    
    # Clean previous test
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    print(f"Running simulation 'NINGUNO' for {Param.T} steps...")
    
    # 2. Run Simulation
    try:
        ejecutar_simulacion(modo_impuesto="NINGUNO", semilla=42, run_id=run_id)
    except Exception as e:
        print(f"CRITICAL ERROR: Simulation failed with exception: {e}")
        return

    # 3. Analyze Results
    print("\n--- Analysis ---")
    
    # Check Globals (DebtRank > 0?)
    globals_path = os.path.join(output_dir, "globals.parquet")
    if not os.path.exists(globals_path):
        print("ERROR: globals.parquet not generated.")
        return
        
    df_g = pd.read_parquet(globals_path)
    avg_dr_mean = df_g['avg_dr'].mean()
    max_dr = df_g['avg_dr'].max()
    final_dr = df_g['avg_dr'].iloc[-1]
    
    print(f"Average System DebtRank (Mean over time): {avg_dr_mean:.6f}")
    print(f"Max System DebtRank: {max_dr:.6f}")
    print(f"Final System DebtRank: {final_dr:.6f}")
    
    if max_dr > 0:
        print("SUCCESS: DebtRank is non-zero. Interbank network is forming.")
    else:
        print("FAILURE: DebtRank is consistently zero. Network not forming.")

    # Check Banks Survival (Equity > 0?)
    banks_path = os.path.join(output_dir, "banks.parquet")
    if os.path.exists(banks_path):
        df_b = pd.read_parquet(banks_path)
        # Check last step
        last_t = df_b['t'].max()
        final_banks = df_b[df_b['t'] == last_t]
        alive_count = (final_banks['eq'] > 0).sum()
        total_banks = len(final_banks)
        
        print(f"Banks Alive at t={last_t}: {alive_count}/{total_banks}")
        
        if alive_count > 0:
            print("SUCCESS: System did not totally collapse.")
        else:
            print("WARNING: Total system collapse (might be expected behavior for some parameters, but check logic).")
    
    # Check Transactions (Did we capture anything?)
    trans_path = os.path.join(output_dir, "transactions.parquet")
    if os.path.exists(trans_path):
        df_t = pd.read_parquet(trans_path)
        print(f"Total Interbank Transactions Recorded: {len(df_t)}")
        if len(df_t) > 0:
            print("SUCCESS: Transactions are occurring and being logged.")
            if 'marginal_sr' in df_t.columns:
                 print(f"Marginal SR (Delta) range: [{df_t['marginal_sr'].min():.6f}, {df_t['marginal_sr'].max():.6f}]")
        else:
            print("WARNING: No transactions recorded.")
    else:
        print("WARNING: transactions.parquet not found (maybe no transactions happened).")

if __name__ == "__main__":
    verify_fix()
