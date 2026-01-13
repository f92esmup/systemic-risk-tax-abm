import pandas as pd
import numpy as np
import os

def check_run(run_path):
    print(f"\n--- Checking stability for: {run_path} ---")
    
    # 1. Check Globals
    globals_path = os.path.join(run_path, "globals.parquet")
    if os.path.exists(globals_path):
        df_g = pd.read_parquet(globals_path)
        print("First 10 steps of globals:")
        print(df_g[['t', 'volume_ib', 'avg_dr', 'total_eq_banks']].head(10))
        
        has_nan = df_g.isnull().values.any()
        print(f"Globals has NaNs: {has_nan}")
        
        print(f"Final average DebtRank: {df_g['avg_dr'].iloc[-1]:.4f}")
    else:
        print("globals.parquet not found")

    # 2. Check Firms
    firms_path = os.path.join(run_path, "firms.parquet")
    if os.path.exists(firms_path):
        df_f = pd.read_parquet(firms_path)
        print("First 10 rows of firms (multiple steps):")
        print(df_f[['t', 'id', 'liq', 'eq', 'prod']].head(10))
        
        has_nan = df_f.isnull().values.any()
        print(f"Firms has NaNs: {has_nan}")
        print(f"Price range: [{df_f['liq'].min():.2f}, {df_f['liq'].max():.2f}] (Liquidity)")
        print(f"Equity range: [{df_f['eq'].min():.2f}, {df_f['eq'].max():.2f}]")
    else:
        print("firms.parquet not found")

if __name__ == "__main__":
    runs = [
        "outputdata/run_NINGUNO_sim_0",
        "outputdata/run_TOBIN_sim_0",
        "outputdata/run_SRT_sim_0"
    ]
    for r in runs:
        if os.path.exists(r):
            check_run(r)
        else:
            print(f"Run {r} does not exist.")

