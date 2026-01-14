import pandas as pd
import glob
import os

def summarize():
    print("\n--- RESUMEN DE RESULTADOS DE SIMULACIÓN ---")
    modes = ["NINGUNO", "SRT", "TOBIN"]
    
    for mode in modes:
        pattern = f"outputdata/run_{mode}_sim_*/globals.parquet"
        files = glob.glob(pattern)
        
        total_cascades = 0
        total_losses = 0.0
        n_runs = len(files)
        
        if n_runs == 0:
            print(f"{mode}: No data found.")
            continue
            
        for f in files:
            df = pd.read_parquet(f)
            # Sumar eventos de quiebra en toda la corrida
            # 'cascade_size' es quiebras por step. Sumamos todo.
            total_cascades += df['cascade_size'].sum()
            total_losses += df['contagion_loss'].sum()
            
        avg_cascades = total_cascades / n_runs
        avg_losses = total_losses / n_runs
        
        print(f"MODO: {mode:<10} | Runs: {n_runs} | Avg Bankruptcies: {avg_cascades:.2f} | Avg Losses: {avg_losses:.2f}")

if __name__ == "__main__":
    summarize()

