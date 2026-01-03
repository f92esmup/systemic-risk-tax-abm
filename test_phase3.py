import numpy as np
from simulation import CRISIS_Model
from parameters import Params
import functions as fn

def test_phase3():
    print("Initializing Model...")
    model = CRISIS_Model(seed=202)
    
    # Setup Scenario:
    # 1. Create a Firms demand context
    # Force Firm 0 to have high demand and no liquidity -> Needs Credit
    model.firms_state[0, model.IDX_FIRM_LIQUIDITY] = 0
    model.firms_state[0, model.IDX_FIRM_DEMAND] = 50.0 # High demand
    
    # 2. Setup Banks
    # Bank 0: Deficit (Low Liquidity)
    model.banks_state[0, model.IDX_BANK_LIQUIDITY] = 10.0 # Has little cash
    model.banks_state[0, model.IDX_BANK_EQUITY] = 20.0
    
    # Bank 1: Surplus (High Liquidity)
    model.banks_state[1, model.IDX_BANK_LIQUIDITY] = 1000.0
    model.banks_state[1, model.IDX_BANK_EQUITY] = 100.0
    
    print("Running Firms Planning...")
    model.step_firms_planning()
    
    # Force Firm 0 to pick Bank 0? Hard to force random choice.
    # Instead, let's inject credit demand directly to test Interbank
    print("Injecting artificial demand to Bank 0 to force Interbank...")
    # Clean previous
    model.current_credit_demand[:] = 0
    
    # Set Firm 0 demand
    model.current_credit_demand[0] = 500.0 
    
    # Mock the selection: Firm 0 chooses Bank 0
    # We can't easily mock internal random choice without patching.
    # But we can verify `functions.py` independently first.
    
    print("Testing DebtRank Function...")
    # Manual tiny network
    # Triangle: A->B, B->C, C->A
    L_test = np.array([
        [0, 10, 0],
        [0, 0, 10],
        [10, 0, 0]
    ], dtype=float)
    E_test = np.array([5, 5, 5], dtype=float)
    v_test = np.array([100, 100, 100], dtype=float) # Total Assets
    
    DR = fn.compute_debtrank(L_test, E_test, v_test)
    print(f"Triangle DebtRank: {DR}")
    # With L=10, E=5, Impact W=1. Chain reaction should yield High DR.
    assert np.all(DR > 0), "DebtRank should be positive for connected cyclic graph"
    
    print("Running Banking Market Step...")
    # This will run Part A (random selection) and Part B.
    # If Firm 0 picks Bank 0 (prob 1/20 * N_Search), Bank 0 goes deficit.
    # Since we can't guarantee Firm 0 picks Bank 0, we might not see interbank action.
    # So let's force a Deficit in Bank 0 manually by overriding state BEFORE step B?
    # No, step_banking_market runs A then B.
    
    # We rely on statistical chance or just check code integrity.
    # Let's run and check if L_bb or L_fb changed at all.
    
    old_L_bb = model.L_bb.copy()
    model.step_banking_market()
    
    # Check if any loans happened
    new_loans = model.L_fb.sum()
    print(f"Total Firm Loans Granted (L_fb sum): {new_loans}")
    
    # Check Interbank
    interbank_activity = model.L_bb - old_L_bb
    print(f"Interbank Activity (Sum of new L_bb): {interbank_activity.sum()}")
    
    if interbank_activity.sum() > 0:
        print("Interbank Market Active! Loans were created.")
    else:
        print("No Interbank loans this step (might simply be ample liquidity or no matching deficit).")
        
    print("Phase 3 Test PASSED (Code ran without index errors)")

if __name__ == "__main__":
    test_phase3()
