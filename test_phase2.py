import numpy as np
from simulation import CRISIS_Model
from parameters import Params

def test_phase2():
    print("Initializing Model...")
    model = CRISIS_Model(seed=101)
    
    # Manually set some initial values to predict outcome
    # Firm 0: High liquidity, should need no credit
    model.firms_state[0, model.IDX_FIRM_LIQUIDITY] = 10000.0
    model.firms_state[0, model.IDX_FIRM_DEMAND] = 10.0 # Small demand
    
    # Firm 1: Low liquidity, should need credit
    model.firms_state[1, model.IDX_FIRM_LIQUIDITY] = 0.0
    model.firms_state[1, model.IDX_FIRM_DEMAND] = 20.0 # Creates wage bill
    
    # Set mixed prices
    model.firms_state[:, model.IDX_FIRM_PRICE] = 1.0
    model.firms_state[0, model.IDX_FIRM_PRICE] = 2.0 # Higher than avg
    
    print("Running step_firms_planning()...")
    credit_demands = model.step_firms_planning()
    
    # Checks
    print("Checking Credit Demand...")
    # Firm 0: Bill = (10 / 0.1) * 1.0 = 100. Liquidity 10000. Gap < 0. Credit 0.
    # Note: Logic initializes demand randomly if 0? No, we set it to 10.0. 
    # Wait, code says: "if np.all(current_demand == 0): init". We set some demands != 0, so it shouldn't re-init all.
    # But wait, we only set firsm 0 and 1. Others are 0. So np.all is False.
    # But others (2..99) operate on 0 demand -> 0 shift -> 0 labor.
    
    # Let's inspect Firm 0
    f0_demand = model.firms_state[0, model.IDX_FIRM_DEMAND]
    f0_workers = model.firms_state[0, model.IDX_FIRM_WORKERS]
    f0_credit = credit_demands[0]
    
    print(f"Firm 0 - Demand (approx 10): {f0_demand:.2f}")
    print(f"Firm 0 - Workers (approx 100): {f0_workers}")
    print(f"Firm 0 - Credit Demand (Expected 0): {f0_credit}")
    
    assert f0_credit == 0, f"Firm 0 shouldn't need credit. Got {f0_credit}"
    
    # Firm 1
    # Demand approx 20. Workers ~200. Bill ~200. Liquidity 0. Credit ~200.
    f1_credit = credit_demands[1]
    print(f"Firm 1 - Credit Demand (Expected ~200+noise): {f1_credit:.2f}")
    assert f1_credit > 100, "Firm 1 should need substantial credit"
    
    # Check Price Convergence (Mean reversion)
    # Firm 0 started at 2.0. Avg was close to 1. Should decrease.
    f0_price = model.firms_state[0, model.IDX_FIRM_PRICE]
    print(f"Firm 0 - Price (Started 2.0, Expected < 2.0): {f0_price:.4f}")
    assert f0_price < 2.0, "Price should adjust towards mean"
    
    # Check Vectorization
    assert credit_demands.shape == (Params.F,), "Output shape mismatch"
    assert np.all(model.firms_state[:, model.IDX_FIRM_WORKERS] >= 0), "Negative workers?"
    
    print("Phase 2 Test PASSED")

if __name__ == "__main__":
    test_phase2()
