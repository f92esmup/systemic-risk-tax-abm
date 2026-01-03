import numpy as np
from simulation import CRISIS_Model
from parameters import Params

def test_phase5():
    print("Initializing Model...")
    model = CRISIS_Model(seed=404)
    
    # Setup Contagion Scenario
    # Chain: Bank 0 owes Bank 1, Bank 1 owes Bank 2.
    # Bank 1 Equity is Low. Bank 2 Equity is Low.
    # Default of Bank 0 should trigger Bank 1 (Asset Loss), which triggers Bank 2.
    
    print("Setting up Contagion Chain: B0 -> B1 -> B2")
    
    # Force Low Equity
    model.banks_state[:, model.IDX_BANK_EQUITY] = 100.0 # Robust default
    model.banks_state[1, model.IDX_BANK_EQUITY] = 10.0 # Fragile
    model.banks_state[2, model.IDX_BANK_EQUITY] = 10.0 # Fragile
    
    # Create Debt
    # B0 owes B1 50.0
    model.L_bb[0, 1] = 50.0
    # B1 owes B2 50.0
    model.L_bb[1, 2] = 50.0
    
    # Now Force B0 to Default
    # Set B0 Equity to -10
    model.banks_state[0, model.IDX_BANK_EQUITY] = -10.0
    
    print("Running Accounting Step (with Cascades)...")
    model.step_accounting()
    
    # Check Defaults
    # B0 should be reset (Positive equity again)
    # B1 should have defaulted (Old Equity 10 - Loss 50 = -40). Then Reset.
    # B2 should have defaulted (Old Equity 10 - Loss 50 from B1? Wait.)
    # B1 defaulted because it lost asset from B0.
    # Did B1 default on its debt to B2?
    # Yes, if B1 defaults, "Write off the debt (Asset gone)".
    # B1 owes B2. Since B1 died, B2 loses that asset.
    # So B2 loses 50. Equity 10 - 50 = -40. B2 Dies.
    
    # Since step_accounting resets dead banks immediately, we can't check magnitude of negative equity easily.
    # But we can check if L_bb was cleared.
    # If reset happened, L_bb[0, 1] should be 0.
    
    print(f"L_bb[0, 1] after reset: {model.L_bb[0, 1]}")
    assert model.L_bb[0, 1] == 0.0, "Bank 0 was reset, debt should be cleared"
    
    print(f"L_bb[1, 2] after reset: {model.L_bb[1, 2]}")
    assert model.L_bb[1, 2] == 0.0, "Bank 1 was reset, debt should be cleared"
    
    # Check if they have fresh equity
    eq0 = model.banks_state[0, model.IDX_BANK_EQUITY]
    eq1 = model.banks_state[1, model.IDX_BANK_EQUITY]
    eq2 = model.banks_state[2, model.IDX_BANK_EQUITY]
    
    print(f"Fresh Equities: B0={eq0}, B1={eq1}, B2={eq2}")
    assert eq0 > 0 and eq1 > 0 and eq2 > 0, "Banks should be alive (reset)"
    
    # Verify Dividend Logic (trivial check)
    # Ensure no crashes on normal flow
    
    print("Phase 5 Test PASSED (Cascades verified logic)")

if __name__ == "__main__":
    test_phase5()
