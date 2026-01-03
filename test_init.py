from simulation import CRISIS_Model
from parameters import Params

def test_init():
    print("Initializing CRISIS_Model...")
    model = CRISIS_Model(seed=42)
    
    print("Checking shapes...")
    print(f"Banks State: {model.banks_state.shape}, Expected: ({Params.B}, {model.N_BANK_FEATURES})")
    assert model.banks_state.shape == (Params.B, model.N_BANK_FEATURES)
    
    print(f"Firms State: {model.firms_state.shape}, Expected: ({Params.F}, {model.N_FIRM_FEATURES})")
    assert model.firms_state.shape == (Params.F, model.N_FIRM_FEATURES)
    
    print(f"Households State: {model.households_state.shape}, Expected: ({Params.H}, {model.N_HH_FEATURES})")
    assert model.households_state.shape == (Params.H, model.N_HH_FEATURES)
    
    print(f"L_bb: {model.L_bb.shape}, Expected: ({Params.B}, {Params.B})")
    assert model.L_bb.shape == (Params.B, Params.B)
    
    print("Checking Indexes...")
    print(f"HH Employer Idx range: [{model.hh_employer_idx.min()}, {model.hh_employer_idx.max()}]")
    assert model.hh_employer_idx.min() >= 0 and model.hh_employer_idx.max() < Params.F
    
    print("Initialization Test PASSED")

if __name__ == "__main__":
    test_init()
