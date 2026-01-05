import numpy as np
import glob

print("Checking 'none' mode data...")
files = sorted(glob.glob("output_data/none/*.npz"))
if not files:
    print("No files found in output_data/none/")
else:
    f = files[-1]
    print(f"Inspecting {f}")
    try:
        d = np.load(f)
        if "L_bb" not in d:
            print("Key 'L_bb' NOT in data!")
        else:
            L_bb = d["L_bb"]
            print(f"L_bb shape: {L_bb.shape}")
            print(f"L_bb (last step) max: {L_bb[-1].max()}")
            print(f"L_bb (last step) sum: {L_bb[-1].sum()}")

            # Check non-zeros
            nz = np.count_nonzero(L_bb[-1])
            print(f"Non-zero elements in last step: {nz}")
    except Exception as e:
        print(f"Error: {e}")
