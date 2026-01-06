import numpy as np

file_path = "output_data/none/run_00000.npz"
try:
    data = np.load(file_path)
    print(f"Loaded {file_path}")
    print("Keys:", data.files)
    
    for key in data.files:
        arr = data[key]
        print(f"\nMatrix: {key}")
        print(f"Shape: {arr.shape}")
        print(f"Max Value: {arr.max()}")
        print(f"Min Value: {arr.min()}")
        print(f"Mean Value: {arr.mean()}")
        if "interbancaria" in key or "credito" in key:
            print(f"Non-zero elements: {np.count_nonzero(arr)}")

except Exception as e:
    print(f"Error loading file: {e}")

