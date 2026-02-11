import h5py

def print_structure(name, obj):
    """Callback function to print the name of every object in the HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name:40} | SHAPE: {obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"GROUP:   {name}")

file_path = "./saywer_demos/demo.hdf5"
with h5py.File(file_path, "r") as f:
    print(f"\n--- FULL STRUCTURE OF {file_path} ---")
    f.visititems(print_structure)
    print("-" * 50)
