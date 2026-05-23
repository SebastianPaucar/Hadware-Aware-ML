import argparse
import h5py

def explore_h5(obj, path="/"):
    """Recursively explore the structure of the HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        print(f"{path} [Dataset] shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{path} [Group]")
        for key, val in obj.attrs.items():
            print(f"  └─ Attribute: {key} = {val}")
        for key in obj:
            explore_h5(obj[key], path + key + "/")
    else:
        print(f"{path} [Unknown Type]")

def main():
    parser = argparse.ArgumentParser(description="Display full structure of an HDF5 (.h5) file")
    parser.add_argument("--dataset", required=True, help="Path to the .h5 file")
    args = parser.parse_args()

    try:
        with h5py.File(args.dataset, "r") as f:
            print(f"Full structure of {args.dataset}:")
            explore_h5(f)
    except Exception as e:
        print(f"Error opening file: {e}")

if __name__ == "__main__":
    main()

