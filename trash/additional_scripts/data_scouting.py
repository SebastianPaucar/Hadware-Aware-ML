import argparse
import h5py
import numpy as np

def parse_range(s):
    """Parses a string like '10:20' into a slice object."""
    try:
        start, end = map(int, s.split(":"))
        return slice(start, end)
    except:
        raise argparse.ArgumentTypeError("Range must be in the format start:end")

def main():
    parser = argparse.ArgumentParser(description="Extract and inspect part of an HDF5 dataset.")
    parser.add_argument("--file", required=True, help="Path to the HDF5 file")
    parser.add_argument("--key", required=True, help="Dataset key (e.g. full_data_qual)")
    parser.add_argument("--row-range", type=parse_range, default=None, help="Row range (start:end)")
    parser.add_argument("--col-range", type=parse_range, default=None, help="Column range (start:end)")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        if args.key not in f:
            print(f"Key '{args.key}' not found in file.")
            return

        dataset = f[args.key]
        shape = dataset.shape
        dtype = dataset.dtype
        print(f"Dataset shape: {shape}, dtype: {dtype}")

        row_slice = args.row_range if args.row_range else slice(0, shape[0])

        # Handle 1D, 2D, or 3D data
        if len(shape) == 1:
            sliced_data = dataset[row_slice]
        elif len(shape) == 2:
            col_slice = args.col_range if args.col_range else slice(0, shape[1])
            sliced_data = dataset[row_slice, col_slice]
        elif len(shape) == 3:
            col_slice = args.col_range if args.col_range else slice(0, shape[1])
            sliced_data = dataset[row_slice, col_slice, :]
        else:
            print("Only supports datasets with 1, 2, or 3 dimensions.")
            return

        print("Extracted data:")
        print(sliced_data)

        # Count metrics
        if dtype == np.bool_:
            true_count = np.count_nonzero(sliced_data)
            print(f"Total True values: {true_count}")
        else:
            if len(sliced_data.shape) >= 2:
                nonzero_rows = np.count_nonzero(np.any(sliced_data != 0, axis=1))
            else:
                nonzero_rows = np.count_nonzero(sliced_data != 0)
            print(f"Rows with non-zero values: {nonzero_rows} out of {sliced_data.shape[0]}")

if __name__ == "__main__":
    main()

