import sys
import pandas as pd
from spinplots.utils import nmr_df

def main():
    """
    Convert Bruker's NMR data to csv files on the terminal
    """
    if len(sys.argv) != 3:
        print("Usage: bruker2csv <path_to_data> <path_to_output>")
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        df = nmr_df(data_path)
        if df.attrs.get('nmr_dim') == 2:
            df.to_csv(output_path, index=True)
            print(f"Data written to {output_path}")
        elif df.attrs.get('nmr_dim') == 1:
            df.to_csv(output_path, index=False)
            print(f"Data written to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()