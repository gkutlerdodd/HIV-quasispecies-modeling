#!/usr/bin/env python3

"""
Compute the maximum observed frequency of each tracked virus variant across multiple simulations.
Usage:
    python3 -m src.preprocessing.max_frequencies
    --topk-files file1.csv file2.csv ...
    --totals-files file1.csv file2.csv ...
    --output max_frequencies.csv

Note that the orders of the topk files and totals files should match.
"""

import argparse
import pandas as pd

def compute_max_frequencies(topk_files, totals_files, output_path):
    if len(topk_files) != len(totals_files):
        raise ValueError(f"Number of topk files ({len(topk_files)}) must match number of totals files ({len(totals_files)}).")

    max_freqs = {}

    for topk, totals in zip(topk_files, totals_files):
        topk_df = pd.read_csv(topk)
        totals_df = pd.read_csv(totals)

        viral_loads = dict(zip(totals_df["time"], totals_df["sum_V"]))

        for _, row in topk_df.iterrows():
            time = row["time"]
            seq = row["sequence"]
            v = float(row["V"])
            freq = v / viral_loads[time]

            if seq not in max_freqs:
                max_freqs[seq] = freq
            else:
                max_freqs[seq] = max(max_freqs[seq], freq)

    result_df = pd.DataFrame([{"sequence": seq, "max_frequency": freq} for seq, freq in max_freqs.items()])
    result_df = result_df.sort_values("max_frequency", ascending=False).reset_index(drop=True)
    result_df.to_csv(output_path, index=False)
    print(f"Saved maximum frequencies to {output_path}.")

def parse_args():
    p = argparse.ArgumentParser(description=("Compute maximum frequency of each variant across multiple simulation runs."))
    p.add_argument("--topk-files", nargs="+", required=True, metavar="PATH", help="One or more topk CSV files.")
    p.add_argument("--totals-files", nargs="+", required=True, metavar="PATH", help="One or more totals CSV files.")
    p.add_argument("--output", default="max_frequencies.csv", metavar="PATH", help="Output CSV path.")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Computing max frequencies from {len(args.topk_files)} seed(s)...")
    compute_max_frequencies(args.topk_files, args.totals_files, args.output)

if __name__ == "__main__":
    main()