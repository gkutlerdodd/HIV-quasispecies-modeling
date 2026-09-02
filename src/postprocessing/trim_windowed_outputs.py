#!/usr/bin/env python3
"""
Reduce the size of windowed flux/node CSVs produced by mutational_graph.py by applying any combination of
    1. Time trim: keep only windows where window_start < --t-max
    2. Column pruning: drop columns not used by visualize_flux_graph.py
    3. Sequence filter: keep only rows whose sequences are in a top-node set.

Requirements: requires pyarrow.

Output: trimmed windowed flux/node CSVs ready to be used for visualize_flux_graph.py.

Usage:
    python3 -m src.postprocessing.trim_windowed_outputs --input FILE [FILE ...] [options]

    Examples:
        Compressing two graph outputs using computed top 40 sequences within 0 < t < 500 days:
            python3 trim_windowed_csvs.py \
                --input windowed_flux_time365_seed1.csv windowed_flux_time365_seed2.csv \
                        windowed_nodes_time365_seed1.csv windowed_nodes_time365_seed2.csv \
                --t-max 500 --prune-columns \
                --windowed-node-csv windowed_nodes_time365_seed1.csv \
                                    windowed_nodes_time365_seed2.csv \
                --top-n 40 --save-top-nodes top40.txt
        Compressing two graph outputs using a pre-computed top 40 sequences:
            python3 trim_windowed_csvs.py \
                --input windowed_flux_time365_seed1.csv windowed_flux_time365_seed2.csv \
                        windowed_nodes_time365_seed1.csv windowed_nodes_time365_seed2.csv \
                --t-max 500 --prune-columns \
                --top-nodes-file top40.txt
"""
import os
import sys
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pac
import pyarrow.compute as pc

EDGE_KEEP_COLS = ["window_start", "window_end", "seq_i", "seq_j", "flux"]
NODE_KEEP_COLS = ["window_start", "window_end", "sequence", "total_abundance"]
EDGE_ID_COLS = {"seq_i", "seq_j", "flux"}
NODE_ID_COLS = {"sequence", "total_abundance"}

_BLOCK_BYTES = 64 * 1024 * 1024 #64 MB streaming block size

###################
##Top node helpers
###################
def load_top_nodes_from_file(path):
    with open(path) as f:
        ranked = [line.strip() for line in f if line.strip()]
    return set(ranked), ranked

def compute_top_nodes_from_totals(total_node_csvs, top_n):
    """Select top_n sequences by total_flux summed across non-windowed node CSVs."""
    acc = {}
    for path in total_node_csvs:
        df = pd.read_csv(path, usecols=["sequence", "total_flux"])
        for row in df.itertuples(index=False):
            acc[row.sequence] = acc.get(row.sequence, 0.0) + row.total_flux
        del df
    ranked = sorted(acc, key=acc.__getitem__, reverse=True)[:top_n]
    return set(ranked), ranked

def compute_top_nodes_from_windowed(windowed_node_csvs, top_n, t_max):
    """Select top_n sequences by total_flux summed across windowed node CSVs,
    for windows where window_start < t_max."""
    acc = {}
    ro = pac.ReadOptions(block_size=_BLOCK_BYTES)
    co = pac.ConvertOptions(include_columns=["window_start", "sequence", "total_flux"])

    for path in windowed_node_csvs:
        with pac.open_csv(path, read_options=ro, convert_options=co) as reader:
            for batch in reader:
                #Filter to the trimmed window
                mask = pc.less(batch.column("window_start"), t_max)
                batch = batch.filter(mask)
                if len(batch) == 0:
                    continue
                seqs = batch.column("sequence").to_pylist()
                fluxes = batch.column("total_flux").to_pylist()
                for seq, flux in zip(seqs, fluxes):
                    acc[seq] = acc.get(seq, 0.0) + flux
    ranked = sorted(acc, key=acc.__getitem__, reverse=True)[:top_n]
    return set(ranked), ranked

def _peek_columns(path):
    """Read the header of a CSV and return the column names."""
    ro = pac.ReadOptions(block_size=4096)
    with pac.open_csv(path, read_options=ro) as reader:
        batch = next(iter(reader))
        return batch.schema.names

def detect_file_type(columns):
    """Return 'edge', 'node', or 'unknown' based on column names."""
    col_set = set(columns)
    if EDGE_ID_COLS.issubset(col_set):
        return "edge"
    if NODE_ID_COLS.issubset(col_set):
        return "node"
    return "unknown"

###################
##CSV filtering
###################
def filter_csv(input_path, output_path, t_max, prune_columns, top_set):
    """Configurable windowed CSV trimming.
    Functionalities for time trimming, column pruning, and sequence filtering."""
    all_cols = _peek_columns(input_path)

    if "window_start" not in all_cols:
        raise ValueError(
            f"'{input_path}' has no 'window_start' column. "
            "Is this a windowed CSV from mutational_graph.py?"
        )

    file_type = detect_file_type(all_cols)
    if file_type == "unknown":
        raise ValueError(f"'{input_path}' could not be identified as an edge or node CSV.")

    #When pruning, skip parsing unused columns
    if prune_columns:
        target_cols = EDGE_KEEP_COLS if file_type == "edge" else NODE_KEEP_COLS
        include_cols = [c for c in target_cols if c in all_cols]
    else:
        include_cols = None

    ro = pac.ReadOptions(block_size=_BLOCK_BYTES)
    co = pac.ConvertOptions(include_columns=include_cols) if include_cols else pac.ConvertOptions()

    top_arr = pa.array(list(top_set)) if top_set is not None else None

    rows_read = 0
    rows_written = 0
    writer = None

    with pac.open_csv(input_path, read_options=ro, convert_options=co) as reader:
        for batch in reader:
            rows_read += len(batch)
            mask = None
            if t_max is not None:
                mask = pc.less(batch.column("window_start"), t_max)

            if top_arr is not None:
                if file_type == "edge":
                    seq_mask = pc.and_(pc.is_in(batch.column("seq_i"), value_set=top_arr), pc.is_in(batch.column("seq_j"), value_set=top_arr))
                else:
                    seq_mask = pc.is_in(batch.column("sequence"), value_set=top_arr)
                mask = seq_mask if mask is None else pc.and_(mask, seq_mask)

            if mask is not None:
                batch = batch.filter(mask)

            if len(batch) == 0:
                continue

            if writer is None:
                schema = pa.Table.from_batches([batch]).schema
                writer = pac.CSVWriter(output_path, schema)

            writer.write_table(pa.Table.from_batches([batch]))
            rows_written += len(batch)

    if writer:
        writer.close()
    else:
        header_cols = include_cols if include_cols else all_cols
        pa.Table.from_pydict({c: [] for c in header_cols}).to_pandas().to_csv(output_path, index=False)

    return rows_read, rows_written

def build_output_path(input_path, output_dir, suffix):
    dirname = output_dir or os.path.dirname(input_path) or "."
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    return os.path.join(dirname, f"{name}{suffix}{ext}")

###################
##CLI
###################
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Reduce windowed flux/node CSVs from mutational_graph.py by "
            "time-trimming, column pruning, and/or sequence filtering. "
        )
    )

    p.add_argument("--input", nargs="+", required=True, metavar="FILE", help="Windowed edge and/or node CSV files to process.")

    #Filtering
    p.add_argument("--t-max", type=float, default=None, help="Keep only windows where window_start < t-max. Omit to skip time filter.")
    p.add_argument("--prune-columns", action="store_true", help="Drop columns not used by the visualizer.")
    seq_group = p.add_mutually_exclusive_group()
    seq_group.add_argument("--top-nodes-file", default=None, metavar="PATH", help="Plain-text file of sequences to keep (one per line), as written by visualize_flux_graph --save-top-nodes.")
    seq_group.add_argument("--total-node-csv", nargs="+", default=None, metavar="PATH", help="Non-windowed node CSVs from mutational_graph. Ranks sequences by total_flux summed over the full simulation.")
    seq_group.add_argument("--windowed-node-csv", nargs="+", default=None, metavar="PATH", help="Windowed node CSVs. Ranks sequences by total_flux summed over windows where window_start < --t-max.")
    p.add_argument("--top-n", type=int, default=40, help="Number of top sequences to keep when using --total-node-csv or --windowed-node-csv (default: 40).")

    #Output
    p.add_argument("--save-top-nodes", default=None, metavar="PATH", help="Write the computed top-node list to this file.")
    p.add_argument("--output-dir", default=None, metavar="DIR", help="Directory for output files.")
    p.add_argument("--suffix", default="_compressed", help="Suffix appended to the filename stem of each output file")

    return p.parse_args()

def main():
    args = parse_args()

    if args.windowed_node_csv is not None and args.t_max is None:
        print("ERROR: --windowed-node-csv requires --t-max.")
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    top_set = None
    top_ranked = None

    if args.top_nodes_file:
        top_set, top_ranked = load_top_nodes_from_file(args.top_nodes_file)
    elif args.total_node_csv:
        top_set, top_ranked = compute_top_nodes_from_totals(args.total_node_csv, args.top_n)
    elif args.windowed_node_csv:
        top_set, top_ranked = compute_top_nodes_from_windowed(args.windowed_node_csv, args.top_n, args.t_max)

    if args.save_top_nodes:
        with open(args.save_top_nodes, "w") as f:
            for seq in top_ranked:
                f.write(seq + "\n")
        print(f" Top-node list saved to {args.save_top_nodes}")

    for input_path in args.input:
        if not os.path.isfile(input_path):
            print(f"WARNING: '{input_path}' not found — skipping.", file=sys.stderr)
            continue

        output_path = build_output_path(input_path, args.output_dir, args.suffix)

        filters = []
        if args.t_max is not None:
            filters.append(f"window_start < {args.t_max}")
        if args.prune_columns:
            filters.append("columns pruned")
        if top_set is not None:
            filters.append(f"sequences filtered to {len(top_set)}")
        filter_desc = ", ".join(filters) if filters else "no filters"

        print(f"\nProcessing: {input_path}")
        print(f"-> {output_path}")
        print(f"Filters: {filter_desc}")

        try:
            rows_read, rows_written = filter_csv(input_path=input_path, output_path=output_path, t_max=args.t_max, prune_columns=args.prune_columns, top_set=top_set)
            pct = 100.0 * rows_written / rows_read if rows_read else 0.0
            print(f"{rows_written:,} / {rows_read:,} rows kept ({pct:.1f}%)")
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()