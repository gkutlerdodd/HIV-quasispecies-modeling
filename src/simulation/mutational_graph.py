#!/usr/bin/env python3

"""
Integrates the ODEs of the quasispecies model and record the flux along each HD-1 edge in the mutational graph.

Output:
    Edge CSV: columns seq_i, seq_j, flux
        Average directed flux along each HD-1 edge throughout a simulation
    Node CSV: columns sequence, out_flux, in_flux, total_flux
        Average flux per node throughout a simulation.
    (Optionally) windowed CSVs showing fluxes through edges and nodes in time windows.

Usage:
    Measure one simulation run:
        python3 -m src.simulation.mutational_graph
        --df data/viral_sequences.csv (path to dataframe of binding affinities (receptor and optionally antibody)
        If recording windows: --window-size X
        If using antibody: --use-Ab --Ab-func [biphasic/constant] --ab-time 365 --ab-dose 2.8e-6

    Average runs:
        python3 -m src.simulation.mutational_graph --mode average-flux \
        --flux-csvs flux_seed1.csv flux_seed2.csv ... \
        --node-csvs nodes_seed1.csv nodes_seed2.csv ... (same order as --flux-csvs)

    Run python3 -m src.simulation.flux_graph -h for all options.
"""

import os
import csv
import argparse
import numpy as np
import pandas as pd

from src.utils import AA_ALPHABET, AA_TO_IDX, seq_to_lin, lin_to_seq, hamming_distance, df_to_ordered_array, iter_hd1_neighbors
from src.params import default_params

def prepare_df(df, N):
    if "sequence" not in df.columns:
        df["sequence"] = df.iloc[:, :N].astype(str).agg("".join, axis=1)
    if "lin_idx" not in df.columns:
        df["lin_idx"] = df["sequence"].map(seq_to_lin)
    return df

###################
##Simulation helpers
###################
def gibbs_multipliers(ddg_array, RT):
    """Compute exp(-ddG / RT) for an array of deltadeltaG values."""
    return np.exp(-ddg_array / RT)

def gibbs_multipliers_saturating(ddg_array, RT, h, K):
    """Compute saturating exp(-ddG / RT) for an array of deltadeltaG values."""
    mult = 1.0/((1.0/(1.0+h))**K)
    return mult * (1.0/(1.0 + h*np.exp(ddg_array / RT))) ** K

def _stochastic_threshold(fluxes, rng):
    """Apply stochastic thresholding to an array of mutation fluxes."""
    out = np.zeros_like(fluxes, dtype=np.float64)
    d_mask = fluxes >= 1.0
    out[d_mask] = fluxes[d_mask]
    s_mask = ~d_mask & (fluxes > 0.0)
    if np.any(s_mask):
        out[s_mask] = (rng.random(s_mask.sum()) < fluxes[s_mask]).astype(np.float64)
    return out

def compute_mutational_flux(I1_tensor, A, gamma, rng):
    """Compute the expected mutation flux gamma * mu * I1_tensor,
    where mu is the Kronecker product of the per-site mutation matrices A.
    Apply thresholding so that if the expected flux < 1, mutation happens
    stochastically with probability equal to the flux."""

    T = I1_tensor
    for axis in range(T.ndim):
        T = np.tensordot(A, T, axes=[1, axis])
        T = np.moveaxis(T, 0, axis)

    mutation_tensor = gamma * T
    return _stochastic_threshold(mutation_tensor, rng)

def biphasic_antibody_decay(p, t):
    """Compute the antibody concentration at time t with biphasic decay and
    administration(s)starting at time(s) Ab_time_on."""
    Ab = 0.0
    for t_dose in p['Ab_time_on']:
        if t >= t_dose:
            t_diff = t - t_dose
            Ab += p['Ab_max'] * (p['k'] * np.exp(-p['a1'] * t_diff) + (1 - p['k']) * np.exp(-p['a2'] * t_diff))

    return Ab

def constant_ab(p, t):
    Ab = 0.0
    for t_dose in p['Ab_time_on']:
        if t >= t_dose:
            Ab += p['Ab_max']
    return Ab

###################
##Mutational graph functions
###################
def score_nodes_from_edges(edge_df):
    """Compute per-node flux from a directed edge DataFrame.
    Returns DataFrame with columns: sequence, out_flux, in_flux, total_flux."""
    flux_out = edge_df.groupby("seq_i")["flux"].sum()
    flux_in = edge_df.groupby("seq_j")["flux"].sum()
    all_nodes = flux_out.index.union(flux_in.index)
    scores = pd.DataFrame({"sequence": all_nodes})
    scores["out_flux"] = scores["sequence"].map(flux_out).fillna(0.0)
    scores["in_flux"] = scores["sequence"].map(flux_in).fillna(0.0)
    scores["total_flux"] = scores["out_flux"] + scores["in_flux"]
    return scores.sort_values("total_flux", ascending=False).reset_index(drop=True)

def _write_window_to_csv(I1_integral_w, V_integral_w, duration, flux_mult, N, w_start, w_end, edge_writer, node_writer):
    """Compute and write one window's edge and node records."""
    nonzero = np.flatnonzero(I1_integral_w > 0.0)
    if nonzero.size == 0:
        return 0
    w_start_f = float(w_start)
    w_end_f   = float(w_end)
    out_flux_acc = {}
    in_flux_acc  = {}

    n_edges = 0
    for src in nonzero:
        flux_src = flux_mult * float(I1_integral_w[src]) / duration
        src_seq  = lin_to_seq(int(src), N)

        out_flux_acc[src_seq] = out_flux_acc.get(src_seq, 0.0) + flux_src

        for dst_seq in iter_hd1_neighbors(src_seq):
            edge_writer.writerow((w_start_f, w_end_f, src_seq, dst_seq, flux_src))
            in_flux_acc[dst_seq] = in_flux_acc.get(dst_seq, 0.0) + flux_src
            n_edges += 1

    all_seqs = set(out_flux_acc) | set(in_flux_acc)
    for seq in all_seqs:
        out_f = out_flux_acc.get(seq, 0.0)
        in_f = in_flux_acc.get(seq, 0.0)
        total = out_f + in_f
        lin = seq_to_lin(seq)
        abund = float(V_integral_w[lin]) / duration if lin < len(V_integral_w) else 0.0
        node_writer.writerow((w_start_f, w_end_f, seq, out_f, in_f, total, abund))

    return n_edges

def _open_windowed_csvs(windowed_edge_csv, windowed_node_csv):
    """Open windowed output files and write their headers."""
    edge_fh = open(windowed_edge_csv, "w", newline="", buffering=1 << 20)
    edge_w = csv.writer(edge_fh)
    edge_w.writerow(("window_start", "window_end", "seq_i", "seq_j", "flux"))
    node_fh = open(windowed_node_csv, "w", newline="", buffering=1 << 20)
    node_w = csv.writer(node_fh)
    node_w.writerow(("window_start", "window_end", "sequence", "out_flux", "in_flux", "total_flux", "total_abundance"))

    return edge_fh, edge_w, node_fh, node_w

def _save_checkpoint(path, I1_integral, V_integral, step, t_max, dt):
    """Save a numpy checkpoint of the running integrals.
    Output is .npz with keys: I1_integral, V_integral, step, t_max, dt"""
    np.savez_compressed(path, I1_integral=I1_integral, V_integral=V_integral, step=np.array(step), t_max=np.array(t_max), dt=np.array(dt))

def measure_flux_graph(df, N, mu, params, t_max=100, dt=0.01, ddg_col='ddG', binding_affinity_shape='none',
    use_Ab=False, Ab_func=None, ddg_col_Ab='ddG_Ab', WT_seq='DIRQH', rng=None, edge_csv="flux_edges.csv", node_csv="node_scores.csv",
    window_size=None, windowed_edge_csv=None, windowed_node_csv=None, flush_every=100, checkpoint_interval=500,
    checkpoint_path=None, progress_every=100):
    """Simulate the quasispecies model write windowed flux records to disk."""

    if rng is None:
        rng = np.random.default_rng()

    use_windows = (window_size is not None)
    df = prepare_df(df, N)
    ddg_array = df_to_ordered_array(df, col=ddg_col, N=N)

    A = np.full((20, 20), mu / 19)
    np.fill_diagonal(A, 1 - mu)

    tensor_shape = (20,) * N
    n_variants = 20 ** N

    if binding_affinity_shape == 'exponential':
        beta_mult = gibbs_multipliers(ddg_array, params['RT'])
    else:
        beta_mult = gibbs_multipliers_saturating(ddg_array, params['RT'], params['h'], params['K'])

    if use_Ab:
        ddg_Ab_array = df_to_ordered_array(df, col=ddg_col_Ab, N=N)
        if not Ab_func:
            raise ValueError("Must include antibody function when use_Ab=True.")
        theta_mult = gibbs_multipliers(ddg_Ab_array, params['RT']) / params['KD_WT']

    #State initialization
    T = params['lambda_T'] / params['d']
    I1 = np.zeros(n_variants, dtype=np.float32)
    I2 = np.zeros(n_variants, dtype=np.float32)
    V = np.zeros(n_variants, dtype=np.float32)
    wt_idx = seq_to_lin(WT_seq)
    V[wt_idx] = 1e2

    I1_integral = np.zeros(n_variants, dtype=np.float64)
    V_integral  = np.zeros(n_variants, dtype=np.float64)

    flux_mult = params["gamma"] * (mu / 19.0) * (1.0 - mu) ** (N - 1)
    n_steps = int(np.ceil(t_max / dt))

    #Set up window steps
    if use_windows:
        flush_steps = {}
        w_idx = 1
        while True:
            step_at_end = int(round(w_idx * window_size / dt))
            if step_at_end > n_steps:
                break
            flush_steps[step_at_end] = ((w_idx - 1) * window_size, w_idx * window_size)
            w_idx += 1

        I1_integral_w = np.zeros(n_variants, dtype=np.float64)
        V_integral_w = np.zeros(n_variants, dtype=np.float64)
        current_w_start = 0.0
        windows_written = 0

        if windowed_edge_csv is None or windowed_node_csv is None:
            raise ValueError("windowed_edge_csv and windowed_node_csv must be provided when window_size is set.")

        edge_fh, edge_w, node_fh, node_w = _open_windowed_csvs(windowed_edge_csv, windowed_node_csv)

        ckpt_path = checkpoint_path or (node_csv + ".ckpt.npz")

    for step in range(n_steps + 1):
        time = step * dt
        if use_Ab:
            Ab = Ab_func(params, time)

        #Accumulate running integrals
        I1_f64 = I1.astype(np.float64)
        V_f64  = V.astype(np.float64)
        I1_integral += I1_f64 * dt
        V_integral  += V_f64  * dt
        if use_windows:
            I1_integral_w += I1_f64 * dt
            V_integral_w  += V_f64  * dt

        if step == n_steps:
            break

        #Flush windows
        if use_windows and step in flush_steps:
            w_start_flush, w_end_flush = flush_steps[step]
            duration = w_end_flush - w_start_flush

            _write_window_to_csv(I1_integral_w, V_integral_w, duration, flux_mult, N, w_start_flush, w_end_flush, edge_w, node_w)

            I1_integral_w[:] = 0.0
            V_integral_w[:]  = 0.0
            current_w_start  = w_end_flush
            windows_written += 1

            if flush_every > 0 and windows_written % flush_every == 0:
                edge_fh.flush()
                node_fh.flush()

            if (checkpoint_interval > 0 and windows_written % checkpoint_interval == 0):
                _save_checkpoint(ckpt_path, I1_integral, V_integral, step, t_max, dt)

            if progress_every > 0 and windows_written % progress_every == 0:
                pct = 100.0 * step / n_steps
                print(
                    f"  [window {windows_written}]  "
                    f"t={w_end_flush:.1f}/{t_max}  ({pct:.1f}%)  "
                    f"sum_V={V.sum():.3g}",
                    flush=True,
                )

        #Model steps
        if use_Ab:
            theta_vector = 1.0 / (1.0 + Ab * theta_mult)
            infection_vector = params['beta_wt'] * beta_mult * theta_vector * V
        else:
            infection_vector = params['beta_wt'] * beta_mult * V

        dT  = params['lambda_T'] - params['d'] * T - np.sum(infection_vector) * T
        dI1 = infection_vector * T - I1 * (params['gamma'] + params['delta1'])
        mutation_flux = compute_mutational_flux(I1.reshape(tensor_shape), A, params['gamma'], rng).ravel()
        dI2 = mutation_flux - params['delta2'] * I2
        dV = params['n'] * params['delta2'] * I2 - params['c'] * V

        T = max(T + dT * dt, 0.0)
        I1 = np.maximum(I1 + dI1 * dt, 0.0)
        I2 = np.maximum(I2 + dI2 * dt, 0.0)
        V = np.maximum(V  + dV  * dt, 0.0)

    if use_windows and I1_integral_w.sum() > 0.0:
        duration = t_max - current_w_start
        if duration > 0:
            _write_window_to_csv(I1_integral_w, V_integral_w, duration, flux_mult, N, current_w_start, t_max, edge_w, node_w)
            windows_written += 1

        if use_windows:
            edge_fh.flush(); edge_fh.close()
            node_fh.flush(); node_fh.close()
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    #Write total windowed flux records
    nonzero = np.flatnonzero(I1_integral > 0.0)
    records = []
    for src in nonzero:
        flux_src = flux_mult * float(I1_integral[src]) / t_max
        src_seq  = lin_to_seq(int(src), N)
        for dst_seq in iter_hd1_neighbors(src_seq):
            records.append({"seq_i": src_seq, "seq_j": dst_seq, "flux": flux_src})

    edge_df = (pd.DataFrame(records).sort_values("flux", ascending=False).reset_index(drop=True))
    edge_df.to_csv(edge_csv, index=False)

    node_df = score_nodes_from_edges(edge_df)
    node_df["lin_idx"] = node_df["sequence"].map(seq_to_lin)
    node_df["total_abundance"] = (V_integral[node_df["lin_idx"].astype(int).values] / t_max)
    node_df = node_df.drop(columns=["lin_idx"])
    node_df.to_csv(node_csv, index=False)

def average_flux_graphs(flux_csvs, output_edge_csv, node_csvs=None, output_node_csv=None):
    """Average flux graphs from multiple simulation runse."""
    n_sims  = len(flux_csvs)
    edge_acc = {}

    for path in flux_csvs:
        df = pd.read_csv(path)
        grouped = df.groupby(["seq_i", "seq_j"], as_index=False)["flux"].sum()
        for row in grouped.itertuples(index=False):
            key = (row.seq_i, row.seq_j)
            edge_acc[key] = edge_acc.get(key, 0.0) + float(row.flux)
        del df, grouped

    avg_edges = pd.DataFrame([(si, sj, f / n_sims) for (si, sj), f in edge_acc.items()], columns=["seq_i", "seq_j", "flux"],).sort_values("flux", ascending=False).reset_index(drop=True)
    avg_edges.to_csv(output_edge_csv, index=False)

    if node_csvs is not None:
        if len(node_csvs) != n_sims:
            raise ValueError(
                f"node_csvs length ({len(node_csvs)}) must match "
                f"flux_csvs length ({n_sims})."
            )
        numeric_cols = ["out_flux", "in_flux", "total_flux", "total_abundance"]
        node_acc = {}
        for path in node_csvs:
            df = pd.read_csv(path, usecols=["sequence"] + numeric_cols)
            grouped = df.groupby("sequence", as_index=False)[numeric_cols].sum()
            for row in grouped.itertuples(index=False):
                seq = row.sequence
                if seq not in node_acc:
                    node_acc[seq] = {col: 0.0 for col in numeric_cols}
                for col in numeric_cols:
                    node_acc[seq][col] += float(getattr(row, col))
            del df, grouped

        avg_nodes = pd.DataFrame(
            [{"sequence": seq, **{col: vals[col] / n_sims for col in numeric_cols}} for seq, vals in node_acc.items()]).sort_values("total_flux", ascending=False).reset_index(drop=True)
        avg_nodes.to_csv(output_node_csv, index=False)
        return avg_edges, avg_nodes

    return avg_edges


def _to_str(t):
    """Create short label for time or dose (0 -> '0', 0.5 -> '0p5', etc)."""
    return f"{t:.6g}".replace(".", "p")


def parse_args():
    """Read arguments from the CLI."""
    p = argparse.ArgumentParser(
        description="Mutational graph helper: measure flux, or average across runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["measure", "average-flux"], default="measure",
                   help="'measure': run simulation and save flux graph. "
                        "'average-flux': average multiple flux CSVs.")
    p.add_argument("--df", required=False, help="Input variant CSV. Columns: N AA columns + ddG (+ optional ddG_Ab).")
    p.add_argument("--out-dir", default="graph_outputs", help="Directory to write results.")
    p.add_argument("--N", type=int, default=5, help="Sequence length (N).")
    p.add_argument("--mu", type=float, default=0.0001, help="Per-site mutation rate mu.")
    p.add_argument("--t-max", type=float, default=365.0, help="Simulation end time.")
    p.add_argument("--dt", type=float, default=0.01, help="Euler time step.")
    p.add_argument("--use-ab", action="store_true", help="Enable antibody dynamics.")
    p.add_argument("--ab-func", choices=["none", "constant", "biphasic"], default="constant", help="Antibody decay function.")
    p.add_argument("--ab-time", type=float, nargs="+", default=[0.0], metavar="T", help="Antibody administration time(s).")
    p.add_argument("--ab-dose", type=float, default=2.8e-6, help="Antibody dose (Ab_max).")
    p.add_argument("--binding-affinity-shape", choices=["saturating", "exponential"], default="saturating", help="Shape of the receptor binding affinity-ddG mapping.")
    p.add_argument("--wt-seq", default="DIRQH", help="Wild-type sequence.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed.")
    p.add_argument("--edge-csv", default=None, help="Output path for the total-simulation edge CSV. Defaults to flux_{label}.csv inside --out-dir.")
    p.add_argument("--node-csv", default=None, help="Output path for the total-simulation node CSV. Defaults to nodes_{label}.csv inside --out-dir.")
    p.add_argument("--window-size", type=float, default=None, help="Width of each time window. Windowed CSVs are written incrementally; total CSVs are unaffected.")
    p.add_argument("--windowed-edge-csv", default=None, help="Output path for windowed edge CSV. Defaults to windowed_flux_{label}.csv.")
    p.add_argument("--windowed-node-csv", default=None, help="Output path for windowed node CSV. Defaults to windowed_nodes_{label}.csv.")
    p.add_argument("--flux-csvs", nargs="+", default=None, help="Flux CSV files to average (average-flux mode).")
    p.add_argument("--averaged-flux-csv", default=None, help="Output path for averaged edge CSV.")
    p.add_argument("--node-csvs", nargs="+", default=None, help="Node score CSV files to average (average-flux mode).")
    p.add_argument("--averaged-node-csv", default=None, help="Output path for averaged node CSV.")
    p.add_argument("--flush-every", type=int, default=100, help="Flush windowed CSV files to disk every N windows.")
    p.add_argument("--checkpoint-interval", type=int, default=500, help="Save a .npz checkpoint of running integrals every N windows (0 = no checkpointing)")
    p.add_argument("--checkpoint-path", default=None, help="Path for the integral checkpoint file. Defaults to <node-csv>.ckpt.npz.")
    p.add_argument("--progress-every", type=int, default=100, help="Print a progress line every N windows (0 = silent).")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    #Averaging flux graphs
    if args.mode == "average-flux":
        if not args.flux_csvs:
            raise ValueError("--flux-csvs is required for average-flux mode.")
        out_edges = args.averaged_flux_csv or os.path.join(args.out_dir, "averaged_flux.csv")
        out_nodes = None
        if args.node_csvs is not None:
            out_nodes = args.averaged_node_csv or os.path.join(args.out_dir, "averaged_nodes.csv")
        average_flux_graphs(flux_csvs=args.flux_csvs, output_edge_csv=out_edges, node_csvs=args.node_csvs, output_node_csv=out_nodes,)
        return

    if args.df is None:
        raise ValueError("--df is required for measure mode.")

    df = pd.read_csv(args.df, index_col=0)
    df = prepare_df(df, args.N)

    params = default_params()
    params['Ab_time_on'] = args.ab_time
    params['Ab_max'] = args.ab_dose

    Ab_func = None
    if args.use_ab:
        Ab_func = biphasic_antibody_decay if args.ab_func == "biphasic" else constant_ab

    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    parts = ([f"time{'_'.join(_to_str(t) for t in args.ab_time)}", f"dose{_to_str(args.ab_dose)}"] if args.use_ab else ["noAb"])
    if args.seed is not None:
        parts.append(f"seed{args.seed}")
    label = "_".join(parts)

    edge_csv = args.edge_csv or os.path.join(args.out_dir, f"flux_{label}.csv")
    node_csv = args.node_csv or os.path.join(args.out_dir, f"nodes_{label}.csv")

    windowed_edge_csv = windowed_node_csv = None
    if args.window_size is not None:
        windowed_edge_csv = args.windowed_edge_csv or os.path.join(args.out_dir, f"windowed_flux_{label}.csv")
        windowed_node_csv = args.windowed_node_csv or os.path.join(args.out_dir, f"windowed_nodes_{label}.csv")

    measure_flux_graph(
        df=df, N=args.N, mu=args.mu, params=params,
        t_max=args.t_max, dt=args.dt,
        binding_affinity_shape=args.binding_affinity_shape,
        use_Ab=args.use_ab, Ab_func=Ab_func,
        WT_seq=args.wt_seq, rng=rng,
        edge_csv=edge_csv, node_csv=node_csv,
        window_size=args.window_size,
        windowed_edge_csv=windowed_edge_csv,
        windowed_node_csv=windowed_node_csv,
        flush_every=args.flush_every,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint_path,
        progress_every=args.progress_every
    )

if __name__ == "__main__":
    main()
