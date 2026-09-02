#!/usr/bin/env python3

"""
Integrates the ODEs of the HIV quasispecies evolution model along with options for dimensonality reduction.

Output: For each simulation run, a summary of each timepoint (file totals_{time_Ab_introduced}_{ab_dose}_{seed}.csv)
        and the abundances of the top K variants at that timepoint (file topk_{time_Ab_introduced}_{ab_dose}_{seed}.csv) is saved.
        Totals columns: time, T (number of uninfected target cells), sum_I1 (sum of all infected cells in I1), sum_I2, sum_V (total viral load), Ab (ab dose if included)
        Topk columns: time, sequence (AA sequence of variant at mutable residues), I1 (number of I1 cells infected by that variant), I2, V, Ab

Usage:
    python3 -m src.simulation.simulate
        --df data/viral_sequences.csv (path to dataframe of binding affinities (receptor and optionally antibody)
        --out-dir sim_outputs --t-max 730 (days in model time to run the simulation for)
        If using antibody: --use-Ab --Ab-func [biphasic/constant] --ab-time 365 --ab-dose 2.8e-6
        If reducing dimensionality: --mode [full/reduced-other/reduced-reject] --track-max-hd 1
            If using strongest receptor-binding variants: --track-top-k N
            If using Hamming distance to the wild type: --track-max-hd N
            If using a graph-selected set of variants: --graph-node-scores data/graph_node_scores.csv --graph-track-k N --graph-prune-mode [total/influx/efflux/abundance]
            If using a frequency-selected set of variants: --freq-file data/frequency_data.csv --freq-threshold X
    Run python3 -m src.simulation.simulate -h for all options.
"""

import os
import argparse
import numpy as np
import pandas as pd

from src.utils import AA_ALPHABET, AA_TO_IDX, seq_to_lin, lin_to_seq, hamming_distance, df_to_ordered_array
from src.params import default_params

def prepare_df(df, N):
    if "sequence" not in df.columns:
        df["sequence"] = df.iloc[:, :N].astype(str).agg("".join, axis=1)
    if "lin_idx" not in df.columns:
        df["lin_idx"] = df["sequence"].map(seq_to_lin)
    return df

def select_tracked_variants(df, wt_seq, track_max_hd=None, track_top_k=None):
    """Select variants to track based on sequence distance to the WT sequence, or top-k variants.
    Includes:
        - all variants with Hamming distance ≤ track_max_hd from wt_seq (if set)
        - the top track_top_k variants by stongest receptor binding (if set)
        - the WT sequence
    """
    if track_max_hd is None and track_top_k is None:
        raise ValueError("Must set at least one of --track_max_hd or --track_top_k.")

    keep = np.zeros(len(df), dtype=bool)

    if track_max_hd is not None:
        keep |= df["sequence"].apply(lambda s: hamming_distance(s, wt_seq)) <= track_max_hd

    if track_top_k is not None:
        top_idx = df["ddG"].nsmallest(track_top_k).index
        keep |= df.index.isin(top_idx)

    keep |= (df["sequence"] == wt_seq)

    tracked_df = df.loc[keep].copy().reset_index(drop=True)
    other_df = df.loc[~keep].copy().reset_index(drop=True)
    return tracked_df, other_df

def select_tracked_variants_by_sequences(df, N, wt_seq, keep_sequences):
    """Select a tracked set from a list of sequences, always including the wild type."""
    df = prepare_df(df.copy(), N)
    keep_set = set(keep_sequences)
    keep_set.add(wt_seq)
    keep = df["sequence"].isin(keep_set)
    tracked_df = df.loc[keep].copy().reset_index(drop=True)
    other_df = df.loc[~keep].copy().reset_index(drop=True)
    missing = keep_set - set(tracked_df["sequence"].tolist())
    if missing:
        raise ValueError(f"Graph-selected sequences missing from variant dataframe: {missing}.")

    return tracked_df, other_df

def select_tracked_variants_by_frequency(df, N, wt_seq, freq_file, freq_threshold):
    """Select a tracked set from sequences with max_frequency >= freq_threshold,
    always including the wild type."""
    df = prepare_df(df.copy(), N)
    freq_df = pd.read_csv(freq_file)
    keep_seqs = set(freq_df.loc[freq_df["max_frequency"] >= freq_threshold, "sequence"].tolist())
    keep_seqs.add(wt_seq)
    keep = df["sequence"].isin(keep_seqs)
    tracked_df = df.loc[keep].copy().reset_index(drop=True)
    other_df = df.loc[~keep].copy().reset_index(drop=True)

    return tracked_df, other_df

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
    """Apply stochastic thresholding to an array of mutation fluxes. Fluxes >= 1 are kept deterministically,
    entries [0,1) are thresholded with probability equal to their value."""
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

def compute_p_within(mask_full, A, N):
    """Precompute probability a variant mutates into the tracked set (for reject flux mode)"""
    T = mask_full.astype(np.float32).reshape((20,) * N)
    for axis in range(N):
        T = np.tensordot(A.T, T, axes=[1, axis])
        T = np.moveaxis(T, 0, axis)
    return T.ravel()

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
##Simulation functions
###################

def simulate(df, N, mu, params, t_max=100, dt=0.01, binding_affinity_shape='none', ddg_col='ddG', use_Ab=False,
             Ab_func=None, ddg_col_Ab='ddG_Ab', WT_seq='DIRQH', rng=None, n_measurements=100, top_k=250,
             include_wt=True, totals_name="totals", topk_name="topk"):
    """Euler integrator for the quasispecies model. Considers mutations between all virus sequences with N mutable
    residues (=20^N). Uses ddG for receptor-binding for each virus mutant (and optionally ddG of antibody binding).
    Takes Euler steps of size dt and saves the abundances of the top_k most abundant variants at each measurement point."""
    if rng is None:
        rng = np.random.default_rng()

    df = prepare_df(df, N)
    ddg_array = df_to_ordered_array(df, col=ddg_col, N=N)

    #Build per-site mutation matrix
    A = np.full((20, 20), mu/19)
    np.fill_diagonal(A, 1-mu)

    tensor_shape = (20,)*N
    n_variants = 20**N

    if binding_affinity_shape == 'exponential':
        beta_mult = gibbs_multipliers(ddg_array, params['RT'])
    else:
        beta_mult = gibbs_multipliers_saturating(ddg_array, params['RT'], params['h'], params['K'])

    #If an antibody will be added during the simulation, precompute strain-specific constants
    if use_Ab:
        ddg_Ab_array = df_to_ordered_array(df, col=ddg_col_Ab, N=N)

        if not Ab_func:
            raise ValueError("Must include antibody function")

        #The fraction of bound virions theta is given by 1/(1 + (Ab/KD,WT) * exp(-ddG,i/RT); here we compute the constant exp(-ddG,i/RT)/KD,WT
        theta_mult = gibbs_multipliers(ddg_Ab_array, params['RT']) / params['KD_WT']

    #Initialize ODEs from the uninfected steady state and adding a small amount of wild-type virus
    T = params['lambda_T'] / params['d']
    I1 = np.zeros(n_variants, dtype=np.float32)
    I2 = np.zeros(n_variants, dtype=np.float32)
    V = np.zeros(n_variants, dtype=np.float32)
    wt_idx = seq_to_lin(WT_seq)
    V[wt_idx] = 10**2

    n_steps = int(np.ceil(t_max / dt))
    record_steps = set(np.linspace(0, n_steps, n_measurements, dtype=int).tolist())
    totals_records = []
    topk_records = []

    #Euler integration
    for step in range(n_steps + 1):
        time = step * dt
        if use_Ab:
            Ab = Ab_func(params, time)
        if step in record_steps:
            #Record summary totals
            sum_I1 = I1.sum()
            sum_I2 = I2.sum()
            sum_V = V.sum()
            totals_record = {
                "time": time,
                "T": T,
                "sum_I1": sum_I1,
                "sum_I2": sum_I2,
                "sum_V": sum_V
            }
            if use_Ab:
                totals_record["Ab"] = Ab
            totals_records.append(totals_record)

            #Get the top K variants
            part = np.argpartition(V, -top_k)[-top_k:]
            chosen_idx = part[np.argsort(V[part])[::-1]]

            #If include_WT=True, always record the abundance of the wild-type sequence
            if include_wt and (wt_idx not in chosen_idx):
                chosen_idx = np.concatenate(([wt_idx], chosen_idx)).astype(np.int64)

            #Record abundances for chosen variants
            for idx in chosen_idx:
                topk_record = {
                    "time": time,
                    "sequence": lin_to_seq(idx, N),
                    "I1": I1[idx],
                    "I2": I2[idx],
                    "V": V[idx]
                }
                if use_Ab:
                    topk_record['Ab'] = Ab
                topk_records.append(topk_record)

        if step == n_steps:
            break

        if use_Ab:
            theta_vector = 1.0/(1.0 + Ab * theta_mult)
            infection_vector = params['beta_wt'] * beta_mult * theta_vector * V
        else:
            infection_vector = params['beta_wt'] * beta_mult * V

        dT = params['lambda_T'] - params['d'] * T - np.sum(infection_vector) * T
        dI1 = infection_vector * T - I1 * (params['gamma'] + params['delta1'])
        mutation_flux = compute_mutational_flux(I1.reshape(tensor_shape), A, params['gamma'], rng).ravel()
        dI2 = mutation_flux - params['delta2'] * I2
        dV = params['n'] * params['delta2'] * I2 - params['c'] * V

        #Euler step
        T = max(T + dT * dt, 0.0)
        I1 = np.maximum(I1 + dI1 * dt, 0.0)
        I2 = np.maximum(I2 + dI2 * dt, 0.0)
        V = np.maximum(V + dV * dt, 0.0)

    totals_df = pd.DataFrame.from_records(totals_records)
    topk_df = pd.DataFrame.from_records(topk_records)

    totals_df.to_csv(f"{totals_name}.csv", index=False)
    topk_df.to_csv(f"{topk_name}.csv", index=False)

def simulate_reduced(df, N, mu, params, t_max=100, dt=0.01, binding_affinity_shape='none', ddg_col='ddG', use_Ab=False, Ab_func=None,
                   ddg_col_Ab='ddG_Ab', WT_seq='DIRQH', rng=None, n_measurements=100, top_k=250, include_wt=True,
                   totals_name="totals", topk_name="topk", track_max_hd=None, track_top_k=None, mode='other', tracked_df=None, other_df=None):
    """Euler integrator for the quasispecies model with reduced dimensionality. Considers mutations between a SUBSET of all virus
    sequences with N mutable residues (=20^N), based on receptor-binding affinity and/or Hamming distance to the wild-type sequence.
    Takes Euler steps of size dt and saves the abundances of the top_k most abundant variants at each measurement point."""
    if rng is None:
        rng = np.random.default_rng()

    #Build tracked set
    df = prepare_df(df, N)
    if tracked_df is None or other_df is None:
        tracked_df, other_df = select_tracked_variants(df, WT_seq, track_max_hd=track_max_hd, track_top_k=track_top_k)

    n_t = len(tracked_df)
    tracked_lin = tracked_df['lin_idx'].values.astype(np.int64)
    has_other = (mode == 'other') and (len(other_df) > 0)
    n_bins = n_t + int(has_other)
    top_k = min(n_bins, top_k)

    #Build per-site mutation matrix
    A = np.full((20, 20), mu/19)
    np.fill_diagonal(A, 1-mu)

    tensor_shape = (20,)*N
    n_variants = 20**N

    if binding_affinity_shape == 'exponential':
        beta_mult_t = gibbs_multipliers(tracked_df[ddg_col].values.astype(np.float64), params['RT'])
    else:
        beta_mult_t = gibbs_multipliers_saturating(tracked_df[ddg_col].values.astype(np.float64), params['RT'], params['h'], params['K'])

    #If an antibody will be added during the simulation, precompute strain-specific constants
    if use_Ab:
        if not Ab_func:
            raise ValueError("Must include antibody function")
        theta_mult_t = (gibbs_multipliers(tracked_df[ddg_col_Ab].values.astype(np.float64), params['RT'])/ params['KD_WT'])

    #Other bin effective multipliers
    beta_mult_o = theta_mult_o = 0.0
    if has_other:
        if binding_affinity_shape == 'exponential':
            beta_mult_o = gibbs_multipliers(other_df[ddg_col].mean(), params['RT'])
        else:
            beta_mult_o = gibbs_multipliers_saturating(other_df[ddg_col].mean(), params['RT'], params['h'], params['K'])

        if use_Ab:
            theta_mult_o = (float(np.exp(-other_df[ddg_col_Ab].mean() / params['RT'])) / params['KD_WT'])

    #Precompute p_within for reject mode
    p_other_t = None
    if mode == 'reject':
        mask_full = np.zeros(n_variants, dtype=np.float32)
        mask_full[tracked_lin] = 1.0
        p_within_full = compute_p_within(mask_full, A, N)
        p_other_t = np.maximum(1.0 - p_within_full[tracked_lin], 0.0)

    #Initialize ODEs from the uninfected steady state and adding a small amount of wild-type virus
    T = params['lambda_T'] / params['d']
    I1_t = np.zeros(n_t, dtype=np.float32)
    I2_t = np.zeros(n_t, dtype=np.float32)
    V_t = np.zeros(n_t, dtype=np.float32)
    wt_idx = seq_to_lin(WT_seq)
    wt_local_idx = int(np.where(tracked_lin == wt_idx)[0][0])
    V_t[wt_local_idx] = 10**2
    I1_o = I2_o = V_o = 0.0
    I1_full = np.zeros(n_variants, dtype=np.float32)

    n_steps = int(np.ceil(t_max / dt))
    record_steps = set(np.linspace(0, n_steps, n_measurements, dtype=int).tolist())
    totals_records = []
    topk_records = []

    #Euler integration
    for step in range(n_steps + 1):
        time = step * dt
        if use_Ab:
            Ab = Ab_func(params, time)
        if step in record_steps:
            #Record summary totals
            sum_I1 = float(I1_t.sum()) + I1_o
            sum_I2 = float(I2_t.sum()) + I2_o
            sum_V = float(V_t.sum()) + V_o
            totals_record = {
                "time": time,
                "T": T,
                "sum_I1": sum_I1,
                "sum_I2": sum_I2,
                "sum_V": sum_V
            }
            if use_Ab:
                totals_record["Ab"] = Ab
            totals_records.append(totals_record)

            #Get the top K variants (including OTHER if present and in the top K)
            if has_other:
                all_V  = np.concatenate([V_t,  [V_o]])
                other_local_idx = n_t
            else:
                all_V = V_t
                other_local_idx = None
            part = np.argpartition(all_V, -top_k)[-top_k:]
            chosen_idx = part[np.argsort(all_V[part])[::-1]]

            if include_wt and (wt_local_idx not in chosen_idx):
                chosen_idx = np.concatenate([[wt_local_idx], chosen_idx])

            #Record abundances for chosen variants
            for idx in chosen_idx:
                if has_other and idx == other_local_idx:
                    rec = {
                        "time": time,
                        "sequence": "other",
                        "I1": I1_o,
                        "I2": I2_o,
                        "V": V_o
                    }
                else:
                    rec = {
                        "time": time,
                        "sequence": lin_to_seq(tracked_lin[idx], N),
                        "I1": I1_t[idx],
                        "I2": I2_t[idx],
                        "V": V_t[idx]
                    }
                if use_Ab:
                    rec['Ab'] = Ab
                topk_records.append(rec)

        if step == n_steps:
            break

        if use_Ab:
            theta_vector_t = 1.0/(1.0 + Ab * theta_mult_t)
            infection_vector_t = params['beta_wt'] * beta_mult_t * theta_vector_t * V_t
            if has_other:
                theta_o = 1.0 / (1.0 + Ab * theta_mult_o)
                infection_vector_o = params['beta_wt'] * beta_mult_o * theta_o * V_o
            else:
                infection_vector_o = 0.0
        else:
            infection_vector_t = params['beta_wt'] * beta_mult_t * V_t
            infection_vector_o = (params['beta_wt'] * beta_mult_o * V_o) if has_other else 0.0
        #Mutational fluxes
        #Embed tracked I1 into full tensor
        I1_full[:] = 0.0
        I1_full[tracked_lin] = I1_t.astype(np.float32)

        #Compute mutational fluxes for the full state space
        T_flux = I1_full.reshape(tensor_shape)
        for axis in range(N):
            T_flux = np.tensordot(A, T_flux, axes=[1, axis])
            T_flux = np.moveaxis(T_flux, 0, axis)
        T_flux = T_flux.ravel()
        flux_tracked_raw = params['gamma'] * T_flux[tracked_lin].astype(np.float64)
        mut_to_tracked = _stochastic_threshold(flux_tracked_raw, rng)

        #Track escaped flux
        if mode == 'other':
            escaped_raw_total = params['gamma'] * float(I1_t.sum()) - float(flux_tracked_raw.sum())
            escaped_raw_total = max(escaped_raw_total, 0.0)
            mut_to_other_raw = escaped_raw_total + params['gamma'] * I1_o
            mut_to_other = float(_stochastic_threshold(np.array([mut_to_other_raw]), rng)[0])

        elif mode == 'reject':
            escaped_raw = params['gamma'] * p_other_t * I1_t
            escaped = _stochastic_threshold(escaped_raw, rng)
            mut_to_tracked = mut_to_tracked + escaped

        total_inf = infection_vector_t.sum() + infection_vector_o
        dT = params['lambda_T'] - params['d'] * T - total_inf * T
        dI1_t = infection_vector_t * T - I1_t * (params['gamma'] + params['delta1'])
        dI2_t = mut_to_tracked - params['delta2'] * I2_t
        dV_t = params['n'] * params['delta2'] * I2_t - params['c'] * V_t

        if has_other:
            dI1_o = infection_vector_o * T - I1_o * (params['gamma'] + params['delta1'])
            dI2_o = mut_to_other - params['delta2'] * I2_o
            dV_o  = params['n'] * params['delta2'] * I2_o - params['c'] * V_o

        #Euler step
        T = max(T + dT * dt, 0.0)
        I1_t = np.maximum(I1_t + dI1_t * dt, 0.0)
        I2_t = np.maximum(I2_t + dI2_t * dt, 0.0)
        V_t = np.maximum(V_t + dV_t * dt, 0.0)
        if has_other:
            I1_o = max(I1_o + dI1_o * dt, 0.0)
            I2_o = max(I2_o + dI2_o * dt, 0.0)
            V_o  = max(V_o  + dV_o  * dt, 0.0)

    totals_df = pd.DataFrame.from_records(totals_records)
    topk_df = pd.DataFrame.from_records(topk_records)

    totals_df.to_csv(f"{totals_name}.csv", index=False)
    topk_df.to_csv(f"{topk_name}.csv", index=False)

def _to_str(t):
    """Create short label for time or dose (0 -> '0', 0.5 -> '0p5', etc)."""
    s = f"{t:.6g}"
    return s.replace(".", "p")

def parse_args():
    """Read arguments from the CLI."""
    p = argparse.ArgumentParser(description="Run single Euler simulation (one config per process).")
    p.add_argument("--df", required=True, help="Input dataframe CSV (variants). Columns: N AA columns then ddG (and optionally ddG_Ab).")
    p.add_argument("--out-dir", default="sim_outputs", help="Directory to write results.")
    p.add_argument("--N", type=int, default=5, help="Sequence length (N).")
    p.add_argument("--mu", type=float, default=0.0001, help="Per-site mutation rate mu.")
    p.add_argument("--t-max", type=float, default=365.0, help="Simulation end time.")
    p.add_argument("--dt", type=float, default=0.01, help="Euler time step.")
    p.add_argument("--n-measurements", type=int, default=365, help="How many measurement timepoints to record.")
    p.add_argument("--top-k", type=int, default=500, help="top-k recorded variants to save at each measurement.")
    p.add_argument("--use-ab", action="store_true", help="Enable antibody dynamics.")
    p.add_argument("--ab-func", choices=["none", "constant", "biphasic"], default="constant", help="Antibody function.")
    p.add_argument("--ab-time", type=float, nargs="+", default=[0.0], metavar="T", help="Antibody administration time(s).")
    p.add_argument("--ab-dose", type=float, default=2.8e-6, help="Antibody dose (Ab_max).")
    p.add_argument("--binding-affinity-shape", choices=["saturating", "exponential"], default="saturating", help="Shape of the receptor binding affinity-ddG mapping.")
    p.add_argument("--wt-seq", default="DIRQH", help="Wild-type sequence (length N).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    p.add_argument("--record-wt", action="store_true", help="Record the abundance of the wild-type sequence in topk at each measurement.")
    p.add_argument("--mode", choices=["full", "reduced-other", "reduced-reject"], default="full",
                   help=("'full': all 20^N variants. "
                         "'reduced-other': tracked set + absorbing OTHER bin. "
                         "'reduced-reject': tracked set only; escaped mutations stay in source bin."))
    p.add_argument("--track-max-hd", type=int, default=None, help="Include all variants with Hamming distance ≤ this from the WT.")
    p.add_argument("--track-top-k", type=int, default=None, help="Include the top-k variants by lowest receptor ddG.")
    p.add_argument("--graph-node-scores", default=None, help="Node score CSV produced by mutational_graph.py.")
    p.add_argument("--graph-track-k", type=int, default=None, help="Number of nodes to keep from the graph ranking.")
    p.add_argument("--graph-prune-mode", choices=["total", "influx", "abundance", "efflux"], default="total", help="Prune nodes based on total, in, or out flux.")
    p.add_argument("--max-freq-file", default=None, metavar="PATH", help="CSV file with variant frequencies (output from max_frequencies.py).")
    p.add_argument("--freq-threshold", type=float, default=0.1, metavar="F", help="Include variants with max_frequency >= threshold.")

    p.add_argument("--count-only", action="store_true", help="Build the tracked variant set and write the actual count to count_<label>.csv in --out-dir.")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.df, index_col=0)
    df = prepare_df(df, args.N)

    params = default_params()
    params['Ab_time_on'] = args.ab_time
    params['Ab_max'] = args.ab_dose

    Ab_func = None
    if args.use_ab:
        Ab_func = biphasic_antibody_decay if args.ab_func == "biphasic" else constant_ab

    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    methods_specified = sum([args.max_freq_file is not None, args.graph_node_scores is not None, args.track_max_hd is not None or args.track_top_k is not None,])
    if args.mode != "full" and methods_specified > 1:
        raise ValueError(
            "Dimensionality reduction methods are mutually exclusive. "
            "Choose one of: --max-freq-file, --graph-node-scores, or (--track-max-hd/--track-top-k)."
        )
    tracked_df = other_df = None

    if args.mode != "full":
        if args.max_freq_file is not None:
            tracked_df, other_df = select_tracked_variants_by_frequency(df, args.N, args.wt_seq, args.max_freq_file, args.freq_threshold)
            print(f"Selected {len(tracked_df)} variants with max_frequency >= {args.freq_threshold}")
        elif args.graph_node_scores is not None:
            if args.graph_track_k is None:
                raise ValueError("--graph-track-k is required when --graph-node-scores is set.")
            graph_df = pd.read_csv(args.graph_node_scores)
            if args.graph_prune_mode == "influx":
                keep_sequences = (graph_df.sort_values("in_flux", ascending=False).head(args.graph_track_k)["sequence"].tolist())
            elif args.graph_prune_mode == "efflux":
                keep_sequences = (graph_df.sort_values("out_flux", ascending=False).head(args.graph_track_k)["sequence"].tolist())
            elif args.graph_prune_mode == "abundance":
                keep_sequences = (graph_df.sort_values("total_abundance", ascending=False).head(args.graph_track_k)["sequence"].tolist())
            else:
                keep_sequences = (graph_df.sort_values("total_flux", ascending=False).head(args.graph_track_k)["sequence"].tolist())
            tracked_df, other_df = select_tracked_variants_by_sequences(df, args.N, args.wt_seq, keep_sequences)
        elif args.track_max_hd is None and args.track_top_k is None:
            raise ValueError("Reduced modes require at least one of: --track-max-hd, --track-top-k, or --graph-node-scores.")

    if args.count_only and args.mode != "full" and tracked_df is None:
        tracked_df, other_df = select_tracked_variants(df, args.wt_seq, track_max_hd=args.track_max_hd, track_top_k=args.track_top_k)

    #Prepare output filenames
    parts = ([f"time{'_'.join(_to_str(t) for t in args.ab_time)}", f"dose{_to_str(args.ab_dose)}"] if args.use_ab else ["noAb"])
    if args.mode != "full":
        tag = args.mode
        if args.max_freq_file is not None:
            freq_str = f"{args.freq_threshold:.6g}".replace(".", "p")
            tag += f"_freq{freq_str}"
        elif args.graph_node_scores is not None:
            tag += f"_graph{args.graph_track_k}"
        else:
            if args.track_max_hd is not None:
                tag += f"_hd{args.track_max_hd}"
            if args.track_top_k is not None:
                tag += f"_top{args.track_top_k}"
        parts.append(tag)
    if args.seed is not None:
        parts.append(f"seed{args.seed}")

    label = "_".join(parts)

    if args.count_only:
        n_tracked = 20**args.N if args.mode == "full" else len(tracked_df)
        count_path = os.path.join(args.out_dir, f"count_{label}.csv")
        pd.DataFrame([{
            "label": label,
            "mode": args.mode,
            "n_tracked": n_tracked,
            "track_max_hd": args.track_max_hd,
            "track_top_k": args.track_top_k,
            "graph_track_k": args.graph_track_k,
            "graph_prune_mode": args.graph_prune_mode,
            "freq_threshold": args.freq_threshold,
            "wt_seq": args.wt_seq,
            "N": args.N,
        }]).to_csv(count_path, index=False)
        return

    totals_base = os.path.join(args.out_dir, f"totals_{label}")
    topk_base = os.path.join(args.out_dir, f"topk_{label}")
    if args.mode == "full":
        simulate(
            df=df, N=args.N, mu=args.mu, params=params,
            t_max=args.t_max, dt=args.dt, binding_affinity_shape=args.binding_affinity_shape,
            use_Ab=args.use_ab, Ab_func=Ab_func,
            WT_seq=args.wt_seq, rng=rng,
            n_measurements=args.n_measurements,
            top_k=args.top_k, include_wt=args.record_wt,
            totals_name=totals_base, topk_name=topk_base,
        )
    else:
        simulate_reduced(
            df=df, N=args.N, mu=args.mu, params=params,
            t_max=args.t_max, dt=args.dt, binding_affinity_shape=args.binding_affinity_shape,
            use_Ab=args.use_ab, Ab_func=Ab_func,
            WT_seq=args.wt_seq, rng=rng,
            n_measurements=args.n_measurements, top_k=args.top_k,
            include_wt=args.record_wt,
            track_max_hd=args.track_max_hd,
            track_top_k=args.track_top_k,
            mode='other' if args.mode == 'reduced-other' else 'reject',
            totals_name=totals_base, topk_name=topk_base,
            tracked_df=tracked_df, other_df=other_df
        )

if __name__ == "__main__":
    main()
