#!/usr/bin/env python3

# ------------------------------------------------------------------------------
# Script Name: model_simulation.py
# Description: This script integrates the ODEs of the HIV quasispecies evolution model.
#
# Output: For each simulation run, a summary of each timepoint (file totals_{time_Ab_introduced}_{ab_dose}_{seed}.csv)
#         and the abundances of the top K variants at that timepoint (file topk_{time_Ab_introduced}_{ab_dose}_{seed}.csv) is saved.
#         Totals columns: time, T (number of uninfected target cells), sum_I1 (sum of all infected cells in I1), sum_I2, sum_V (total viral load), Ab (ab dose if included)
#         Topk columns: time, sequence (AA sequence of variant at mutable residues), I1 (number of I1 cells infected by that variant), I2, V, Ab
#
# Author: Griffin Kutler Dodd
#
# Date Created: 2025-10-04
# Last Modified: 2025-10-31
#
# Requirements:
#   - This script is designed to be run using Python 3.13.5
#   - If using BLOSUM sequence distance penalties, Biopython must be installed
#
# Usage:
#   Designed to be run from the commandline using python3 model_simulation.py [args]
#   Required argument: --df (path to dataframe of binding affinities (receptor and optionally antibody) for each viral sequence).
#   Expects input dataframe to have column "ddG" (receptor-binding ddG), and optionally "ddG_Ab"
#   Can specify the antibody decay function (--ab-func constant = no decay, biphasic = biphasic decay)
#   and time (--ab-time t) and dose (--ab-dose d) at which the antibody is introduced.
#   Run python3 model_simulation.py --help for details about all arguments.
#
# ------------------------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

def df_to_ordered_array(df, col='ddG', N=5):
    """Convert a column of an input dataframe into an array ordered by base-20 encoding of the sequences."""

    #Encode first N rows of the dataframe as integers
    seq_ints = np.empty((len(df), N), dtype=np.int16)
    for i in range(N):
        seq_ints[:, i] = df.iloc[:, i].map(AA_TO_IDX).values

    #Map to linear index
    multipliers = (20 ** np.arange(N - 1, -1, -1)).astype(np.int64)
    lin_idx = (seq_ints * multipliers).sum(axis=1)

    ordered_array = np.empty(20**N, dtype=np.float32)
    ordered_array[lin_idx] = df[col].to_numpy(dtype=np.float32)

    return ordered_array

def seq_to_lin(seq):
    """Convert an amino acid sequence to its linear index with base-20 encoding."""
    idx = 0
    for ch in seq:
        idx = idx * 20 + AA_TO_IDX[ch]
    return idx

def r0_fun(ddg_receptor, distance, p, alpha, ddg_ab=None, Ab_dose=0):
    """Compute the R0 of a viral strain based on its receptor-binding affinity, antibody-binding affinity,
    and sequence distance from the WT."""
    if ddg_ab:
        return (p['beta_wt']*np.exp(-(ddg_receptor/p['RT'])-alpha*distance)*p['lambda_T']*p['gamma']*p['n'])/(p['c']*p['d']*(p['delta1']+p['gamma'])*(1+((Ab_dose/p['KD_WT'])*np.exp(-ddg_ab/p['RT']))))
    else:
        return (p['beta_wt']*np.exp(-(ddg_receptor/p['RT'])-alpha*distance)*p['lambda_T']*p['gamma']*p['n'])/(p['c']*p['d']*(p['delta1']+p['gamma']))

def hamming_distance(s1, s2):
    """Compute the Hamming distance between two sequences."""
    return sum(e1 != e2 for e1, e2 in zip(s1, s2))

def lin_to_seq(idx, N):
    """Convert a linear index to an amino acid sequence."""
    chars = []
    for _ in range(N):
        idx, r = divmod(idx, 20)
        chars.append(AA_ALPHABET[r])
    return ''.join(reversed(chars))

def gibbs_multipliers(ddg_array, RT):
    """Compute exp(-ddG / RT) for an array of deltadeltaG values."""
    return np.exp(-ddg_array / RT)

def alpha_multipliers(distance_array, alpha):
    """Compute exp(-alpha * distance) for an array of deltadeltaG values."""
    return np.exp(-distance_array * alpha)

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

    #Stochasticity for fluxes < 1
    flux_tensor = np.zeros_like(mutation_tensor)
    d_mask = mutation_tensor >= 1.0
    flux_tensor[d_mask] = mutation_tensor[d_mask]

    s_mask = ~d_mask
    if np.any(s_mask):
        fluxes = mutation_tensor[s_mask]
        random_vals = rng.random(fluxes.shape)
        flux_tensor[s_mask] = (random_vals < fluxes).astype(mutation_tensor.dtype)

    return flux_tensor

def biphasic_antibody_decay(p, t):
    """Compute the antibody concentration at time t with biphasic decay starting at time Ab_time_on."""
    if t < p['Ab_time_on']:
        return 0
    else:
        return p['Ab_max'] * (p['k']*np.exp(-p['a1']*(t-p['Ab_time_on'])) + (1-p['k'])*np.exp(-p['a2']*(t-p['Ab_time_on'])))

def constant_ab(p, t):
    if t < p['Ab_time_on']:
        return 0
    else:
        return p['Ab_max']


def simulate_euler(df, N, mu, params, t_max=100, dt=0.01, ddg_col='ddG', use_distance_penalty=False, use_Ab=False, Ab_func=None,
                   ddg_col_Ab='ddG_Ab', distance_column='distance', WT_seq='DIRQH', rng=None, n_measurements=100, top_k=250,
                   include_wt=True, totals_name="totals", topk_name="topk"):
    """Euler integrator for the HIV quasispecies model. Considers mutations between all virus sequences with N mutable
    residues (=20^N). Uses ddG for receptor-binding for each virus mutant (and optionally ddG of antibody binding).
    Takes Euler steps of size dt and saves the abundances of the top_k most abundant variants at each measurement point."""
    if rng is None:
        rng = np.random.default_rng()

    #Build receptor ddG array
    ddg_array = df_to_ordered_array(df, col=ddg_col, N=N)

    #If applying a penalty for sequence distance to WT, compute the alpha multipliers for each sequence
    if use_distance_penalty:
        distance_array = df_to_ordered_array(df, col=distance_column, N=N)
        alpha_mult = alpha_multipliers(distance_array, params['alpha'])
    else:
        alpha_mult = np.ones(20**N, dtype=np.float32)

    #Build per-site mutation matrix
    A = np.full((20, 20), mu/19)
    np.fill_diagonal(A, 1-mu)

    tensor_shape = (20,)*N
    n_variants = 20**N

    beta_mult = gibbs_multipliers(ddg_array, params['RT'])

    #If an antibody will be added during the simulation, precompute strain-specific constants
    if use_Ab:
        ddg_Ab_array = df_to_ordered_array(df, col=ddg_col_Ab, N=N)

        if not Ab_func:
            raise Exception("Must include antibody function")

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
    record_steps = np.linspace(0, n_steps, n_measurements, dtype=int)
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
            theta_vector = 1/(1 + Ab * theta_mult)
            infection_vector = params['beta_wt'] * alpha_mult * beta_mult * theta_vector * V
        else:
            infection_vector = params['beta_wt'] * alpha_mult * beta_mult * V

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
    p.add_argument("--ab-time", type=float, default=0.0, help="Time to start antibodies (Ab_time_on).")
    p.add_argument("--ab-dose", type=float, default=1e-8, help="Antibody dose (Ab_max).")
    p.add_argument("--use-distance-penalty", action="store_true", help="Enable a penalty to the infection rate based on distance to the WT sequence (requires specified sequence distance column in the input dataframe).")
    p.add_argument("--distance-metric", choices=["hamming", "blosum"], default="hamming", help="Sequence distance metric.")
    p.add_argument("--wt-seq", default="DIRQH", help="Wild-type sequence (length N).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional).")
    p.add_argument("--record-wt", action="store_true", help="Record the abundance of the wild-type sequence in topk at each measurement.")
    return p.parse_args()

def default_params():
    """Default parameters used in the model."""
    return {
        "lambda_T": 1e7,
        "d": 0.2,
        "beta_wt": 2e-9,
        "gamma": 1.0,
        "delta1": 0.5,
        "delta2": 1.0,
        "n": 1000.0,
        "c": 20.0,
        "RT": 0.616,
        "KD_WT": 1e-8,
        "alpha": 0
    }

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.df, index_col=0)

    params = default_params()
    params['Ab_time_on'] = args.ab_time
    params['Ab_max'] = args.ab_dose

    Ab_func = None
    if args.use_ab:
        if args.ab_func == "biphasic":
            Ab_func = biphasic_antibody_decay
        else:
            Ab_func = constant_ab

    rng = None
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)

    #Prepare output filenames
    if args.use_ab:
        tlabel = _to_str(args.ab_time)
        dlabel = _to_str(args.ab_dose)
        parts = [f"time{tlabel}", f"dose{dlabel}"]
    else:
        parts = ["noAb"]
    if args.seed is not None:
        parts.append(f"seed{args.seed}")
        
    label = "_".join(parts)
    totals_base = os.path.join(args.out_dir, f"totals_{label}")
    topk_base = os.path.join(args.out_dir, f"topk_{label}")

    #If using distance penalty, compute sequence distances
    if args.use_distance_penalty:
        if 'sequence' not in df.columns:
            df['sequence'] = df.iloc[:, :args.N].astype(str).agg(''.join, axis=1)
        if args.distance_metric == "hamming":
            df['distance'] = df['sequence'].apply((lambda x: hamming_distance(x, args.wt_seq)))
        else:
            #Compute BLOSUM distances
            from Bio import Align
            blosum_matrix = Align.substitution_matrices.load("BLOSUM62")
            aligner = Align.PairwiseAligner()
            aligner.substitution_matrix = blosum_matrix
            WT_score = aligner.score(args.wt_seq, args.wt_seq)
            df['distance'] = df['sequence'].apply(lambda x: WT_score - aligner.score(x, args.wt_seq))

        #Compute minimum alpha s.t. WT sequence is the fittest
        from scipy.optimize import bisect
        ddg_receptor_list, dist_list = df['ddG'], df['distance']
        WT_index = int(np.where(df['sequence'] == args.wt_seq)[0][0])
        def diff(alpha):
            r0_vals = r0_fun(ddg_receptor=ddg_receptor_list, distance=dist_list, p=params, alpha=alpha)
            target_r0 = r0_vals[WT_index]
            max_other = np.max(r0_vals[np.arange(len(r0_vals)) != WT_index])
            return target_r0 - max_other
        params['alpha'] = bisect(diff, 0, 10)

    simulate_euler(
        df=df,
        N=args.N,
        mu=args.mu,
        params=params,
        t_max=args.t_max,
        dt=args.dt,
        use_distance_penalty=args.use_distance_penalty,
        use_Ab=args.use_ab,
        Ab_func=Ab_func,
        WT_seq=args.wt_seq,
        rng=rng,
        n_measurements=args.n_measurements,
        top_k=args.top_k,
        include_wt=args.record_wt,
        totals_name=totals_base,
        topk_name=topk_base
    )

if __name__ == "__main__":
    main()
