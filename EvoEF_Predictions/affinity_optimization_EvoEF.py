#!/usr/bin/env python3

# ------------------------------------------------------------------------------
# Script Name: affinity_optimization_EvoEF.py
# Description: This script optimizes an input broadly neutralizing HIV-1 antibody
#              by finding a combination of mutations that minimizes the maximal
#              fitness (R0) the virus can obtain. Given an input antibody, it makes
#              all single point mutants at pre-selected residues, then computes antibody-virus
#              binding affinities for a set of inputted viral mutants using EvoEF (Pearce et al. 2019).
#              Together with virus-receptor binding affinities, it computes viral fitness for
#              each antibody and selects the best antibody (with probability 1-epsilon)
#              or a random antibody (with probability epsilon), then generates all single point mutants
#              and iterates until convergence or a maximum number of iterations is reached.
#
# Outputs:
# 1. Per-antibody binding affinities to all input viral mutants (one CSV per antibody in OUTPUT_DIR)
# 2. CSVs containing a summary of each optimization per dose, criterion, and random seed, and an
#    overall summary per dose and criterion (summarizing the outputs of each random seed)
#
# Author: Griffin Kutler Dodd
#
# Date Created: 2025-09-24
# Last Modified: 2025-10-29
#
# Requirements:
#   - This script is designed to be run using Python 3.13.5
#
# Usage:
#   In CONFIG, change EVOEF_PATH to the full path to the EvoEF executable (https://github.com/tommyhuangthu/EvoEF)
#   Change PDB_PATH to the full path to a .pdb of the virus-antibody structure
#   Change VIRUS_VARIANTS_CSV to a CSV containing the receptor-binding affinities of the virus mutants to be analyzed
#   Expected format of VIRUS_VARIANTS_CSV:
#       first N columns are the N residues of the virus that are mutated
#       column ddg_receptor that give the deltadeltaG of receptor binding of each variant relative to a reference
#       column sequence (first N rows concatenated)
#
#   Change residues_virus to the residues on the virus that are mutated (format: tuple (chain, residue number, reference AA))
#   Change residues_Ab to the residues on the antibody that can be optimized (format: tuple (chain, residue number, reference AA))
#   Pick antibody doses (DOSES list, unit M), number of runs per dose (N_RUNS), and what criteri(a)(on) should be optimized (list CRITERIA, options max_r0 or delta_r0)
#
#   Temporary files will be stored in WORKDIR, and all output files will be stored in the folders OUTPUT_DIR and optimization_runs
#
# ------------------------------------------------------------------------------


import os
import shutil
import subprocess
import csv
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
import glob

# -----------------------
# CONFIG
# -----------------------
EVOEF_PATH = "/home/griffin/Major_research_project/EvoEF/EvoEF"
PDB_PATH = "/home/griffin/Major_research_project/Antibody_optimization/7UCF.pdb"
WORKDIR = "evoef_workdir_sharedcache"
CPU_TOTAL = cpu_count()
CONCURRENT_BATCHES = CPU_TOTAL#min(28, CPU_TOTAL)
PER_BATCH_THREADS = max(1, CPU_TOTAL // CONCURRENT_BATCHES)
OUTPUT_DIR = "per_antibody_outputs_sharedcache"
KEEP_BATCH_DIRS = False #set to True to keep per-batch directories
VIRUS_VARIANTS_CSV = "predictions_subset_ddg_receptor_0.csv"
VERBOSE = False

DOSES = [2.8e-7, 2.8e-6]
N_RUNS = 5
CRITERIA = ["max_r0", "delta_r0"]

#Parameters for computing R0
MODEL_PARAMS = {
    "lambda_T": 1e7,
    "d": 0.2,
    "beta_wt": 2e-9,
    "gamma": 1.0,
    "delta1": 0.5,
    "delta2": 1.0,
    "n": 1000.0,
    "c": 20.0,
    "RT": 0.616,
    "KD_WT": 1e-8
}

#Stochasticity: decay epsilon (probability of random step) as epsilon0 * exp(-decay_rate * iteration)
epsilon0 = 0.3
decay_rate = 0.3
MAX_ITER = 100

#Residues to vary in the virus (list of (chain, residue_number, reference_AA) tuples)
residues_virus = [
    ("G", 325, "D"),
    ("G", 326, "I"),
    ("G", 327, "R"),
    ("G", 328, "Q"),
    ("G", 330, "H"),
]

#Map from positions in the virus-receptor structure to the virus-antibody structure
#Used if position names are different in the input CSV and the virus-antibody structure
WT_position_mapping = {
    "G:324": "G:325",
    "G:325": "G:326",
    "G:326": "G:327",
    "G:327": "G:328",
    "G:329": "G:330",
}

#Residues to vary in the Ab
residues_Ab = [
    ("H", 106, "G"),
    ("H", 110, "F"),
    ("H", 112, "E"),
    ("L", 96, "S"),
    ("L", 97, "R"),
]

#If there are more than two chains (e.g. antibody heavy and light chain and virus), specify how the chains should be split
#For example, if H is the heavy chain, L is the light chain, and V the virus, use HL,V
chain_split = "G,HL"

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

# -----------------------
# End CONFIG
# -----------------------

def pos_label(chain, resnum):
    return f"{chain}:{resnum}"

def r0_fun(ddg_receptor, ddg_ab, Ab_dose, p):
    """Compute the R0 of a virus sequence based on its receptor-binding affinity (ddg_receptor), antibody-binding affinity
    (ddg_ab), dose of the antibody (Ab_dose), and other model parameters (p)"""
    return (p['beta_wt']*np.exp(-ddg_receptor/p['RT'])*p['lambda_T']*p['gamma']*p['n'])/(p['c']*p['d']*(p['delta1']+p['gamma'])*(1+((Ab_dose/p['KD_WT'])*np.exp(-ddg_ab/p['RT']))))

def find_ddg_receptor_for_sequence(seq_str, virus_csv_path, residues_order):
    """Search input CSV for the ddg_receptor for a given virus sequence."""
    chunksize = 10000
    for chunk in pd.read_csv(virus_csv_path, usecols=lambda c: c in ('sequence', 'ddg_receptor'), chunksize=chunksize, dtype=str):
        matches = chunk['sequence'].astype(str) == seq_str
        if matches.any():
            val = chunk.loc[matches.idxmax(), 'ddg_receptor']
            return float(val)

def run_cmd(cmd, cwd=None):
    """Run a command, return (returncode, stdout, stderr)."""
    if VERBOSE:
        print(">>> Running: {} (cwd={})".format(" ".join(cmd), cwd))
    proc = subprocess.run(cmd,
                          cwd=cwd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    return proc.returncode, proc.stdout, proc.stderr

def parse_total_from_evoef_stdout(text):
    """Parse the 'Total' line from EvoEF ComputeBinding output."""
    m = re.search(r"Total\s*=\s*([-+]?\d+\.\d+|\d+)", text)
    if m:
        return float(m.group(1))
    return None

def compute_binding_single(evoef_path, model_path):
    """Run EvoEF ComputeBinding on model_path and parse the 'Total' value."""
    cmd = [evoef_path, "--command=ComputeBinding", f"--pdb={os.path.basename(model_path)}", f"--split={chain_split}"]
    ret, out, err = run_cmd(cmd, cwd=os.path.dirname(model_path))
    combined = out + "\n" + err
    val = parse_total_from_evoef_stdout(combined)
    return val, combined

def repair_structure(evoef_path, pdb_path, workdir):
    """Run EvoEF RepairStructure and return path to repaired file named 'structure_repaired.pdb' in workdir."""
    os.makedirs(workdir, exist_ok=True)

    #Extract filename and copy it into the working directory
    pdb_basename = os.path.basename(pdb_path)
    shutil.copy(pdb_path, os.path.join(workdir, pdb_basename))

    #Run EvoEF RepairStructure
    cmd = [evoef_path, "--command=RepairStructure", f"--pdb={pdb_basename}"]
    ret, out, err = run_cmd(cmd, cwd=workdir)
    if ret != 0:
        print("RepairStructure failed. stdout/stderr:")
        print(out)
        print(err)
        raise RuntimeError("RepairStructure failed")

    #Get path to the repaired structure
    root, _ = os.path.splitext(pdb_basename)
    repaired_name = f"{root}_Repair.pdb"
    repaired_path = os.path.join(workdir, repaired_name)
    if not os.path.exists(repaired_path):
        raise FileNotFoundError(f"Expected repaired file {repaired_path} not found")

    #Return the path to the repaired structure
    canonical = os.path.join(workdir, "structure_repaired.pdb")
    shutil.copy(repaired_path, canonical)
    print(f"Repaired structure copied to {canonical}")
    return canonical

def make_mutant_line_from_refs(residues_order_with_ref, mutant_tuple):
    """Build a file of mutants for EvoEF, where each line is a different viral mutant.
    Multiple single mutations are joined by commas and each line ends with a semicolon."""
    parts = []
    for (chain, resnum, ref_aa), mut_aa in zip(residues_order_with_ref, mutant_tuple):
        if mut_aa != ref_aa:
            parts.append(f"{ref_aa}{chain}{resnum}{mut_aa}")
    if not parts:
        return ""
    return ",".join(parts) + ";"

def build_mutants_in_batch(evoef_path, repaired_basename, batch_dir):
    """Run BuildMutant inside batch_dir using a mutant file individual_list.txt."""
    cmd = [evoef_path, "--command=BuildMutant", f"--pdb={repaired_basename}", f"--mutant_file=individual_list.txt"]
    ret, out, err = run_cmd(cmd, cwd=batch_dir)
    if ret != 0:
        print("BuildMutant failed for", batch_dir, "stdout/stderr:", out, err)
        return False
    return True

def safe_remove(path):
    """Safely remove all files in a directory. Used for cleaning EvoEF batch outputs."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except Exception as e:
        print("Warning: could not remove", path, ":", e)

def prepare_virus_entries_from_df(df, residues_order):
    """From a dataframe with column 'sequence' (ordered letters matching residues_order),
    produce list of (mutant_line, seq_tuple) (mutant_line is mutations formatted for EvoEF)."""
    entries = []
    seqs = df['sequence'].astype(str)
    ddg_series = df['ddg_receptor'].astype(float).tolist()

    for seq, ddg_rec in zip(seqs, ddg_series):
        parts = []
        for (chain, resnum, ref_aa), aa in zip(residues_order, seq):
            if aa != ref_aa:
                parts.append(f"{ref_aa}{chain}{resnum}{aa}")
        mutant_line = ",".join(parts) + ";" if parts else ""
        entries.append((mutant_line, tuple(seq), ddg_rec))

    return entries


def generate_single_mutants_for_antibody(residues_Ab):
    """Return list of (mutant_line_string, label, tuple) for antibody single mutants."""
    entries = []
    wt = tuple([ref for (_, _, ref) in residues_Ab])
    for i, (chain, resnum, ref_aa) in enumerate(residues_Ab):
        for mut_aa in AA_ALPHABET:
            if mut_aa == ref_aa:
                continue
            new = list(wt)
            new[i] = mut_aa
            seq_tuple = tuple(new)
            mut_line = seq_tuple_to_mutant_line(seq_tuple)
            label = seq_tuple_to_label(seq_tuple)
            entries.append((mut_line, label, seq_tuple))
    return entries

def combine_mutant_lines(virus_line, antibody_line):
    """Combine virus and antibody mutant strings into one EvoEF mutant line."""
    #Remove trailing semicolons and whitespaces
    v = virus_line.strip().rstrip(';').strip()
    a = antibody_line.strip().rstrip(';').strip()
    parts = []
    if v:
        parts.append(v)
    if a:
        parts.append(a)
    if not parts:
        return ";"
    return ",".join(parts) + ";"


def process_antibody_batch(batch_index, antibody_entry, virus_entries, repaired_pdb, repaired_basename, output_csv_path, csv_lock, per_batch_threads, wt_binding):
    """Helper to compute virus-antibody binding affinities for a batch of antibody sequences."""
    antibody_line, antibody_label, antibody_meta = antibody_entry
    batch_dir = os.path.join(WORKDIR, f"batch_ab_{batch_index:06d}_{antibody_label}")
    os.makedirs(batch_dir, exist_ok=True)
    shutil.copy(repaired_pdb, os.path.join(batch_dir, repaired_basename))

    combined_lines = []
    row_metadata = []
    precomputed_rows = []
    wt_seq = ''.join([r[2] for r in residues_virus])
    for virus_line, virus_seq_tuple, virus_ddg_rec in virus_entries:
        is_virus_wt = (virus_line == "") or (''.join(virus_seq_tuple) == wt_seq)
        is_antibody_wt = (antibody_label == 'WT') or (antibody_line == "")
        if is_antibody_wt and is_virus_wt:
            precomputed_rows.append((virus_seq_tuple, virus_ddg_rec))
        else:
            combined = combine_mutant_lines(virus_line, antibody_line)
            combined_lines.append(combined)
            row_metadata.append((virus_seq_tuple, virus_ddg_rec))

    if precomputed_rows:
        with csv_lock:
            file_exists = os.path.exists(output_csv_path)
            with open(output_csv_path, 'a', newline='') as cf:
                writer = csv.writer(cf)
                if not file_exists:
                    virus_labels = [pos_label(c, r) for c, r, _ in residues_virus]
                    header = virus_labels + ["antibody_mutation", "binding_energy", "ddg_ab"]
                    writer.writerow(header)
                for seq_tuple, ddg_rec in precomputed_rows:
                    binding_energy = wt_binding
                    ddg_ab = 0.0
                    row = list(seq_tuple) + [antibody_label] + [binding_energy] + [ddg_ab]
                    writer.writerow(row)

    individual_list_path = os.path.join(batch_dir, "individual_list.txt")
    with open(individual_list_path, "w") as fh:
        for line in combined_lines:
            fh.write(line + "\n")
    if VERBOSE:
        print(f"[batch {batch_index} - {antibody_label}] Wrote {len(combined_lines)} mutants to {individual_list_path}")

    ok = build_mutants_in_batch(EVOEF_PATH, repaired_basename, batch_dir)
    if not ok:
        print(f"[batch {batch_index} - {antibody_label}] BuildMutant failed -> skipping batch")
        if not KEEP_BATCH_DIRS:
            safe_remove(batch_dir)
        return 0

    model_paths = []
    for i in range(1, len(combined_lines) + 1):
        model_fname = f"{os.path.splitext(repaired_basename)[0]}_Model_{i:04d}.pdb"
        model_path = os.path.join(batch_dir, model_fname)
        if os.path.exists(model_path):
            seq_tuple, ddg_rec = row_metadata[i-1]
            model_paths.append((model_path, seq_tuple, ddg_rec))
        else:
            print(f"[batch {batch_index} - {antibody_label}] WARNING: expected model {model_path} not found (skipping)")

    processed = 0
    if model_paths:
        with ThreadPoolExecutor(max_workers=per_batch_threads) as exe:
            future_to_info = {exe.submit(compute_binding_single, EVOEF_PATH, mp): (mp, st, dr) for mp, st, dr in model_paths}
            for fut in as_completed(future_to_info):
                (model_path, virus_seq_tuple, virus_ddg_rec) = future_to_info[fut]
                try:
                    val, stdout_text = fut.result()
                except Exception as e:
                    print(f"[batch {batch_index} - {antibody_label}] ComputeBinding failed for {model_path}: {e}")
                    val = None
                ddg_ab = float(val) - float(wt_binding)
                row = list(virus_seq_tuple) + [antibody_label] + [val if val is not None else ""] + [ddg_ab]
                with csv_lock:
                    file_exists = os.path.exists(output_csv_path)
                    with open(output_csv_path, "a", newline="") as cf:
                        writer = csv.writer(cf)
                        if not file_exists:
                            virus_labels = [pos_label(c, r) for c, r, _ in residues_virus]
                            header = virus_labels + ["antibody_mutation", "binding_energy", "ddg_ab"]
                            writer.writerow(header)
                        writer.writerow(row)
                try:
                    os.remove(model_path)
                except Exception as e:
                    print(f"[batch {batch_index} - {antibody_label}] Could not remove model {model_path}: {e}")
                processed += 1

    if not KEEP_BATCH_DIRS:
        safe_remove(batch_dir)
    else:
        if VERBOSE:
            print(f"[batch {batch_index} - {antibody_label}] Kept directory for debugging: {batch_dir}")

    if VERBOSE:
        print(f"[batch {batch_index} - {antibody_label}] Done - processed {processed} models")
    return processed

def wt_ab_seq_tuple():
    """Return WT antibody sequence tuple based on residues_Ab."""
    return tuple([ref for (_, _, ref) in residues_Ab])

def seq_tuple_to_label(seq_tuple):
    """Create a compact label for an antibody sequence tuple."""
    return "Ab_" + ''.join(seq_tuple)

def label_to_seq_tuple(label):
    """Convert a sequence-based label back to a seq_tuple."""
    if isinstance(label, str) and label.startswith('Ab_'):
        seq = label[3:]
        return tuple(seq)
    return None

def seq_tuple_to_mutant_line(seq_tuple):
    """Create EvoEF mutant line for an antibody sequence tuple."""
    return make_mutant_line_from_refs(residues_Ab, seq_tuple)

def generate_neighbors(seq_tuple):
    """Generate all single-point neighbor antibody sequence tuples around given seq_tuple."""
    neighbors = []
    for i, aa in enumerate(seq_tuple):
        for mut in AA_ALPHABET:
            if mut == aa:
                continue
            new = list(seq_tuple)
            new[i] = mut
            neighbors.append(tuple(new))
    return neighbors

def _format_dose_for_name(dose):
    #e.g., 2.8e-6 -> 2p8e-6
    s = f"{dose:.1e}"
    s = s.replace(".", "p")
    return s


def regenerate_overview_for_dose(ab_dose):
    """Scan OUTPUT_DIR for per-antibody CSVs and recreate a dose-specific overview.
    For each antibody, compute R0, then find max(R0) and delta(R0): R0 w/o Ab - R0 with Ab for the fittest virus sequence."""
    overview_rows = []
    pattern = os.path.join(OUTPUT_DIR, 'predictions_ab_*.csv')
    files = glob.glob(pattern)
    structure_labels = [pos_label(c, r) for c, r, _ in residues_virus]

    for path in files:
        df = pd.read_csv(path)
        fname = os.path.basename(path)
        label = fname.replace('predictions_ab_', '').replace('.csv', '')

        #Compute R0
        r0_vals = []
        seq_strings = []
        ddg_ab_vals = []
        for _, row in df.iterrows():
            seq = ''.join(str(row[col]) for col in structure_labels)
            ddg_rec = find_ddg_receptor_for_sequence(seq, VIRUS_VARIANTS_CSV, residues_virus)
            ddg_ab = float(row.get('ddg_ab'))
            r0_val = float(r0_fun(ddg_rec, ddg_ab, ab_dose, MODEL_PARAMS))
            r0_vals.append(r0_val)
            ddg_ab_vals.append(ddg_ab)
            seq_strings.append(seq)
        if not r0_vals:
            continue
        max_r0 = np.max(r0_vals)

        #Get sequences with max r0
        best_idx = [i for i, v in enumerate(r0_vals) if v == max_r0]
        best_seqs = ';'.join(seq_strings[i] for i in best_idx)

        #Compute delta R0 of the fittest sequence
        ddg_rec_best = find_ddg_receptor_for_sequence(seq_strings[best_idx[0]], VIRUS_VARIANTS_CSV, residues_virus)
        ddg_ab_best = r0_vals[best_idx[0]]
        r0_noab = r0_fun(ddg_rec_best, ddg_ab_best, 0.0, MODEL_PARAMS)
        delta_r0 = float(r0_noab - maxr0)
        overview_rows.append({'antibody_mutation': label, 'max_r0': float(max_r0), 'delta_r0': delta_r0, 'best_sequences': best_seqs})

    dose_tag = _format_dose_for_name(ab_dose)
    overview_path = os.path.join(OUTPUT_DIR, f'overview_per_antibody_dose_{dose_tag}.csv')
    with open(overview_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['antibody_mutation', 'max_r0', 'delta_r0', 'best_sequences'])
        for row in overview_rows:
            writer.writerow([row['antibody_mutation'], row['max_r0'], row['delta_r0'], row['best_sequences']])
    return overview_path, pd.DataFrame(overview_rows)

def load_overview_cache(overview_path):
    """Load overview CSV into a dict for fast lookup during optimization."""
    odf = pd.read_csv(overview_path)
    cache = {}
    for _, r in odf.iterrows():
        cache[str(r['antibody_mutation'])] = r.to_dict()
    return cache

def evaluate_antibody_sequences(seq_tuples, repaired_pdb, repaired_basename, virus_entries, wt_binding_val):
    """Create batches for a set of antibodies to compute new binding affinity predictions (for those not already in the cache)."""

    #Determine precomputed antibodies
    labels = [seq_tuple_to_label(s) for s in seq_tuples]
    to_run = []
    for seq, label in zip(seq_tuples, labels):
        outpath = os.path.join(OUTPUT_DIR, f"predictions_ab_{label}.csv")
        if not os.path.exists(outpath):
            mut_line = seq_tuple_to_mutant_line(seq)
            to_run.append((mut_line, label, seq))
    csv_lock = threading.Lock()
    if to_run:
        batch_executor = ThreadPoolExecutor(max_workers=CONCURRENT_BATCHES)
        futures = []
        for i, entry in enumerate(to_run):
            _, label, _ = entry
            outpath = os.path.join(OUTPUT_DIR, f"predictions_ab_{label}.csv")
            fut = batch_executor.submit(process_antibody_batch, i, entry, virus_entries, repaired_pdb, repaired_basename, outpath, csv_lock, PER_BATCH_THREADS, wt_binding_val)
            futures.append(fut)
        for fut in futures:
            fut.result()
        batch_executor.shutdown(wait=True)
    return labels

def select_fittest_label_from_overview(labels, overview_cache, criterion, rng):
    """Select the best antibody  from a list of antibodies based on a given criterion (max_r0 or delta_r0)."""
    metric_name = 'delta_r0' if criterion == 'delta_r0' else 'max_r0'
    candidates = []
    for lab in labels:
        entry = overview_cache.get(lab)
        val = float(entry.get(metric_name))
        candidates.append((lab, val))
    if not candidates:
        return rng.choice(labels)
    min_val = min(v for (_, v) in candidates)
    #Randomly break ties
    ties = [lab for (lab, v) in candidates if v == min_val]
    idx = int(rng.integers(low=0, high=len(ties)))
    return ties[idx]

def main():
    #Ensure valid paths to EvoEF and input structure
    if not os.path.isfile(EVOEF_PATH):
        raise FileNotFoundError("EVOEF_PATH does not exist: " + EVOEF_PATH)
    if not os.path.isfile(PDB_PATH):
        raise FileNotFoundError("PDB_PATH does not exist: " + PDB_PATH)

    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Repair the structure and get its basename
    repaired_pdb = repair_structure(EVOEF_PATH, PDB_PATH, WORKDIR)
    repaired_basename = os.path.basename(repaired_pdb)

    #Pre-compute WT binding (wild-type virus & wild-type antibody) using the repaired structure
    print("Computing WT binding affinity for ddG calculations...")
    wt_binding_val, _ = compute_binding_single(EVOEF_PATH, repaired_pdb)
    print(f"WT binding affinity = {wt_binding_val}")

    #Build antibody entries including WT
    antibody_entries = generate_single_mutants_for_antibody(residues_Ab)
    wt_seq = wt_ab_seq_tuple()
    wt_label = seq_tuple_to_label(wt_seq)
    wt_mut_line = seq_tuple_to_mutant_line(wt_seq)
    antibody_entries.insert(0, (wt_mut_line, wt_label, wt_seq))

    #Prepare virus entries
    virus_df = pd.read_csv(VIRUS_VARIANTS_CSV)
    virus_df = virus_df.rename(columns=WT_position_mapping)
    virus_entries = prepare_virus_entries_from_df(virus_df, residues_virus)

    csv_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=CONCURRENT_BATCHES) as batch_executor:
        futures = []
        for batch_index, ab_entry in enumerate(antibody_entries):
            _, ab_label, _ = ab_entry
            safe_label = ab_label.replace("/", "_")
            output_csv_path = os.path.join(OUTPUT_DIR, f"predictions_ab_{safe_label}.csv")
            if os.path.exists(output_csv_path):
                continue
            fut = batch_executor.submit(
                process_antibody_batch,
                batch_index,
                ab_entry,
                virus_entries,
                repaired_pdb,
                repaired_basename,
                output_csv_path,
                csv_lock,
                PER_BATCH_THREADS,
                wt_binding_val,
            )
            futures.append((fut, ab_label))
        for i, (fut, ab_label) in enumerate(futures):
            try:
                processed = fut.result()
                if VERBOSE:
                    print(f"[cache] Batch {i} for antibody {ab_label} processed {processed} models.")
            except Exception as e:
                print(f"[cache] Exception in batch for {ab_label}: {e}")

    #Run across criteria, doses, and N_RUNS with independent RNGs
    base_ss = np.random.SeedSequence()
    for criterion in CRITERIA:
        for dose in DOSES:
            dose_tag = _format_dose_for_name(dose)
            overview_path, _ = regenerate_overview_for_dose(dose)
            base_dir = os.path.join("optimization_runs", criterion, f"dose_{dose_tag}")
            os.makedirs(base_dir, exist_ok=True)
            summary_path = os.path.join(base_dir, "summary_all_seeds.csv")
            if not os.path.exists(summary_path):
                with open(summary_path, 'w', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["seed_index", "final_label", "final_metric", "iterations", "converged"])

            for run_idx in range(N_RUNS):
                child_ss = base_ss.spawn(1)[0]
                rng = np.random.default_rng(child_ss)
                run_dir = os.path.join(base_dir, f"seed_{run_idx}")
                os.makedirs(run_dir, exist_ok=True)
                report_path = os.path.join(run_dir, "optimization_trace.csv")
                if not os.path.exists(report_path):
                    with open(report_path, 'w', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow(['iteration', 'selected_ab_label', 'selected_ab_seq', 'max_r0', 'delta_r0', 'best_viral_sequences', 'selection_method', 'epsilon'])

                current_label = wt_label
                converged = False
                iteration = 1
                overview_cache = load_overview_cache(overview_path)
                entry0 = overview_cache.get(current_label, {})
                cur_max_r0 = entry0.get('max_r0', None)
                cur_delta_r0 = entry0.get('delta_r0', None)
                cur_best_seqs = entry0.get('best_sequences', '')
                with open(report_path, 'a', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow([iteration, current_label, ''.join(label_to_seq_tuple(current_label)), cur_max_r0, cur_delta_r0, cur_best_seqs, 'init', epsilon0])

                #Optimization loop
                while iteration < MAX_ITER:
                    iteration += 1
                    epsilon = float(epsilon0 * np.exp(-decay_rate * (iteration - 1)))

                    cur_seq = label_to_seq_tuple(current_label)
                    neighbors = generate_neighbors(cur_seq)
                    candidates = [cur_seq] + neighbors
                    candidate_labels = [seq_tuple_to_label(s) for s in candidates]

                    #Ensure overview up to date and evaluate any missing candidate antibodies
                    overview_cache = load_overview_cache(overview_path)
                    missing_seqs = [s for s, lab in zip(candidates, candidate_labels) if lab not in overview_cache]
                    if missing_seqs:
                        evaluate_antibody_sequences(missing_seqs, repaired_pdb, repaired_basename, virus_entries, wt_binding_val)
                        overview_path, _ = regenerate_overview_for_dose(dose)
                        overview_cache = load_overview_cache(overview_path)

                    #Choose best according to criterion
                    cand_vals = []
                    metric_key = 'delta_r0' if criterion == 'delta_r0' else 'max_r0'
                    for lab in candidate_labels:
                        ent = overview_cache.get(lab)
                        if ent is None:
                            continue
                        cand_vals.append((lab, float(ent.get(metric_key)), ent))
                    if not cand_vals:
                        break
                    min_val = min(v for (_, v, _) in cand_vals)
                    best_candidates = [lab for (lab, v, _) in cand_vals if v == min_val]

                    #Epsilon-greedy
                    if rng.random() < (1 - epsilon):
                        idx = int(rng.integers(low=0, high=len(best_candidates)))
                        selected_label = best_candidates[idx]
                        method = 'greedy'
                    else:
                        pool = [lab for lab in candidate_labels if lab != current_label]
                        idx = int(rng.integers(low=0, high=len(pool))) if pool else 0
                        selected_label = pool[idx] if pool else current_label
                        method = 'random'

                    selected_entry = overview_cache.get(selected_label, {})
                    sel_max_r0 = selected_entry.get('max_r0', None)
                    sel_delta_r0 = selected_entry.get('delta_r0', None)
                    sel_best_seqs = selected_entry.get('best_sequences', '')

                    with open(report_path, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow([iteration, selected_label, ''.join(label_to_seq_tuple(selected_label)), sel_max_r0, sel_delta_r0, sel_best_seqs, method, epsilon])

                    #Convergence
                    if selected_label == current_label:
                        break
                        converged = True
                    current_label = selected_label

                #Record to summary
                final_entry = load_overview_cache(overview_path).get(current_label, {})
                final_metric = final_entry.get(metric_key, None)
                with open(summary_path, 'a', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow([run_idx, current_label, final_metric, iteration, converged])

    print("All runs completed.")

if __name__ == "__main__":
    main()
