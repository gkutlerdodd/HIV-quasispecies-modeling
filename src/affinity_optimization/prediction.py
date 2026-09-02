#!/usr/bin/env python3

"""
Predict the binding affinities of all 20^N viral mutants (N = number of mutant residues)
to a given host protein (e.g., host receptor, antibody.) using EvoEF (Pearce et al. 2019).

Usage:
    python3 -m src.affinity_optimization.prediction

Before running, update CONFIG with:
    EVOEF_PATH: path to EvoEF executable
    PDB_PATH: path to input PDB structure
    residues_virus: list of (chain, residue_number, reference_AA) viral residues to mutate
    chain_split: how to partition proteins into chains if there are more than two chains

    And optionally:
    OUTPUT_CSV: path to the output CSV file
    mutant_residues_prot: list of (chain, residue_number, reference_AA, mutant_AA) to add to the protein

"""
import os
import shutil
import subprocess
import itertools
import csv
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from src.utils import AA_ALPHABET

# -----------------------
# CONFIG
# -----------------------

EVOEF_PATH = "path/to/EvoEF/executable"
PDB_PATH = "path/to/structure.pdb"
WORKDIR = "evoef_workdir"
OUTPUT_CSV = "predictions.csv"

CPU_TOTAL = cpu_count()
CONCURRENT_BATCHES = CPU_TOTAL/2#min(28, CPU_TOTAL)
PER_BATCH_THREADS = max(1, CPU_TOTAL // CONCURRENT_BATCHES)
BATCH_SIZE = 100
MAX_WORKERS = cpu_count()
KEEP_BATCH_DIRS = False #set to True to keep per-batch directories
VERBOSE = False

#Viral residues to vary
#Format: list of (chain, residue_number, reference_AA) tuples
residues_virus = [
    ("G", 325, "D"),
    ("G", 326, "I"),
    ("G", 327, "R"),
    ("G", 328, "Q"),
    ("G", 330, "H"),
]

#If analyzing binding to a mutated protein, specifiy the residues and what they should be mutated to
#Format: list of (chain, residue_number, reference_AA, mutant_AA) tuples
mutant_residues_prot = []

#If there are more than two chains (e.g. antibody heavy and light chain and virus), specify how the chains should be split
#For example, if H is the heavy chain, L is the light chain, and V the virus, use HL,V
chain_split = "HL,G"

def pos_label(chain, resnum):
    return f"{chain}:{resnum}"

def run_cmd(cmd, cwd=None):
    """Run a command, return (returncode, stdout, stderr)."""
    if VERBOSE:
        print(">>> Running: {} (cwd={})".format(" ".join(cmd), cwd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

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
    if VERBOSE:
        print(f"Repaired structure copied to {canonical}")
    return canonical

def make_mutant_line_from_refs(residues_order_with_ref, mutant_tuple, prot_string=None):
    """Build a file of mutants for EvoEF, where each line is a different viral mutant.
    Multiple single mutations are joined by commas and each line ends with a semicolon.
    prot_string argument is used for adding static mutations to each line"""
    parts = []
    for (chain, resnum, ref_aa), mut_aa in zip(residues_order_with_ref, mutant_tuple):
        #EvoEF format: <refAA><chain><resnum><mutAA>
        parts.append(f"{ref_aa}{chain}{resnum}{mut_aa}")
    if prot_string:
        parts.append(prot_string)
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
    """Remove all files in a directory."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except Exception as e:
        print("Warning: could not remove", path, ":", e)

def process_batch(batch_index, mutant_lines, mutant_tuples, repaired_pdb, repaired_basename, csv_path, csv_lock, per_batch_threads):
    """Process one batch: create a batch directory, copy the repaired structure, write a mutant file,
    run BuildMutant and ComputeBinding, append to a CSV, and clean up the directory."""
    batch_dir = os.path.join(WORKDIR, f"batch_{batch_index:06d}")
    os.makedirs(batch_dir, exist_ok=True)
    shutil.copy(repaired_pdb, os.path.join(batch_dir, repaired_basename))

    #Write a mutant file
    individual_list_path = os.path.join(batch_dir, "individual_list.txt")
    with open(individual_list_path, "w") as fh:
        for line in mutant_lines:
            fh.write(line + "\n")
    if VERBOSE:
        print(f"[batch {batch_index}] Wrote {len(mutant_lines)} mutants to {individual_list_path}")

    ok = build_mutants_in_batch(EVOEF_PATH, repaired_basename, batch_dir)
    if not ok:
        print(f"[batch {batch_index}] BuildMutant failed -> skipping batch")
        if not KEEP_BATCH_DIRS:
            safe_remove(batch_dir)
        return 0

    #Collect model file paths
    model_paths = []
    for i, mutant_tuple in enumerate(mutant_tuples, start=1):
        model_fname = f"{os.path.splitext(repaired_basename)[0]}_Model_{i:04d}.pdb"
        model_path = os.path.join(batch_dir, model_fname)
        if os.path.exists(model_path):
            model_paths.append((model_path, mutant_tuple))
        else:
            print(f"[batch {batch_index}] WARNING: expected model {model_path} not found (skipping)")

    #Run ComputeBinding in parallel
    processed = 0
    if model_paths:
        with ThreadPoolExecutor(max_workers=per_batch_threads) as exe:
            future_to_info = {exe.submit(compute_binding_single, EVOEF_PATH, mp): (mp, mt) for mp, mt in model_paths for mt in [mt]}
            for fut in as_completed(future_to_info):
                (model_path, mutant_tuple) = future_to_info[fut]
                try:
                    val, _ = fut.result()
                except Exception as e:
                    print(f"[batch {batch_index}] ComputeBinding failed for {model_path}: {e}")
                    val = None

                #Append CSV row
                row = list(mutant_tuple) + [val if val is not None else ""]
                with csv_lock:
                    with open(csv_path, "a", newline="") as cf:
                        writer = csv.writer(cf)
                        writer.writerow(row)

                #Clean up model
                try:
                    os.remove(model_path)
                except Exception as e:
                    print(f"[batch {batch_index}] Could not remove model {model_path}: {e}")
                processed += 1

    #Clean batch directory
    if not KEEP_BATCH_DIRS:
        safe_remove(batch_dir)

    return processed

def main():
    #Ensure valid paths to EvoEF and input structure
    if not os.path.isfile(EVOEF_PATH):
        raise FileNotFoundError("EVOEF_PATH does not exist: " + EVOEF_PATH)
    if not os.path.isfile(PDB_PATH):
        raise FileNotFoundError("PDB_PATH does not exist: " + PDB_PATH)

    os.makedirs(WORKDIR, exist_ok=True)

    #Repair the structure and get its basename
    repaired_pdb = repair_structure(EVOEF_PATH, PDB_PATH, WORKDIR)
    repaired_basename = os.path.basename(repaired_pdb)

    #Format viral residues to mutate
    residues_order = []
    for item in residues_virus:
        chain, resnum, ref_aa = item
        ref_aa = ref_aa.upper()
        if ref_aa not in AA_ALPHABET:
            raise ValueError(f"Reference AA '{ref_aa}' not in alphabet: {AA_ALPHABET}")
        residues_order.append((chain, int(resnum), ref_aa))

    #Prepare CSV header and start CSV
    position_labels = [pos_label(chain, resnum) for (chain, resnum, _) in residues_order]
    header = position_labels + ["binding_energy"]
    with open(OUTPUT_CSV, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
    print(f"CSV header written to {OUTPUT_CSV}")

    #Prepare protein string with mutations
    if mutant_residues_prot:
        prot_parts = []
        for (chain, resnum, ref_aa, mut_aa) in mutant_residues_prot:
            prot_parts.append(f"{ref_aa}{chain}{resnum}{mut_aa}")
        prot_string = ",".join(prot_parts)
    else:
        prot_string = None

    #Generate batches
    batch_index = 0
    batch_mutant_lines = []
    batch_mutant_tuples = []
    csv_lock = threading.Lock()
    batch_executor = ThreadPoolExecutor(max_workers=CONCURRENT_BATCHES)
    batch_futures = []

    #Helper to create mutant line
    def make_line_and_store(combo, prot_string):
        line = make_mutant_line_from_refs(residues_order, combo, prot_string)
        return line


    print(f"Starting : CPU_TOTAL={CPU_TOTAL}  CONCURRENT_BATCHES={CONCURRENT_BATCHES}  PER_BATCH_THREADS={PER_BATCH_THREADS}")
    processed_total = 0

    #Create mutant batches
    for combo in itertools.product(AA_ALPHABET, repeat=len(residues_order)):
        mutant_line = make_line_and_store(combo, prot_string)
        if mutant_line is None:
            continue
        batch_mutant_lines.append(mutant_line)
        batch_mutant_tuples.append(combo)
        if len(batch_mutant_lines) >= BATCH_SIZE:
            #Submit batch to executor
            if VERBOSE:
                print(f"Submitting batch {batch_index} (size {len(batch_mutant_lines)}) to executor")
            fut = batch_executor.submit(process_batch,
                                        batch_index,
                                        list(batch_mutant_lines),
                                        list(batch_mutant_tuples),
                                        repaired_pdb,
                                        repaired_basename,
                                        OUTPUT_CSV,
                                        csv_lock,
                                        PER_BATCH_THREADS)
            batch_futures.append(fut)
            batch_index += 1
            batch_mutant_lines = []
            batch_mutant_tuples = []

    #Submit final partial batch
    if batch_mutant_lines:
        if VERBOSE:
            print(f"Submitting batch {batch_index} (size {len(batch_mutant_lines)}) to executor")
        fut = batch_executor.submit(process_batch,
                                    batch_index,
                                    list(batch_mutant_lines),
                                    list(batch_mutant_tuples),
                                    repaired_pdb,
                                    repaired_basename,
                                    OUTPUT_CSV,
                                    csv_lock,
                                    PER_BATCH_THREADS)
        batch_futures.append(fut)

    #Wait for batches to finish and combine results
    for i, fut in enumerate(batch_futures):
        try:
            processed = fut.result()
            processed_total += processed
            if VERBOSE:
                print(f"[main] Batch future {i} returned processed={processed}")
        except Exception as e:
            print(f"[main] Batch future {i} raised exception: {e}")

    batch_executor.shutdown(wait=True)
    print("All batches complete. Total processed models:", processed_total)
    print("Output:", os.path.abspath(OUTPUT_CSV))

if __name__ == "__main__":
    main()
