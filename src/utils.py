"""
Shared utilities for simulation and analysis
"""

import numpy as np

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

#Sequence encoding/decoding
def seq_to_lin(seq):
    """Convert an amino acid sequence to its linear index with base-20 encoding."""
    idx = 0
    for ch in seq:
        idx = idx * 20 + AA_TO_IDX[ch]
    return idx

def lin_to_seq(idx, N):
    """Convert a linear index to an amino acid sequence."""
    chars = []
    for _ in range(N):
        idx, r = divmod(idx, 20)
        chars.append(AA_ALPHABET[r])
    return ''.join(reversed(chars))

#Sequence distances
def hamming_distance(s1, s2):
    """Compute the Hamming distance between two sequences."""
    return sum(e1 != e2 for e1, e2 in zip(s1, s2))

def iter_hd1_neighbors(seq):
    """Return all HD-1 neighbors of a sequence."""
    seq_list = list(seq)
    for pos in range(len(seq_list)):
        orig = seq_list[pos]
        for aa in AA_ALPHABET:
            if aa == orig:
                continue
            dst = seq_list.copy()
            dst[pos] = aa
            yield "".join(dst)

#Array utilities
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

#Output naming formatters
def format_dose_label(dose):
    """Format antibody dose as a string suitable for filenames."""
    s = f"{dose:.1e}"
    return s.replace(".", "p")

def format_time_label(t):
    """Format time as a string suitable for filenames."""
    s = f"{t:.6g}"
    return s.replace(".", "p")