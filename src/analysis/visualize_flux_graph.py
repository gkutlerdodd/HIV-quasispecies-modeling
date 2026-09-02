#!/usr/bin/env python3
"""
Render flux graphs from simulation results as network movies/PNG stills.
Modes (--mode):
    render (default)    Load windowed CSVs and produce movies / PNG stills.
    top-nodes           Compute and save the top-N node list from total node CSVs

Output (render mode):
    flux_movie_time{T}.mp4   one movie per Ab administration time
    fitness_static.png       static R0 node plot (if --fitness-file provided)
    frame_t{NNNN}.png        PNG stills at requested window_starts
    flux_norms.json          LogNorm bounds for edge flux and node abundance

Example usage — render:
    python3 visualize_flux_graph.py \
        --edge-files windowed_flux_time365_dose2p8e-06_seed1_compressed.csv \
        --node-files windowed_nodes_time365_dose2p8e-06_seed1_compressed.csv \
        --load-top-nodes top40_nodes.txt \
        --annotate DIRQH:red AIRQH:blue \
        --save-frames 0 30 \
        --top-n 40 --fps 10

Example usage — top-nodes:
    python3 visualize_flux_graph.py --mode top-nodes \
        --total-node-files nodes_time365_seed1.csv nodes_time365_seed2.csv \
        --top-n 40 --save-top-nodes top40_nodes.txt

Run python3 -m src.analysis.visualize_flux_graph -h for all options.
"""

import os
import re
import sys
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
import networkx as nx

from src.utils import hamming_distance
from src.params import default_params

###################
##Movie config options
###################
NODE_SIZE = 350
NODE_SHRINK = 10.0
NODE_EDGE_COLOR = "#1A1A1A"
NODE_EDGE_LW = 0.5
ARROW_RAD = 0.20 #Arc curvature
ARROW_SCALE = 8.0 #Arrowhead size
LW_MIN, LW_MAX = 0.4, 2.5
ALPHA_BASE = 0.08 #Alpha of weakest edges
ALPHA_SCALE = 0.85 #Alpha ramp range
ANNOT_LW = 1.5
#Spine color: red after Ab, gray before antibody
AB_SPINE_COLOR  = "#B71C1C"
AB_SPINE_LW = 2.5
PRE_AB_SPINE_COLOR = "#9E9E9E"
PRE_AB_SPINE_LW = 1.0
CB_HEIGHT = 0.035
CB_BOTTOM = 0.085
CB_LEFT_LEFT = 0.10
CB_LEFT_WIDTH = 0.34
CB_RIGHT_LEFT = 0.56
CB_RIGHT_WIDTH = 0.34
CB_PLOT_BOTTOM = 0.10

###################
##Data loading
###################
def parse_ab_time(filename):
    """Extract integer ab_time from a filename containing 'timeN'."""
    m = re.search(r'time(\d+)', os.path.basename(filename))
    if m is None:
        raise ValueError(f"Cannot find 'timeN' in filename: {filename}")
    return int(m.group(1))

def parse_annotations(annot_list):
    """Parse ['SEQ:color', ...] into {seq: color}."""
    result = {}
    for entry in (annot_list or []):
        parts = entry.split(":", 1)
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError(f"--annotate entry '{entry}' must be SEQ:color.")
        result[parts[0].strip()] = parts[1].strip()
    return result

def _check_columns(path, df, required):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"'{path}' is missing columns: {missing}")

def _iter_windows_from_csv(path, top_set, kind, chunksize=100000):
    """Load (window_key, data_dict) one window at a time."""
    if kind == "edge":
        required = {"window_start", "window_end", "seq_i", "seq_j", "flux"}
    else:
        required = {"window_start", "window_end", "sequence", "total_abundance"}

    current_key = None
    current_data = {}
    first = True

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if first:
            _check_columns(path, chunk, required)
            first = False

        if kind == "edge":
            mask = chunk["seq_i"].isin(top_set) & chunk["seq_j"].isin(top_set)
            chunk = chunk.loc[mask]
        else:
            chunk = chunk.loc[chunk["sequence"].isin(top_set)]

        for row in chunk.itertuples(index=False):
            key = (row.window_start, row.window_end)
            if key != current_key:
                if current_key is not None:
                    yield current_key, current_data
                current_key = key
                current_data = {}
            if kind == "edge":
                current_data[(row.seq_i, row.seq_j)] = row.flux
            else:
                current_data[row.sequence] = row.total_abundance

    if current_key is not None:
        yield current_key, current_data

def _load_single_window(path, top_set, kind, target_ws, chunksize=100000):
    """Load only the rows belonging to window_start==target_ws."""
    if kind == "edge":
        required = {"window_start", "window_end", "seq_i", "seq_j", "flux"}
    else:
        required = {"window_start", "window_end", "sequence", "total_abundance"}

    data = {}
    key = None
    first = True

    for chunk in pd.read_csv(path, chunksize=chunksize):
        if first:
            _check_columns(path, chunk, required)
            first = False

        chunk = chunk.loc[chunk["window_start"] == target_ws]

        if kind == "edge":
            chunk = chunk.loc[chunk["seq_i"].isin(top_set) & chunk["seq_j"].isin(top_set)]
            for row in chunk.itertuples(index=False):
                key = (row.window_start, row.window_end)
                data[(row.seq_i, row.seq_j)] = row.flux
        else:
            chunk = chunk.loc[chunk["sequence"].isin(top_set)]
            for row in chunk.itertuples(index=False):
                key = (row.window_start, row.window_end)
                data[row.sequence] = row.total_abundance

    if key is None:
        raise ValueError(f"window_start={target_ws} not found in '{path}'.")
    return {key: data}

def get_all_windows(edge_files, chunksize=100_000):
    """Return sorted list of (window_start, window_end) tuples across all files."""
    windows = set()
    for path in edge_files:
        for chunk in pd.read_csv(path, usecols=["window_start", "window_end"], chunksize=chunksize):
            for row in chunk.drop_duplicates().itertuples(index=False):
                windows.add((row.window_start, row.window_end))
    return sorted(windows)

###################
##Normalization
###################
_RESERVOIR_SIZE = 50000 #Take a random sample to compute normalization range

def _scan_norm_stats(paths, top_set, kind, chunksize=100000):
    rng = np.random.default_rng(0)

    if kind == "edge":
        filter_fn = lambda c: c.loc[
            c["seq_i"].isin(top_set) & c["seq_j"].isin(top_set), "flux"]
        required = {"window_start", "window_end", "seq_i", "seq_j", "flux"}
    else:
        filter_fn = lambda c: c.loc[c["sequence"].isin(top_set), "total_abundance"]
        required = {"window_start", "window_end", "sequence", "total_abundance"}

    reservoir = np.empty(_RESERVOIR_SIZE, dtype=np.float64)
    res_count = 0 #Number of values seen so far
    res_filled = 0 #How many reservoir slots are filled
    running_max = -np.inf

    for path in paths:
        first = True
        for chunk in pd.read_csv(path, chunksize=chunksize):
            if first:
                _check_columns(path, chunk, required)
                first = False
            vals = filter_fn(chunk)
            vals = vals[vals > 0].to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue

            #Update running max
            chunk_max = float(vals.max())
            if chunk_max > running_max:
                running_max = chunk_max

            #Reservoir sampling
            for v in vals:
                res_count += 1
                if res_filled < _RESERVOIR_SIZE:
                    reservoir[res_filled] = v
                    res_filled += 1
                else:
                    j = int(rng.integers(0, res_count))
                    if j < _RESERVOIR_SIZE:
                        reservoir[j] = v

    return running_max, reservoir[:res_filled]

def compute_log_norms_streaming(edge_paths, node_paths, top_set, flux_pct=5, abundance_pct=5, chunksize=100000):
    """Compute global LogNorm bounds."""
    def _make_norm(running_max, reservoir, pct):
        if reservoir.size == 0 or running_max <= 0:
            return mcolors.LogNorm(vmin=1e-6, vmax=1.0)
        vmax = float(running_max)
        vmin = max(float(np.percentile(reservoir, pct)), vmax * 1e-10)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)

    flux_max, flux_res = _scan_norm_stats(edge_paths, top_set, "edge", chunksize)
    abund_max, abund_res = _scan_norm_stats(node_paths, top_set, "node", chunksize)

    return (_make_norm(flux_max, flux_res, flux_pct), _make_norm(abund_max, abund_res, abundance_pct))

def save_norm_file(path, flux_norm, node_norm):
    with open(path, "w") as f:
        json.dump({"flux_vmin":  float(flux_norm.vmin), "flux_vmax":  float(flux_norm.vmax), "abund_vmin": float(node_norm.vmin), "abund_vmax": float(node_norm.vmax)}, f, indent=2)
    print(f"Norm file → {path}")

def load_norm_file(path):
    with open(path) as f:
        d = json.load(f)
    return (mcolors.LogNorm(vmin=d["flux_vmin"],  vmax=d["flux_vmax"]), mcolors.LogNorm(vmin=d["abund_vmin"], vmax=d["abund_vmax"]))

###################
##Top node selection
###################
def select_top_nodes_from_totals(total_node_files, top_n):
    acc = {}
    for path in total_node_files:
        df = pd.read_csv(path)
        if "sequence" not in df.columns or "total_flux" not in df.columns:
            raise ValueError(f"'{path}' must have 'sequence' and 'total_flux' columns.")
        for row in df.itertuples(index=False):
            acc[row.sequence] = acc.get(row.sequence, 0.0) + row.total_flux
    return sorted(acc, key=acc.__getitem__, reverse=True)[:top_n]

def save_top_nodes(path, top_nodes):
    with open(path, "w") as f:
        for seq in top_nodes:
            f.write(seq + "\n")
    print(f"Saved {len(top_nodes)} nodes → {path}")

def load_top_nodes(path):
    with open(path) as f:
        nodes = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(nodes)} nodes from {path}")
    return nodes

###################
##Graph layout and drawing
###################
def find_hd1_pairs(top_nodes):
    """Return all directed (u, v) pairs in top_nodes with Hamming distance 1."""
    pairs = []
    for u, v in combinations(top_nodes, 2):
        if hamming_distance(u, v) == 1:
            pairs.extend([(u, v), (v, u)])
    return pairs


def compute_layout(top_nodes, hd1_pairs):
    """Compute fixed node positions."""
    G = nx.Graph()
    G.add_nodes_from(top_nodes)
    for u, v in hd1_pairs:
        if not G.has_edge(u, v):
            G.add_edge(u, v)
    return nx.kamada_kawai_layout(G)

def _log_norm_value(value, norm):
    """Map to [0, 1] on the log scale defined by norm."""
    if value <= 0:
        return 0.0
    clipped = float(np.clip(value, norm.vmin, norm.vmax))
    return float(np.clip((np.log10(clipped) - np.log10(norm.vmin)) / (np.log10(norm.vmax) - np.log10(norm.vmin)),0.0, 1.0))

def _arrow_offset(u_pos, v_pos, sign):
    """Offset arrow endpoints perpendicularly so u→v and v→u don't overlap."""
    dx, dy = v_pos[0] - u_pos[0], v_pos[1] - u_pos[1]
    length = max(np.hypot(dx, dy), 1e-9)
    px, py = -dy / length, dx / length
    ox, oy = ARROW_OFFSET * sign * px, ARROW_OFFSET * sign * py
    return ((u_pos[0] + ox, u_pos[1] + oy), (v_pos[0] + ox, v_pos[1] + oy))



def _draw_nodes(ax, seqs, pos, node_colors_map, node_size, node_edge_color,
                node_edge_lw, front_edge_colors_map=None, annot_lw=None, zorder_bg=3, zorder_fg=6):
    """Draw nodes on the graph."""
    _alw = annot_lw if annot_lw is not None else ANNOT_LW
    front = set(front_edge_colors_map or {})

    #Background nodes (everything not in front set)
    bg_seqs = [s for s in seqs if s not in front]
    if bg_seqs:
        ax.scatter([pos[s][0] for s in bg_seqs], [pos[s][1] for s in bg_seqs], s=node_size, c=[node_colors_map[s] for s in bg_seqs], edgecolors=node_edge_color, linewidths=node_edge_lw, zorder=zorder_bg)

    #Front nodes drawn on top (highlighted sequences)
    if front_edge_colors_map:
        fg_seqs = [s for s in seqs if s in front and s in pos]
        if fg_seqs:
            ax.scatter([pos[s][0] for s in fg_seqs], [pos[s][1] for s in fg_seqs], s=node_size, c=[node_colors_map[s] for s in fg_seqs], edgecolors=[front_edge_colors_map[s] for s in fg_seqs], linewidths=_alw, zorder=zorder_fg)

def add_colorbars(fig, edge_cmap, flux_norm, node_norm):
    """Add colorbars to the bottom of the figure."""
    fig.subplots_adjust(bottom=CB_PLOT_BOTTOM)

    for rect, cmap, norm, label in [
        ([CB_LEFT_LEFT,  CB_BOTTOM, CB_LEFT_WIDTH,  CB_HEIGHT],
         edge_cmap, flux_norm, "Mutation flux (cells day$^{-1}$)"),
        ([CB_RIGHT_LEFT, CB_BOTTOM, CB_RIGHT_WIDTH, CB_HEIGHT],
         plt.cm.gray_r, node_norm, "Viral abundance (virions)"),
    ]:
        ax_cb = fig.add_axes(rect)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
        cb.set_label(label, labelpad=3, fontsize=12)
        cb.ax.tick_params(length=2.5, width=0.6, labelsize=11)
        cb.outline.set_linewidth(0.5)

def draw_flux_panel(ax, window_start, edge_lookup, node_lookup, pos, hd1_pairs, top_nodes,
        flux_norm, node_norm, ab_time=None, annotations=None, edge_style="both",
        min_flux_thresh=0.0, edge_cmap=None, show_title=True, title_fontsize=15, node_size=None,
        node_edge_color=None, node_edge_lw=None, annot_lw=None, show_edges=True, front_nodes=None):
    """Draw a single flux-graph panel onto *ax* for the given window_start."""
    _node_size = node_size if node_size is not None else NODE_SIZE
    _nec = node_edge_color if node_edge_color is not None else NODE_EDGE_COLOR
    _nelw = node_edge_lw if node_edge_lw is not None else NODE_EDGE_LW
    _alw = annot_lw if annot_lw is not None else ANNOT_LW
    if edge_cmap is None:
        edge_cmap = matplotlib.colormaps["plasma"]

    matching = [(ws, we) for (ws, we) in edge_lookup if ws == window_start]
    if not matching:
        matching = [(ws, we) for (ws, we) in node_lookup if ws == window_start]
    if not matching:
        raise ValueError(f"window_start={window_start} not found in edge_lookup or node_lookup.")
    ws, we = matching[0]
    window_key = (ws, we)

    ax.clear()
    ax.axis("off")

    #Title and spine
    if ws >= ab_time:
        spine_color, spine_lw = AB_SPINE_COLOR, AB_SPINE_LW
        title_color = AB_SPINE_COLOR
        title_suffix = f" — Ab administered (t = {ab_time})"
    else:
        spine_color, spine_lw = PRE_AB_SPINE_COLOR, PRE_AB_SPINE_LW
        title_color = "black"
        title_suffix = f" — before Ab (t = {ab_time})"

    if show_title:
        ax.set_title(f"t = {ws:.0f}{title_suffix}", fontsize=title_fontsize, pad=3, color=title_color, loc="center")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(spine_lw)

    if show_edges:
        node_radius_pts = np.sqrt(_node_size / np.pi)
        window_edges = edge_lookup.get(window_key, {})
        from matplotlib.patches import FancyArrowPatch
        for (u, v) in hd1_pairs:
            flux = window_edges.get((u, v), 0.0)
            if flux <= 0 or (min_flux_thresh > 0 and flux < min_flux_thresh):
                continue
            nf = _log_norm_value(flux, flux_norm)
            alpha = ALPHA_BASE + ALPHA_SCALE * nf
            color = edge_cmap(nf) if edge_style in ("color", "both") else edge_cmap(1.0)
            lw = (LW_MIN + (LW_MAX - LW_MIN) * nf if edge_style in ("thickness", "both") else (LW_MIN + LW_MAX) / 2.0)
            rad = ARROW_RAD if u < v else -ARROW_RAD
            patch = FancyArrowPatch(posA=pos[u], posB=pos[v], arrowstyle=f"-|>,head_width={ARROW_SCALE * 0.065},head_length={ARROW_SCALE * 0.08}",
                connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, alpha=alpha, shrinkA=node_radius_pts + 1.0, shrinkB=node_radius_pts + 1.0, zorder=2)
            ax.add_patch(patch)

    #Draw nodes
    window_nodes = node_lookup.get(window_key, {})
    node_colors_map = {s: plt.cm.gray_r(_log_norm_value(window_nodes.get(s, 0.0), node_norm)) for s in top_nodes}
    if front_nodes is None and annotations:
        _front_set = set(annotations.keys())
    else:
        _front_set = set(front_nodes or [])
    front_edge_map = {}
    for s in _front_set:
        if annotations and s in annotations:
            front_edge_map[s] = annotations[s]
        elif s in pos:
            front_edge_map[s] = _nec

    _draw_nodes(ax, top_nodes, pos, node_colors_map, node_size=_node_size, node_edge_color=_nec, node_edge_lw=_nelw,
        front_edge_colors_map=front_edge_map if front_edge_map else None, annot_lw=_alw, zorder_bg=3, zorder_fg=6)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    data_w = max(all_x) - min(all_x) or 1.0
    data_h = max(all_y) - min(all_y) or 1.0

    try:
        fig = ax.get_figure()
        bbox = ax.get_window_extent(fig.canvas.get_renderer())
        aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0
    except Exception:
        aspect = 1.0

    PAD = 0.10
    cx = (min(all_x) + max(all_x)) / 2
    cy = (min(all_y) + max(all_y)) / 2
    if aspect >= 1.0:
        target_w = data_h * aspect
        ax.set_xlim(cx - target_w / 2 * (1 + PAD), cx + target_w / 2 * (1 + PAD))
        ax.set_ylim(cy - data_h / 2 * (1 + PAD), cy + data_h / 2 * (1 + PAD))
    else:
        target_h = data_w / aspect
        ax.set_xlim(cx - data_w / 2 * (1 + PAD), cx + data_w / 2 * (1 + PAD))
        ax.set_ylim(cy - target_h / 2 * (1 + PAD), cy + target_h / 2 * (1 + PAD))

###################
##Fitness plot
###################
def _compute_r0(ddg, p):
    mult = (1.0+p['h'])**p['K']
    ddg_receptor_transform = mult*(1.0/(1.0 + p['h']*np.exp(ddg / p['RT']))) ** p['K']
    return (p['beta_wt']*ddg_receptor_transform*p['lambda_T']*p['gamma']*p['n'])/(p['c']*p['d']*(p['delta1']+p['gamma']))

def make_fitness_plot(fitness_file, top_nodes, pos, hd1_pairs, output_path, params, annotations=None, front_nodes=None):
    """Nodes coloured by R0 w/o antibody."""
    df = pd.read_csv(fitness_file)
    _check_columns(fitness_file, df, {"sequence", "ddG"})
    ddg_map = dict(zip(df["sequence"], df["ddG"]))

    r0_vals = []
    for s in top_nodes:
        ddg = ddg_map.get(s, np.nan)
        r0_vals.append(np.nan if np.isnan(ddg) else _compute_r0(ddg, params))

    idx_missing = [i for i, v in enumerate(r0_vals) if np.isnan(v)]
    idx_below = [i for i, v in enumerate(r0_vals) if not np.isnan(v) and v < 1.0]
    idx_above = [i for i, v in enumerate(r0_vals) if not np.isnan(v) and v >= 1.0]
    r0_above = [r0_vals[i] for i in idx_above]
    fitness_norm = mcolors.Normalize(vmin=1.0, vmax=max(r0_above) if r0_above else 2.0)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    drawn = set()
    for (u, v) in hd1_pairs:
        key = tuple(sorted([u, v]))
        if key not in drawn:
            drawn.add(key)
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#bbbbbb", lw=0.6, zorder=1, alpha=0.5)

    node_colors_map = {}
    for i, s in enumerate(top_nodes):
        if i in idx_missing:
            node_colors_map[s] = "#dddddd"
        elif i in idx_below:
            node_colors_map[s] = "white"
        else:
            node_colors_map[s] = matplotlib.colormaps["viridis"](fitness_norm(r0_vals[i]))

    sc = None
    if idx_above:
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=fitness_norm)
        sm.set_array(r0_above)
        sc = sm

    if front_nodes is None and annotations:
        _front_set = set(annotations.keys())
    else:
        _front_set = set(front_nodes or [])
    front_edge_map = {}
    for s in _front_set:
        if annotations and s in annotations:
            front_edge_map[s] = annotations[s]
        elif s in pos:
            front_edge_map[s] = NODE_EDGE_COLOR

    _draw_nodes(ax, top_nodes, pos, node_colors_map, node_size=NODE_SIZE, node_edge_color=NODE_EDGE_COLOR, node_edge_lw=NODE_EDGE_LW, front_edge_colors_map=front_edge_map if front_edge_map else None, zorder_bg=3, zorder_fg=6)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    ax.set_xlim(min(all_x) - 0.12, max(all_x) + 0.12)
    ax.set_ylim(min(all_y) - 0.12, max(all_y) + 0.12)

    fig.subplots_adjust(bottom=0.14)
    ax.set_position([0.04, 0.22, 0.92, 0.75])
    if sc is not None:
        ax_cb = fig.add_axes([0.15, 0.04, 0.70, CB_HEIGHT])
        cb = fig.colorbar(sc, cax=ax_cb, orientation="horizontal")
        cb.set_label(f"R$_0$ (white: R$_0$ < 1)", labelpad=3)
        cb.ax.tick_params(length=2.5, width=0.6)
        cb.outline.set_linewidth(0.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved fitness plot -> {output_path}")

###################
##Rendering
###################
def save_still_png(window_start, ab_time, edge_path, node_path, top_set,
                   pos, hd1_pairs, top_nodes, flux_norm, node_norm,
                   output_path, edge_cmap, edge_style="both", min_flux_thresh=0.0, annotations=None):
    edge_lookup = _load_single_window(edge_path, top_set, "edge", window_start)
    node_lookup = _load_single_window(node_path, top_set, "node", window_start)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("white")
    add_colorbars(fig, edge_cmap, flux_norm, node_norm)
    ax.set_position([0.04, CB_PLOT_BOTTOM + 0.02, 0.92, 1.0 - CB_PLOT_BOTTOM - 0.04])

    draw_flux_panel(ax=ax, window_start=window_start, edge_lookup=edge_lookup, node_lookup=node_lookup, pos=pos, hd1_pairs=hd1_pairs, top_nodes=top_nodes, flux_norm=flux_norm, node_norm=node_norm, ab_time=ab_time, annotations=annotations, edge_style=edge_style, min_flux_thresh=min_flux_thresh, edge_cmap=edge_cmap, show_title=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved still -> {output_path}")

def make_single_movie(ab_time, edge_path, node_path, top_set, all_windows,
                      pos, hd1_pairs, top_nodes, flux_norm, node_norm,
                      output_path, fps, edge_style, min_flux_thresh,
                      edge_cmap, annotations=None):
    label = "No Ab" if ab_time is None else f"Ab time {ab_time}"
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("white")
    add_colorbars(fig, edge_cmap, flux_norm, node_norm)
    ax.set_position([0.04, CB_PLOT_BOTTOM + 0.02, 0.92, 1.0 - CB_PLOT_BOTTOM - 0.02 - 0.06])

    writer = FFMpegWriter(fps=fps, metadata={"title": label}, bitrate=2000)
    n_frames = len(all_windows)
    with writer.saving(fig, output_path, dpi=150):
        edge_iter = _iter_windows_from_csv(edge_path, top_set, "edge")
        node_iter = _iter_windows_from_csv(node_path, top_set, "node")

        frame_idx = 0
        for (e_key, edge_data), (n_key, node_data) in zip(edge_iter, node_iter):
            edge_lookup = {e_key: edge_data}
            node_lookup = {n_key: node_data}
            draw_flux_panel(ax=ax, window_start=e_key[0], edge_lookup=edge_lookup, node_lookup=node_lookup, pos=pos, hd1_pairs=hd1_pairs, top_nodes=top_nodes, flux_norm=flux_norm, node_norm=node_norm, ab_time=ab_time, annotations=annotations, edge_style=edge_style, min_flux_thresh=min_flux_thresh, edge_cmap=edge_cmap)
            writer.grab_frame()
            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == n_frames:
                print(f"Frame {frame_idx}/{n_frames}")

    plt.close(fig)
    print(f"Saved movie -> {output_path}")

###################
##CLI and main
###################
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Render windowed mutational flux graphs as movies and/or PNG stills.\n\n"
            "Modes:\n"
            "  render      (default) Produce movies / stills from windowed CSVs.\n"
            "  top-nodes   Compute and save the top-N node list only (no windowed\n"
            "              files needed). Requires --total-node-files and\n"
            "              --save-top-nodes; exits after writing the list."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--mode", choices=["render", "top-nodes"], default="render", help="'render': full pipeline. 'top-nodes': write node list and exit.")

    rnd = p.add_argument_group("render mode")
    rnd.add_argument("--edge-files", nargs="+", default=None, metavar="PATH", help="Windowed edge CSV files (one per Ab administration time).")
    rnd.add_argument("--node-files", nargs="+", default=None, metavar="PATH", help="Windowed node CSV files, same order as --edge-files.")
    rnd.add_argument("--fitness-file", default=None, metavar="PATH", help="CSV with 'sequence' and 'ddG' for the R0 plot.")
    rnd.add_argument("--save-frames", nargs="+", type=float, default=None, metavar="T", help="Save a PNG still (frame_t{NNNN}.png) at each window_start T.")
    rnd.add_argument("--norm-file", default=None, metavar="PATH", help="Output path for flux_norms.json (default: <output-dir>/flux_norms.json).")

    top = p.add_argument_group("top-node selection (both modes)")
    top.add_argument("--total-node-files", nargs="+", default=None, metavar="PATH", help="Total (non-windowed) node CSVs for ranking.")
    top.add_argument("--save-top-nodes", default=None, metavar="PATH", help="Write the top-N node list to this file.")
    top.add_argument("--load-top-nodes", default=None, metavar="PATH", help="Load a top-N node list.")
    top.add_argument("--top-n", type=int, default=40, help="Number of top nodes to use.")

    vis = p.add_argument_group("visual options")
    vis.add_argument("--annotate", nargs="*", default=[], metavar="SEQ:color", help="Highlight nodes with a coloured ring: DIRQH:red AIRQH:blue")
    vis.add_argument("--min-flux", type=float, default=0.0, help="Skip edges below this flux value.")
    vis.add_argument("--flux-vmin-pct", type=float, default=5.0, help="Percentile of positive flux values used as colormap vmin.")
    vis.add_argument("--abundance-vmin-pct", type=float, default=5.0, help="Percentile of positive abundance values used as colormap vmin.")
    vis.add_argument("--edge-style", choices=["color", "thickness", "both"], default="both")
    vis.add_argument("--edge-cmap", default="plasma")
    vis.add_argument("--fps", type=int, default=8)
    vis.add_argument("--output-dir", default="movies")

    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "top-nodes":
        if not args.total_node_files:
            sys.exit("top-nodes mode requires --total-node-files.")
        if not args.save_top_nodes:
            sys.exit("top-nodes mode requires --save-top-nodes.")
        print(f"Selecting top {args.top_n} nodes ...")
        top_nodes = select_top_nodes_from_totals(args.total_node_files, args.top_n)
        save_top_nodes(args.save_top_nodes, top_nodes)
        return

    if not args.edge_files or not args.node_files:
        sys.exit("render mode requires --edge-files and --node-files.")
    if len(args.edge_files) != len(args.node_files):
        sys.exit("--edge-files and --node-files must have the same number of entries.")

    annotations = parse_annotations(args.annotate)
    if annotations:
        print("Annotating: " + ", ".join(f"{s}({c})" for s, c in annotations.items()))

    ab_times = [parse_ab_time(ef) for ef in args.edge_files]
    order = sorted(range(len(ab_times)), key=lambda i: ab_times[i])
    ab_times = [ab_times[i] for i in order]
    edge_files = [args.edge_files[i] for i in order]
    node_files = [args.node_files[i] for i in order]

    if args.load_top_nodes:
        top_nodes = load_top_nodes(args.load_top_nodes)
    elif args.total_node_files:
        print(f"Selecting top {args.top_n} nodes ...")
        top_nodes = select_top_nodes_from_totals(args.total_node_files, args.top_n)
    else:
        sys.exit("Provide either --load-top-nodes or --total-node-files.")

    if args.save_top_nodes:
        save_top_nodes(args.save_top_nodes, top_nodes)

    top_set = set(top_nodes)

    print("Computing HD-1 pairs and layout ...")
    hd1_pairs = find_hd1_pairs(top_nodes)
    print(f"{len(hd1_pairs)} directed HD-1 edges.")
    pos = compute_layout(top_nodes, hd1_pairs)

    all_edge_sources = edge_files
    all_windows = get_all_windows(all_edge_sources)
    print(f"{len(all_windows)} windows found.")

    all_edge_paths = edge_files
    all_node_paths = node_files
    flux_norm, node_norm = compute_log_norms_streaming(all_edge_paths, all_node_paths, top_set, flux_pct=args.flux_vmin_pct, abundance_pct=args.abundance_vmin_pct)
    print(f"Flux norm:      vmin={flux_norm.vmin:.3g}  vmax={flux_norm.vmax:.3g}")
    print(f"Abundance norm: vmin={node_norm.vmin:.3g}  vmax={node_norm.vmax:.3g}")

    norm_path = args.norm_file or os.path.join(args.output_dir, "flux_norms.json")
    save_norm_file(norm_path, flux_norm, node_norm)

    edge_cmap = matplotlib.colormaps[args.edge_cmap]

    if args.fitness_file:
        print("Generating fitness plot ...")
        make_fitness_plot(fitness_file=args.fitness_file, top_nodes=top_nodes, pos=pos, hd1_pairs=hd1_pairs, output_path=os.path.join(args.output_dir, "fitness_static.png"), params=default_params(), annotations=annotations)

    if args.save_frames:
        print(f"Saving {len(args.save_frames)} PNG still(s) ...")
        if ab_times:
            t_still = ab_times[0]
            ef_still = edge_files[0]
            nf_still = node_files[0]

        for ws_req in args.save_frames:
            save_still_png(window_start=ws_req, ab_time=t_still, edge_path=ef_still, node_path=nf_still, top_set=top_set, pos=pos, hd1_pairs=hd1_pairs, top_nodes=top_nodes, flux_norm=flux_norm, node_norm=node_norm, output_path=os.path.join(args.output_dir, f"frame_t{int(ws_req):04d}.png"), edge_cmap=edge_cmap, edge_style=args.edge_style, min_flux_thresh=args.min_flux, annotations=annotations)

    for t, ef, nf in zip(ab_times, edge_files, node_files):
        make_single_movie(ab_time=t, edge_path=ef, node_path=nf, top_set=top_set, all_windows=all_windows, pos=pos, hd1_pairs=hd1_pairs, top_nodes=top_nodes, flux_norm=flux_norm, node_norm=node_norm, output_path=os.path.join(args.output_dir, f"flux_movie_time{t}.mp4"), fps=args.fps, edge_style=args.edge_style, min_flux_thresh=args.min_flux, edge_cmap=edge_cmap, annotations=annotations)

    print("Done.")

if __name__ == "__main__":
    matplotlib.use('Agg')
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial"],
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    })
    main()