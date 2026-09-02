#!/usr/bin/env python3
import os
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from src.params import default_params
from src.analysis.visualize_flux_graph import _load_single_window, _compute_r0, _log_norm_value, _draw_nodes, load_top_nodes, load_norm_file, find_hd1_pairs, compute_layout, hamming_distance, draw_flux_panel, NODE_SIZE, NODE_EDGE_COLOR, NODE_EDGE_LW, ANNOT_LW, CB_HEIGHT, CB_BOTTOM, CB_LEFT_LEFT, CB_LEFT_WIDTH, CB_RIGHT_LEFT, CB_RIGHT_WIDTH, CB_PLOT_BOTTOM

def _scan_window_starts(path, chunksize=100000):
    starts = set()
    for chunk in pd.read_csv(path, usecols=["window_start"], chunksize=chunksize):
        starts.update(chunk["window_start"].unique().tolist())
    return sorted(starts)

class FluxPanelData:
    """Container for everything needed to draw flux panels for manuscript figures."""

    def __init__(self, edge_file, node_file, top_nodes, norm_file, ab_time=None, annotations=None, pos=None, edge_cmap="plasma", edge_style="both", min_flux_thresh=0.0, chunksize=100000):
        self.edge_file = edge_file
        self.node_file = node_file
        self.chunksize = chunksize

        if isinstance(top_nodes, str):
            self.top_nodes = load_top_nodes(top_nodes)
        else:
            self.top_nodes = list(top_nodes)
        self.top_set = set(self.top_nodes)

        self.flux_norm, self.node_norm = load_norm_file(norm_file)

        self.hd1_pairs = find_hd1_pairs(self.top_nodes)
        self.pos = pos if pos is not None else compute_layout(self.top_nodes, self.hd1_pairs)

        self.ab_time = ab_time
        self.annotations = annotations or {}
        self.edge_style = edge_style
        self.min_flux_thresh = min_flux_thresh
        self.edge_cmap = (matplotlib.colormaps[edge_cmap] if isinstance(edge_cmap, str) else edge_cmap)

        print(f"Scanning window index: {os.path.basename(edge_file)}")
        self._window_starts = _scan_window_starts(edge_file, chunksize=chunksize)

    @property
    def window_starts(self):
        return self._window_starts

    def closest_window_start(self, t):
        """Return the available window_start closest to t."""
        return min(self._window_starts, key=lambda s: abs(s - t))

    def draw(self, ax, window_start, show_title=True, title_fontsize=11, ab_time=None, annotations=None,
            edge_style=None, min_flux_thresh=None, edge_cmap=None, node_size=None, node_edge_color=None,
            node_edge_lw=None, annot_lw=None, show_edges=True, front_nodes=None):
        """Draw the flux panel for window_start onto *ax*.
        Optional visual overrides for the defaults in visualize_flux_graph for customization."""
        edge_lookup = _load_single_window(self.edge_file, self.top_set, "edge", window_start, self.chunksize)
        node_lookup = _load_single_window(self.node_file, self.top_set, "node", window_start, self.chunksize)
        draw_flux_panel(ax=ax, window_start=window_start, edge_lookup=edge_lookup, node_lookup=node_lookup, pos=self.pos, hd1_pairs=self.hd1_pairs, top_nodes=self.top_nodes, flux_norm=self.flux_norm, node_norm=self.node_norm, ab_time=ab_time if ab_time is not None else self.ab_time, annotations=annotations if annotations is not None else self.annotations, edge_style=edge_style if edge_style is not None else self.edge_style, min_flux_thresh=min_flux_thresh if min_flux_thresh is not None else self.min_flux_thresh, edge_cmap=edge_cmap if edge_cmap is not None else self.edge_cmap, show_title=show_title, title_fontsize=title_fontsize, show_edges=show_edges, front_nodes=front_nodes, node_size=node_size if node_size is not None else NODE_SIZE, node_edge_color=node_edge_color if node_edge_color is not None else NODE_EDGE_COLOR, node_edge_lw=node_edge_lw if node_edge_lw is not None else NODE_EDGE_LW, annot_lw=annot_lw if annot_lw is not None else ANNOT_LW)

class FitnessPanelData:
    """Container for drawing R0 fitness panels."""

    def __init__(self, fitness_file, top_nodes, wt_seq, params, annotations=None, pos=None, fitness_cmap="viridis", front_nodes=None):
        if isinstance(top_nodes, str):
            self.top_nodes = load_top_nodes(top_nodes)
        else:
            self.top_nodes = list(top_nodes)

        self.wt_seq = wt_seq
        self.params = default_params()
        self.annotations = annotations or {}
        self.fitness_cmap = (matplotlib.colormaps[fitness_cmap] if isinstance(fitness_cmap, str) else fitness_cmap)
        self.front_nodes = front_nodes
        self.hd1_pairs = find_hd1_pairs(self.top_nodes)
        self.pos = pos if pos is not None else compute_layout(self.top_nodes, self.hd1_pairs)

        df = pd.read_csv(fitness_file)
        if not {"sequence", "ddG"}.issubset(df.columns):
            raise ValueError(f"'{fitness_file}' must have 'sequence' and 'ddG' columns.")
        ddg_map = dict(zip(df["sequence"], df["ddG"]))

        self._r0_vals = []
        for s in self.top_nodes:
            ddg = ddg_map.get(s, np.nan)
            self._r0_vals.append(np.nan if (isinstance(ddg, float) and np.isnan(ddg)) else _compute_r0(ddg, params))

    def draw(self, ax, show_colorbar=True, colorbar_rect=None, annotations=None, fitness_cmap=None, node_size=None,
             node_edge_color=None, node_edge_lw=None, annot_lw=None, topo_lw=0.6, topo_alpha=0.5,
             topo_color="#bbbbbb", front_nodes=None):
        """Draw the flux panel for window_start onto *ax*.
        Optional visual overrides for the defaults in visualize_flux_graph for customization."""
        _annot = annotations if annotations is not None else self.annotations
        _cmap = fitness_cmap if fitness_cmap is not None else self.fitness_cmap
        if isinstance(_cmap, str):
            _cmap = matplotlib.colormaps[_cmap]
        _nsize = node_size if node_size is not None else NODE_SIZE
        _nec = node_edge_color if node_edge_color is not None else NODE_EDGE_COLOR
        _nelw = node_edge_lw if node_edge_lw is not None else NODE_EDGE_LW
        _alw = annot_lw if annot_lw is not None else ANNOT_LW

        r0_vals = self._r0_vals
        idx_missing = [i for i, v in enumerate(r0_vals) if isinstance(v, float) and np.isnan(v)]
        idx_below = [i for i, v in enumerate(r0_vals) if not (isinstance(v, float) and np.isnan(v)) and v < 1.0]
        idx_above = [i for i, v in enumerate(r0_vals) if not (isinstance(v, float) and np.isnan(v)) and v >= 1.0]
        r0_above = [r0_vals[i] for i in idx_above]
        fitness_norm = mcolors.Normalize(vmin=1.0, vmax=max(r0_above) if r0_above else 2.0)

        ax.clear()
        ax.axis("off")

        #Draw edges
        drawn = set()
        for (u, v) in self.hd1_pairs:
            key = tuple(sorted([u, v]))
            if key not in drawn:
                drawn.add(key)
                ax.plot([self.pos[u][0], self.pos[v][0]], [self.pos[u][1], self.pos[v][1]], color=topo_color, lw=topo_lw, zorder=1, alpha=topo_alpha)

        xs = [self.pos[s][0] for s in self.top_nodes]
        ys = [self.pos[s][1] for s in self.top_nodes]

        if idx_missing:
            ax.scatter([xs[i] for i in idx_missing], [ys[i] for i in idx_missing], s=_nsize, c="#dddddd", edgecolors=_nec, linewidths=_nelw, zorder=3)
        if idx_below:
            ax.scatter([xs[i] for i in idx_below], [ys[i] for i in idx_below], s=_nsize, c="white", edgecolors=_nec, linewidths=_nelw, zorder=3)

        node_colors_map = {}
        for i, s in enumerate(self.top_nodes):
            if i in idx_missing:
                node_colors_map[s] = "#dddddd"
            elif i in idx_below:
                node_colors_map[s] = "white"
            else:
                node_colors_map[s] = _cmap(fitness_norm(r0_vals[i]))

        sc = None
        if idx_above:
            sm = plt.cm.ScalarMappable(cmap=_cmap, norm=fitness_norm)
            sm.set_array(r0_above)
            sc = sm

        _front = front_nodes if front_nodes is not None else self.front_nodes
        if _front is None and _annot:
            _front = list(_annot.keys())

        _front_set = set(_front or [])
        front_edge_map = {}
        for s in _front_set:
            if _annot and s in _annot:
                front_edge_map[s] = _annot[s]
            elif s in self.pos:
                front_edge_map[s] = _nec

        _draw_nodes(ax, self.top_nodes, self.pos, node_colors_map, node_size=_nsize, node_edge_color=_nec, node_edge_lw=_nelw, front_edge_colors_map=front_edge_map if front_edge_map else None, annot_lw=_alw, zorder_bg=3, zorder_fg=6)

        all_x = [p[0] for p in self.pos.values()]
        all_y = [p[1] for p in self.pos.values()]
        data_w = max(all_x) - min(all_x) or 1.0
        data_h = max(all_y) - min(all_y) or 1.0
        PAD = 0.10
        cx = (min(all_x) + max(all_x)) / 2
        cy = (min(all_y) + max(all_y)) / 2
        try:
            fig = ax.get_figure()
            bbox = ax.get_window_extent(fig.canvas.get_renderer())
            aspect = bbox.width / bbox.height if bbox.height > 0 else 1.0
        except Exception:
            aspect = 1.0
        if aspect >= 1.0:
            target_w = data_h * aspect
            ax.set_xlim(cx - target_w / 2 * (1 + PAD), cx + target_w / 2 * (1 + PAD))
            ax.set_ylim(cy - data_h / 2 * (1 + PAD), cy + data_h / 2 * (1 + PAD))
        else:
            target_h = data_w / aspect
            ax.set_xlim(cx - data_w / 2 * (1 + PAD), cx + data_w / 2 * (1 + PAD))
            ax.set_ylim(cy - target_h / 2 * (1 + PAD), cy + target_h / 2 * (1 + PAD))

        if show_colorbar and sc is not None:
            rect = colorbar_rect or [0.15, 0.04, 0.70, CB_HEIGHT]
            ax_cb = ax.get_figure().add_axes(rect)
            cb = ax.get_figure().colorbar(sc, cax=ax_cb, orientation="horizontal")
            cb.set_label(f"R$_0$ (white: R$_0 < 1$)", fontsize=9, labelpad=3)
            cb.ax.tick_params(labelsize=7, length=2.5, width=0.6)
            cb.outline.set_linewidth(0.5)

def add_flux_colorbars(fig, data=FluxPanelData, rect_flux=None, rect_abund=None, label_flux="Mutation flux", label_abund="Viral abundance", label_fontsize=9, tick_fontsize=7):
    """Add shared edge-flux and node-abundance colorbars to fig."""
    if rect_flux  is None:
        rect_flux  = [CB_LEFT_LEFT,  CB_BOTTOM, CB_LEFT_WIDTH,  CB_HEIGHT]
    if rect_abund is None:
        rect_abund = [CB_RIGHT_LEFT, CB_BOTTOM, CB_RIGHT_WIDTH, CB_HEIGHT]

    colorbars = []
    for rect, cmap, norm, label in [(rect_flux,  data.edge_cmap, data.flux_norm, label_flux), (rect_abund, plt.cm.gray_r,  data.node_norm, label_abund)]:
        ax_cb = fig.add_axes(rect)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
        cb.set_label(label, fontsize=label_fontsize, labelpad=3)
        cb.ax.tick_params(labelsize=tick_fontsize, length=2.5, width=0.6)
        cb.outline.set_linewidth(0.5)
        colorbars.append(cb)

    return tuple(colorbars)

def load_panel_data_shared_layout(conditions, top_nodes, norm_file, edge_cmap="plasma", **shared_kwargs):
    """Build multiple FluxPanelData objects sharing the same node layout."""

    if isinstance(top_nodes, str):
        nodes = load_top_nodes(top_nodes)
    else:
        nodes = list(top_nodes)

    hd1_pairs = find_hd1_pairs(nodes)
    shared_pos = compute_layout(nodes, hd1_pairs)
    shared_kwargs.setdefault("edge_cmap", edge_cmap)

    return [FluxPanelData(top_nodes=nodes, norm_file=norm_file, pos=shared_pos, **{**shared_kwargs, **cond}) for cond in conditions]