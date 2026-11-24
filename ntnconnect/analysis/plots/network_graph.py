from __future__ import annotations
import numpy as np
from ntnconnect.analysis.plots.base import BasePlot, register_plot


@register_plot("network_graph")
class NetworkGraphPlot(BasePlot):
    """Topology plot. Expects a DataFrame of edges with sat_a/sat_b columns.
    cfg keys: title, layout (default 'spring')."""
    def render(self, df, cfg, out_paths):
        if df is None or df.empty: return
        try:
            import networkx as nx
        except ImportError:
            return
        G = nx.Graph()
        edges = df[["sat_a", "sat_b"]].drop_duplicates().values.tolist()
        G.add_edges_from((int(a), int(b)) for a, b in edges)
        layout = cfg.get("layout", "spring")
        pos = nx.spring_layout(G, seed=42) if layout == "spring" else nx.circular_layout(G)
        fig, ax = self._fig(figsize=(9, 6))
        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_size=30, ax=ax)
        ax.set_title(cfg.get("title", "Active topology"))
        ax.axis("off")
        self._save(fig, out_paths)
