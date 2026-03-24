"""
TxGNN calls dgl.dataloading.EdgeDataLoader (removed in DGL 2.x).

The modern stack uses DataLoader + as_edge_prediction_sampler with the same
minibatch tuple (input_nodes, pos_graph, neg_graph, blocks). Call apply() after
dgl is importable (e.g. right after importing txgnn).
"""

from __future__ import annotations


def apply() -> None:
    import dgl.dataloading as dl

    if hasattr(dl, "EdgeDataLoader"):
        return

    def EdgeDataLoader(
        g,
        eids,
        block_sampler,
        device=None,
        kwargs_for_sampler=None,
        negative_sampler=None,
        exclude=None,
        reverse_eids=None,
        reverse_etypes=None,
        **kwargs,
    ):
        _ = kwargs_for_sampler
        edge_sampler = dl.as_edge_prediction_sampler(
            block_sampler,
            exclude=exclude,
            reverse_eids=reverse_eids,
            reverse_etypes=reverse_etypes,
            negative_sampler=negative_sampler,
        )
        return dl.DataLoader(
            g,
            eids,
            edge_sampler,
            device=device,
            **kwargs,
        )

    dl.EdgeDataLoader = EdgeDataLoader  # type: ignore[assignment]
