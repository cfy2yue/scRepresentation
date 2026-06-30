"""NicheNet graph loading and per-batch subgraph (for future graph bias)."""

import pickle

import numpy as np
import torch

from model.utils.data.vocab import GeneVocab


class NicheNetGraph:
    """Loads NicheNet graph, filters to vocabulary."""

    def __init__(self, graph_pkl_path: str, vocab: GeneVocab):
        with open(graph_pkl_path, "rb") as f:
            g = pickle.load(f)

        nichenet_node2idx = g["node2idx"]
        raw_edge_index = g["edge_index"]

        nichenet_idx_to_token: dict[int, int] = {}
        for gene in vocab.vocab_genes:
            if gene in nichenet_node2idx:
                nichenet_idx_to_token[nichenet_node2idx[gene]] = vocab.gene2token[gene]

        vocab_nichenet_set = set(nichenet_idx_to_token.keys())
        mask = np.isin(raw_edge_index[0], list(vocab_nichenet_set)) & np.isin(
            raw_edge_index[1], list(vocab_nichenet_set)
        )

        remap = np.vectorize(nichenet_idx_to_token.get)
        self._edges_token_src = remap(raw_edge_index[0, mask]).astype(np.int64)
        self._edges_token_dst = remap(raw_edge_index[1, mask]).astype(np.int64)

        self._token_adj: dict[int, list[int]] = {}
        for s, d in zip(self._edges_token_src, self._edges_token_dst):
            self._token_adj.setdefault(int(s), []).append(int(d))

        print(
            f"[NicheNetGraph] Vocab size: {len(vocab)}, "
            f"Filtered edges: {mask.sum()}"
        )

    def build_edge_index(
        self,
        expressed_gene_tokens: torch.Tensor,
        device: str = "cpu",
        add_cls: bool = True,
    ) -> torch.Tensor:
        """Build sparse edge_index for expressed genes (+ optional CLS)."""
        tokens = expressed_gene_tokens.cpu().numpy()
        n_tokens = len(tokens)
        offset = 1 if add_cls else 0

        token_to_local = np.empty(40002, dtype=np.int64)
        token_to_local[:] = -1
        for i, tok in enumerate(tokens):
            token_to_local[tok] = i + offset

        token_set = set(tokens.tolist())
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []

        for token_int in token_set:
            neighbors = self._token_adj.get(token_int)
            if neighbors is None:
                continue
            arr = np.array(neighbors, dtype=np.int64)
            local_dst = token_to_local[arr]
            valid = local_dst >= 0
            if valid.any():
                local_src_val = token_to_local[token_int]
                n_valid = int(valid.sum())
                src_parts.append(np.full(n_valid, local_src_val, dtype=np.int64))
                dst_parts.append(local_dst[valid])

        if add_cls:
            gene_indices = np.arange(1, n_tokens + 1, dtype=np.int64)
            cls_zeros = np.zeros(n_tokens, dtype=np.int64)
            src_parts.append(cls_zeros)
            dst_parts.append(gene_indices)
            src_parts.append(gene_indices)
            dst_parts.append(cls_zeros)

        if not src_parts:
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        all_src = np.concatenate(src_parts)
        all_dst = np.concatenate(dst_parts)
        return torch.from_numpy(np.stack([all_src, all_dst])).to(device)
