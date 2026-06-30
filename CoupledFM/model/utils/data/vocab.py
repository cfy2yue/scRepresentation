"""Gene vocabulary — intersection of CellNavi gene_name.txt and NicheNet nodes."""

import json


class GeneVocab:
    """Vocabulary built from CellNavi gene_name.txt ∩ NicheNet node list."""

    def __init__(self, gene_name_path: str, nichenet_node2idx_path: str):
        self.cellnavi_gene2token: dict[str, int] = {}
        with open(gene_name_path) as f:
            for i, line in enumerate(f):
                self.cellnavi_gene2token[line.strip()] = i

        with open(nichenet_node2idx_path) as f:
            nichenet_genes = set(json.load(f).keys())

        self.vocab_genes = sorted(
            self.cellnavi_gene2token.keys() & nichenet_genes,
            key=lambda g: self.cellnavi_gene2token[g],
        )
        self.gene2token: dict[str, int] = {
            g: self.cellnavi_gene2token[g] for g in self.vocab_genes
        }
        self.token2gene: dict[int, str] = {v: k for k, v in self.gene2token.items()}

    def __len__(self) -> int:
        return len(self.vocab_genes)

    def __contains__(self, gene: str) -> bool:
        return gene in self.gene2token
