"""
Chemical perturbation embedding package (ChemPert) — placeholders and contracts only.

Concrete encoders live here or in sibling packages once a backbone is chosen
(e.g. UniMol caches, fingerprints). The trainer reads vectors via:

* :func:`condition_emb.genepert.chem_embedding_hook.resolve_chem_embedding`
  (wired from ``CoupledFMDataset`` / ``CrossDatasetFMDataset`` when
  ``pert_chem_enabled=True`` plus ``chem_emb_source_dir`` / obs column).

Implementations should populate :class:`~condition_emb.genepert.perturbation.ConditionMetadata`
``chem_emb`` before :class:`~condition_emb.genepert.perturbation.PerturbationBatch.from_metadata_list`
runs, or expose a NumPy/cache loader consumed by ``resolve_chem_embedding``.
"""


from .chem_resolver import (
    chem_keys_for_metadata,
    load_chemical_embed_backend,
    resolve_chemical_embeddings_for_metadata,
    resolve_first_chemical_embedding,
)
from .drug_cache import DrugEmbeddingCache, RandomDrugEmbeddingFallback, deterministic_standard_normal_vec


class ChemPertEncoderProvider:
    """Protocol-style placeholder: subclass and return ``(B, D)`` float tensors.

    Intended call site: dataset metadata enrichment or a dedicated collate_fn.
    Not imported by trainers directly until a concrete backbone is merged.
    """

    embed_dim: int = 0

    def encode_batch(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError(
            "Implement encode_batch in a ChemPertEncoderProvider subclass; "
            "see condition_emb/chempert/README.md"
        )


__all__ = [
    "ChemPertEncoderProvider",
    "DrugEmbeddingCache",
    "RandomDrugEmbeddingFallback",
    "chem_keys_for_metadata",
    "deterministic_standard_normal_vec",
    "load_chemical_embed_backend",
    "resolve_chemical_embeddings_for_metadata",
    "resolve_first_chemical_embedding",
]
