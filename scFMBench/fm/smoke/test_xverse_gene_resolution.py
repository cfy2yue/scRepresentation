import anndata as ad
import numpy as np
import pandas as pd

from adapters.xverse.encoder import _resolve_ensembl_series


def test_xverse_resolves_symbol_var_names_when_ensembl_column_missing():
    adata = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        var=pd.DataFrame(index=["POLR1C", "GXYLT1", "HLA-C"]),
    )
    vals, report = _resolve_ensembl_series(
        adata,
        gene_col=None,
        model_ensg=["ENSG00000171453", "ENSG00000151233", "ENSG00000204525"],
    )
    assert report["source"] == "var_names_symbol_map"
    assert report["mapped_ensg"] == 3
    assert report["aligned_to_xverse"] == 3
    assert vals.tolist() == ["ENSG00000171453", "ENSG00000151233", "ENSG00000204525"]


def test_xverse_ignores_symbol_like_ensembl_column_and_falls_back():
    adata = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        var=pd.DataFrame({"Ensembl_ID": ["CNIH4", "RNLS"]}, index=["CNIH4", "RNLS"]),
    )
    vals, report = _resolve_ensembl_series(
        adata,
        gene_col=None,
        model_ensg=["ENSG00000143771", "ENSG00000184719"],
    )
    assert report["source"] == "var_names_symbol_map"
    assert report["attempted_sources"][0]["source"] == "var_column:Ensembl_ID"
    assert report["attempted_sources"][0]["mapped_ensg"] == 0
    assert vals.tolist() == ["ENSG00000143771", "ENSG00000184719"]


def test_xverse_keeps_valid_ensembl_column():
    adata = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        var=pd.DataFrame({"Ensembl_ID": ["ENSG00000131374.7", "ENSG00000151233"]}, index=["TBC1D5", "GXYLT1"]),
    )
    vals, report = _resolve_ensembl_series(
        adata,
        gene_col=None,
        model_ensg=["ENSG00000131374", "ENSG00000151233"],
    )
    assert report["source"] == "var_column:Ensembl_ID"
    assert report["mapped_ensg"] == 2
    assert vals.tolist() == ["ENSG00000131374", "ENSG00000151233"]
