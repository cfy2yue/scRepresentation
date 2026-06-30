# Third-Party Source Mirrors

`fm/third_party/` is only a placeholder in the lightweight code repo.

Default external root:

```text
<delivery_root>/scFM_third_party/
  Geneformer/
  uce/
  state/
  stack/
  scGPT-main/
  scFoundation/
  scldm/
  xVERSE_code/
  CellNavi/
  dataset_fitted_baseline/
```

Override with `SCFM_THIRD_PARTY_ROOT=/path/to/scFM_third_party`.

Run `PYTHONPATH=<delivery_root>/scFM/fm python -m tools.validate_resources
--skip-import-test` after placing sources and pretrained assets.
