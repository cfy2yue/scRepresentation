from pathlib import Path


def test_full_stack_launcher_exposes_selection_metric_env_knob():
    script = Path("model/latent/scripts/run_full_stack_latentfm.sh").read_text()
    assert 'SELECTION_METRIC="${SELECTION_METRIC:-test_mmd}"' in script
    assert 'SELECTION_MMD_LAMBDA="${SELECTION_MMD_LAMBDA:-1.0}"' in script
    assert '--selection-metric "${SELECTION_METRIC}"' in script
    assert '--selection-mmd-lambda "${SELECTION_MMD_LAMBDA}"' in script
