#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path("/data/cyx/1030/scLatent")
LAUNCHER = ROOT / "runs/latentfm_stack_composite_selection_20260618/launch_posthoc_if_ready.py"


def load_module():
    spec = importlib.util.spec_from_file_location("stack_posthoc_launcher", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load launcher: {LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_equal(got, expected, label: str) -> None:
    if got != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {got!r}")


def main() -> int:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        module.OUT_ROOT = tmp_path / "out"
        module.RUN_ROOT = tmp_path / "run"
        module.LOG = tmp_path / "logs/launcher.log"
        module.STATUS = tmp_path / "run/POSTHOC_LAUNCH_STATUS.md"
        module.POSTHOC_ONE = tmp_path / "run/posthoc_one.sh"
        module.POSTHOC_ONE.parent.mkdir(parents=True, exist_ok=True)
        module.POSTHOC_ONE.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

        tag = "ready_tag"
        module.TARGETS = [tag]
        launched: list[list[str]] = []
        tmux_sessions: set[str] = set()

        def fake_tmux_exists(session: str) -> bool:
            return session in tmux_sessions

        def fake_choose_gpus(max_jobs: int):
            assert_equal(max_jobs, 1, "max_jobs for one ready target")
            return [3], "chosen=[3]"

        def fake_check_call(cmd: list[str]):
            launched.append(cmd)
            return 0

        module.tmux_exists = fake_tmux_exists
        module.choose_gpus = fake_choose_gpus
        module.subprocess.check_call = fake_check_call

        # Missing run directory: no launch.
        rc = module.main()
        assert_equal(rc, 0, "missing run dir rc")
        assert_equal(launched, [], "missing run dir launch")
        assert "run_dir_missing" in module.STATUS.read_text(encoding="utf-8")

        # Active training session: no launch even if run dir/best exist.
        run_dir = module.OUT_ROOT / tag
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_text("dummy", encoding="utf-8")
        tmux_sessions.add(f"latentfm_{tag}")
        rc = module.main()
        assert_equal(rc, 0, "active training rc")
        assert_equal(launched, [], "active training launch")
        assert "training_active" in module.STATUS.read_text(encoding="utf-8")
        tmux_sessions.clear()

        # Ready target launches exactly one posthoc tmux command.
        rc = module.main()
        assert_equal(rc, 0, "ready rc")
        assert_equal(len(launched), 1, "ready launch count")
        assert_equal(launched[0][:5], ["tmux", "new-session", "-d", "-s", f"posthoc_{tag}"], "tmux prefix")
        assert str(module.POSTHOC_ONE) in launched[0][-1]
        assert "ready_tag" in launched[0][-1]
        assert " 3" in launched[0][-1]

        # Existing posthoc outputs: no duplicate launch.
        launched.clear()
        posthoc = run_dir / "posthoc_eval"
        posthoc.mkdir()
        (posthoc / "split_group_eval_best_ode20_mse2048_mmd2048.json").write_text("{}", encoding="utf-8")
        (posthoc / "condition_family_eval_best_ode20_mse2048_mmd2048.json").write_text("{}", encoding="utf-8")
        rc = module.main()
        assert_equal(rc, 0, "complete posthoc rc")
        assert_equal(launched, [], "complete posthoc launch")
        assert "posthoc_complete" in module.STATUS.read_text(encoding="utf-8")

    print("stack posthoc launcher validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
