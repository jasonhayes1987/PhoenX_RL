"""Regression tests that a built wheel actually embeds example configs.

Editable installs resolve ``importlib.resources`` into the source tree, so a
broken ``package-data`` glob can ship an empty wheel while source-tree tests
still pass. These tests build a real wheel with ``pip wheel`` and inspect the
artifact with ``zipfile``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_SRC = REPO_ROOT / "src" / "phoenx" / "examples" / "configs"
WHEEL_CONFIG_PREFIX = "phoenx/examples/configs/"
DOCUMENTED_SAC = (
    "phoenx/examples/configs/LunarLanderContinuous-v3/sac.yml"
)
DOUBLED_SEGMENT = "phoenx/examples/phoenx/examples"
_BUILD_TIMEOUT_S = 180


def _git_porcelain() -> list[str]:
    """Return ``git status --porcelain`` lines for the repo root.

    Raises:
        pytest.skip.Exception: If ``git`` is unavailable.
    """
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not available to verify working-tree isolation")
    result = subprocess.run(
        [git, "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"git status failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line]


def _source_config_wheel_paths() -> set[str]:
    """Return expected wheel member paths for every source ``*.yml`` config.

    Paths are relative to the wheel root, e.g.
    ``phoenx/examples/configs/LunarLanderContinuous-v3/sac.yml``.
    """
    if not CONFIGS_SRC.is_dir():
        return set()
    return {
        f"{WHEEL_CONFIG_PREFIX}{path.relative_to(CONFIGS_SRC).as_posix()}"
        for path in CONFIGS_SRC.rglob("*.yml")
        if path.is_file()
    }


def _wheel_yml_paths(wheel_path: Path) -> set[str]:
    """Return ``.yml`` member paths under ``phoenx/examples/configs/``."""
    with zipfile.ZipFile(wheel_path) as zf:
        return {
            name
            for name in zf.namelist()
            if name.startswith(WHEEL_CONFIG_PREFIX) and name.endswith(".yml")
        }


def _wheel_all_names(wheel_path: Path) -> list[str]:
    """Return every member path inside the built wheel."""
    with zipfile.ZipFile(wheel_path) as zf:
        return list(zf.namelist())


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build ``phoenx-rl`` once into a disposable directory and return the wheel.

    Uses build isolation (required: env setuptools is older than the
    ``license = "MIT"`` SPDX form needs). Skips only when pip is missing or
    the build itself fails (e.g. offline / network error provisioning
    setuptools). Does not skip a successful build that produced a bad wheel.
    """
    # Invoke pip as a module of the test interpreter so we never hardcode
    # conda paths; skip if that module cannot be started.
    probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"pip cannot be invoked via {sys.executable!r}: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )

    outdir = tmp_path_factory.mktemp("phoenx-wheel")
    before = _git_porcelain()
    # Drop leftover setuptools build trees so a prior good build cannot mask a
    # broken package-data glob (stale build/lib would still contain the ymls).
    for stale in (REPO_ROOT / "build", *REPO_ROOT.glob("*.egg-info"),
                  *REPO_ROOT.joinpath("src").glob("*.egg-info")):
        if stale.exists():
            shutil.rmtree(stale)
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "-w",
            str(outdir),
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=_BUILD_TIMEOUT_S,
    )
    after = _git_porcelain()
    if before != after:
        pytest.fail(
            "pip wheel changed tracked git status; before="
            f"{before!r} after={after!r}"
        )

    if build.returncode != 0:
        detail = (build.stderr or build.stdout or "").strip()
        pytest.skip(f"pip wheel failed (likely build/network): {detail}")

    wheels = sorted(outdir.glob("phoenx_rl-*.whl"))
    if not wheels:
        pytest.fail(f"pip wheel succeeded but no phoenx_rl-*.whl in {outdir}")
    return wheels[0]


@pytest.mark.slow
def test_wheel_contains_at_least_one_example_config(built_wheel: Path) -> None:
    """Assert the wheel embeds at least one ``configs/**/*.yml`` resource.

    Guards the original failure mode where a misresolved package-data glob
    produced a wheel with zero example configs.
    """
    ymls = _wheel_yml_paths(built_wheel)
    assert ymls, (
        f"{built_wheel.name} contains no .yml under {WHEEL_CONFIG_PREFIX!r}"
    )


@pytest.mark.slow
def test_wheel_contains_documented_sac_config(built_wheel: Path) -> None:
    """Assert the docs-advertised LunarLander Continuous SAC config is present."""
    names = set(_wheel_all_names(built_wheel))
    assert DOCUMENTED_SAC in names, (
        f"{built_wheel.name} missing documented config {DOCUMENTED_SAC!r}"
    )


@pytest.mark.slow
def test_wheel_yml_set_matches_source_tree(built_wheel: Path) -> None:
    """Assert wheel config paths match ``src/phoenx/examples/configs/**/*.yml``.

    Catches partial-glob regressions that would ship only a subset of configs.
    """
    expected = _source_config_wheel_paths()
    assert expected, f"no source configs found under {CONFIGS_SRC}"
    actual = _wheel_yml_paths(built_wheel)
    assert actual == expected, (
        f"wheel config set mismatch vs source tree:\n"
        f"  missing from wheel: {sorted(expected - actual)}\n"
        f"  unexpected in wheel: {sorted(actual - expected)}"
    )


@pytest.mark.slow
def test_wheel_has_no_doubled_package_data_path(built_wheel: Path) -> None:
    """Assert no wheel member contains the doubled ``phoenx/examples`` segment.

    That path is the signature of a package-data glob resolved relative to
    the package directory but written as if it were repo-root-relative.
    """
    bad = [n for n in _wheel_all_names(built_wheel) if DOUBLED_SEGMENT in n]
    assert not bad, (
        f"{built_wheel.name} has doubled package-data paths: {bad}"
    )


@pytest.mark.slow
def test_wheel_contains_phoenx_package(built_wheel: Path) -> None:
    """Assert ``phoenx/__init__.py`` is in the wheel so emptiness cannot pass."""
    names = set(_wheel_all_names(built_wheel))
    assert "phoenx/__init__.py" in names, (
        f"{built_wheel.name} missing phoenx/__init__.py"
    )
