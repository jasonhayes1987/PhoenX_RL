"""Pytest coverage for the PhoenX conda-env activation record and scripts.

Exercises ``scripts/use-env.ps1`` (via a throwaway copy under ``tmp_path`` so
the real repo-root ``.phoenx-env`` is never touched), static checks on
``.gitignore``, ``.vscode/settings.json``, and ``scripts/activate.sh``, and
integration probes that actually run ``activate.ps1`` / ``activate.sh`` against
the machine-local ``.phoenx-env`` record.

Windows / PowerShell / Git Bash dependencies skip cleanly on other platforms.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_USE_ENV = REPO_ROOT / "scripts" / "use-env.ps1"
REAL_ACTIVATE_PS1 = REPO_ROOT / "scripts" / "activate.ps1"
REAL_ACTIVATE_SH = REPO_ROOT / "scripts" / "activate.sh"
REAL_RECORD = REPO_ROOT / ".phoenx-env"
REAL_GITIGNORE = REPO_ROOT / ".gitignore"
REAL_VSCODE_SETTINGS = REPO_ROOT / ".vscode" / "settings.json"

_ACTIVATION_TIMEOUT_S = 60
_PHXTEST_RE = re.compile(r"^PHXTEST\|([^|\r\n]+)\|(.*)$", re.MULTILINE)

# Captured by the module-scoped protection fixture; compared at teardown and
# by the explicit isolation test.
_BEFORE_REAL_RECORD: bytes | None = None

GIT_BASH_CANDIDATES = (
    Path(r"C:\Program Files\Git\bin\bash.exe"),
    Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
)

_IS_WINDOWS = sys.platform == "win32"
_POWERSHELL = shutil.which("powershell") if _IS_WINDOWS else None


def _find_git_bash() -> Path | None:
    """Return the Git Bash executable, preferring known install paths.

    Avoids ``shutil.which("bash")``, which on Windows often resolves to WSL
    rather than Git Bash (backslash / ``source`` semantics differ).
    """
    for candidate in GIT_BASH_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


_GIT_BASH = _find_git_bash() if _IS_WINDOWS else None

requires_windows = pytest.mark.skipif(
    not _IS_WINDOWS, reason="Windows-only env activation scripts"
)
requires_powershell = pytest.mark.skipif(
    not _IS_WINDOWS or _POWERSHELL is None,
    reason="powershell not available",
)
requires_git_bash = pytest.mark.skipif(
    _GIT_BASH is None, reason="Git Bash not found at standard install paths"
)


# =============================================================================
# Helpers
# =============================================================================


def _snapshot_real_record() -> bytes | None:
    """Return the current repo-root ``.phoenx-env`` bytes, or None if absent."""
    if REAL_RECORD.is_file():
        return REAL_RECORD.read_bytes()
    return None


@pytest.fixture(scope="module", autouse=True)
def _protect_repo_phoenx_env():
    """Assert the real repo-root ``.phoenx-env`` is unchanged after this module.

    ``use-env.ps1`` always writes beside its own ``scripts/`` parent. Tests must
    only invoke a *copy* under ``tmp_path``; this fixture is the safety net.
    """
    global _BEFORE_REAL_RECORD
    _BEFORE_REAL_RECORD = _snapshot_real_record()
    yield
    after = _snapshot_real_record()
    assert after == _BEFORE_REAL_RECORD, (
        "Repo-root .phoenx-env was modified by tests; isolation failed. "
        f"before={_BEFORE_REAL_RECORD!r} after={after!r}"
    )


def _make_throwaway_layout(
    tmp_path: Path,
    *,
    conda_root_name: str = "Miniconda3",
    env_name: str = "rl_env",
    with_conda_exe: bool = True,
    with_python_exe: bool = True,
    with_prefix_dir: bool = True,
) -> tuple[Path, Path, Path]:
    """Build a fake conda tree and a copy of ``use-env.ps1`` under ``tmp_path``.

    Args:
        tmp_path: Pytest temporary directory (acts as fake repo root).
        conda_root_name: Leaf name for the fake conda install root.
        env_name: Environment leaf name under ``envs/``.
        with_conda_exe: If False, omit ``Scripts/conda.exe``.
        with_python_exe: If False, omit ``python.exe`` under the prefix.
        with_prefix_dir: If False, omit the env prefix directory entirely.

    Returns:
        ``(script_copy, conda_root, prefix)`` absolute paths.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_copy = scripts_dir / "use-env.ps1"
    shutil.copy2(REAL_USE_ENV, script_copy)

    conda_root = tmp_path / conda_root_name
    prefix = conda_root / "envs" / env_name

    if with_prefix_dir:
        prefix.mkdir(parents=True, exist_ok=True)
        if with_python_exe:
            (prefix / "python.exe").write_bytes(b"")

    scripts_bin = conda_root / "Scripts"
    scripts_bin.mkdir(parents=True, exist_ok=True)
    if with_conda_exe:
        (scripts_bin / "conda.exe").write_bytes(b"")

    return script_copy, conda_root, prefix


def _run_use_env(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Invoke a throwaway ``use-env.ps1`` via ``powershell -NoProfile -File``.

    Args:
        script: Absolute path to the copied ``use-env.ps1``.
        *args: Extra arguments forwarded to the script.

    Returns:
        Completed process with text stdout/stderr captured.
    """
    assert _POWERSHELL is not None
    cmd = [_POWERSHELL, "-NoProfile", "-File", str(script), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _parse_record(text: str) -> dict[str, str]:
    """Parse ``KEY='VALUE'`` lines from a ``.phoenx-env`` body.

    Args:
        text: Decoded record file contents.

    Returns:
        Mapping of keys to unquoted values.
    """
    pattern = re.compile(r"^(\w+)='(.*)'$")
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        match = pattern.match(line)
        assert match is not None, f"unexpected record line: {line!r}"
        out[match.group(1)] = match.group(2)
    return out


def _to_bash_path(path: Path) -> str:
    """Convert a Windows path to a form Git Bash can ``source``.

    Args:
        path: Absolute filesystem path.

    Returns:
        Forward-slash path string suitable for Git Bash.
    """
    return str(path.resolve()).replace("\\", "/")


def _norm_path(path: str | None) -> str:
    """Normalize a Windows (or Git-Bash POSIX) path for equality checks.

    Comparison is case-insensitive and ignores a trailing separator. POSIX
    paths like ``/e/Miniconda3/...`` are converted to ``E:\\Miniconda3\\...``.

    Args:
        path: Raw path string, or None / empty.

    Returns:
        Normalized path, or ``""`` when ``path`` is empty.
    """
    if not path:
        return ""
    text = path.strip().replace("\\", "/")
    posix = re.match(r"^/([A-Za-z])/(.*)$", text)
    if posix:
        text = f"{posix.group(1)}:/{posix.group(2)}"
    text = text.replace("/", "\\").rstrip("\\")
    return os.path.normcase(text)


def _paths_equal(left: str | None, right: str | None) -> bool:
    """Return True when two paths name the same location under ``_norm_path``.

    Args:
        left: First path.
        right: Second path.

    Returns:
        Whether the normalized forms are equal and non-empty.
    """
    a, b = _norm_path(left), _norm_path(right)
    return bool(a) and a == b


def _parse_phtest(stdout: str) -> dict[str, str]:
    """Extract ``PHXTEST|<key>|<value>`` sentinel lines from mixed stdout.

    Args:
        stdout: Combined child stdout (may include vcvars / conda noise).

    Returns:
        Mapping of sentinel keys to values (last write wins).
    """
    return {m.group(1): m.group(2) for m in _PHXTEST_RE.finditer(stdout)}


def _ps_encoded_command(script: str) -> str:
    """Base64-encode a PowerShell script as UTF-16LE for ``-EncodedCommand``.

    Args:
        script: PowerShell source text.

    Returns:
        ASCII base64 payload safe to pass as a single argv element.
    """
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _require_real_activation_record() -> dict[str, str]:
    """Load repo-root ``.phoenx-env`` or skip when activation cannot be probed.

    Returns:
        Parsed record mapping with ``PHOENX_ENV_PREFIX`` and
        ``PHOENX_CONDA_ROOT``.

    Raises:
        pytest.skip.Exception: When the record or ``conda.exe`` is missing.
    """
    if not REAL_RECORD.is_file():
        pytest.skip(".phoenx-env not present (machine-local / gitignored)")
    parsed = _parse_record(REAL_RECORD.read_text(encoding="utf-8"))
    prefix = parsed.get("PHOENX_ENV_PREFIX")
    conda_root = parsed.get("PHOENX_CONDA_ROOT")
    if not prefix or not conda_root:
        pytest.skip(".phoenx-env missing PHOENX_ENV_PREFIX or PHOENX_CONDA_ROOT")
    conda_exe = Path(conda_root) / "Scripts" / "conda.exe"
    if not conda_exe.is_file():
        pytest.skip(f"conda.exe absent at recorded root: {conda_exe}")
    if not REAL_ACTIVATE_PS1.is_file():
        pytest.skip(f"missing {REAL_ACTIVATE_PS1}")
    if not REAL_ACTIVATE_SH.is_file():
        pytest.skip(f"missing {REAL_ACTIVATE_SH}")
    return parsed


def _path_is_under_prefix(entry: str, env_prefix: str) -> bool:
    """Return True when a PATH entry equals or lies under ``env_prefix``.

    Args:
        entry: One PATH component.
        env_prefix: Recorded conda environment prefix.

    Returns:
        Whether ``entry`` is the prefix itself or a subdirectory of it.
    """
    norm_entry = _norm_path(entry)
    norm_prefix = _norm_path(env_prefix)
    if not norm_entry or not norm_prefix:
        return False
    return norm_entry == norm_prefix or norm_entry.startswith(norm_prefix + "\\")


def _cold_subprocess_env(env_prefix: str) -> dict[str, str]:
    """Build a child ``env=`` mapping with conda activation state stripped.

    Starts from ``os.environ``, drops every key starting with ``CONDA_`` plus
    ``_CE_CONDA`` / ``_CE_M``, and removes PATH entries that lie under
    ``env_prefix`` so ``python`` cannot resolve into the target env via a
    stale PATH entry alone.

    Args:
        env_prefix: Recorded ``PHOENX_ENV_PREFIX`` used to filter PATH.

    Returns:
        A new environment dict suitable for ``subprocess.run(..., env=...)``.
    """
    drop_exact = {"_CE_CONDA", "_CE_M"}
    cold = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CONDA_") and key not in drop_exact
    }
    raw_path = cold.get("PATH", "")
    kept = [
        entry
        for entry in raw_path.split(os.pathsep)
        if entry and not _path_is_under_prefix(entry, env_prefix)
    ]
    cold["PATH"] = os.pathsep.join(kept)
    return cold


def _assert_baseline_not_target(
    markers: dict[str, str], expected_prefix: str
) -> None:
    """Assert the cold child baseline is outside the recorded environment.

    With ``_cold_subprocess_env`` this is guaranteed by construction; a match
    means the probe setup itself is broken and must fail loudly.

    Args:
        markers: Parsed ``PHXTEST`` sentinels including ``BASE_CONDA_PREFIX``.
        expected_prefix: Recorded ``PHOENX_ENV_PREFIX``.
    """
    base = markers.get("BASE_CONDA_PREFIX", "")
    assert not _paths_equal(base, expected_prefix), (
        "cold probe setup broken: baseline CONDA_PREFIX already equals "
        f"target ({base!r}); child inherited activation state"
    )


def _run_powershell_activate_probe(
    *, no_profile: bool, env_prefix: str
) -> dict[str, str]:
    """Run ``activate.ps1`` in a cold PowerShell and return PHXTEST markers.

    The child starts with conda activation keys stripped (see
    ``_cold_subprocess_env``). Echoes baseline ``CONDA_PREFIX`` / python before
    invoking activation, then the post-activation values. Uses
    ``-EncodedCommand`` so nested quoting cannot mangle the probe.

    Args:
        no_profile: If True, pass ``-NoProfile``. If False, load the user
            profile the way an interactive VS Code / Cursor terminal would.
        env_prefix: Recorded prefix used to build the cold child environment.

    Returns:
        Parsed ``PHXTEST`` key/value markers from stdout.
    """
    assert _POWERSHELL is not None
    activate = str(REAL_ACTIVATE_PS1.resolve()).replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Continue'
Write-Output ('PHXTEST|BASE_CONDA_PREFIX|' + [string]$env:CONDA_PREFIX)
$basePy = ''
$baseCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $baseCmd) {{ $basePy = [string]$baseCmd.Source }}
Write-Output ('PHXTEST|BASE_PYTHON|' + $basePy)
& '{activate}'
Write-Output ('PHXTEST|CONDA_PREFIX|' + [string]$env:CONDA_PREFIX)
$py = ''
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $pyCmd) {{ $py = [string]$pyCmd.Source }}
Write-Output ('PHXTEST|PYTHON|' + $py)
"""
    cmd = [_POWERSHELL]
    if no_profile:
        cmd.append("-NoProfile")
    cmd.extend(["-EncodedCommand", _ps_encoded_command(script)])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_ACTIVATION_TIMEOUT_S,
        env=_cold_subprocess_env(env_prefix),
        check=False,
    )
    assert result.returncode == 0, (
        f"powershell activate probe failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    markers = _parse_phtest(result.stdout)
    assert "CONDA_PREFIX" in markers, (
        f"missing PHXTEST|CONDA_PREFIX in stdout:\n{result.stdout}"
    )
    assert "PYTHON" in markers, (
        f"missing PHXTEST|PYTHON in stdout:\n{result.stdout}"
    )
    return markers


def _assert_activated(markers: dict[str, str], expected_prefix: str) -> None:
    """Assert post-activation CONDA_PREFIX and python resolve to ``expected``.

    Args:
        markers: Parsed post-activation ``PHXTEST`` markers.
        expected_prefix: Recorded environment prefix.
    """
    assert _paths_equal(markers.get("CONDA_PREFIX"), expected_prefix), (
        f"CONDA_PREFIX={markers.get('CONDA_PREFIX')!r} "
        f"expected {expected_prefix!r}"
    )
    expected_python = str(Path(expected_prefix) / "python.exe")
    assert _paths_equal(markers.get("PYTHON"), expected_python), (
        f"python={markers.get('PYTHON')!r} expected {expected_python!r}"
    )


def _bash_probe_baseline(env_prefix: str) -> str:
    """Return a cold Git Bash session's ``CONDA_PREFIX`` (Windows form).

    Args:
        env_prefix: Recorded prefix used to build the cold child environment.

    Returns:
        Baseline prefix string (possibly empty).
    """
    assert _GIT_BASH is not None
    script = (
        'if [ -n "${CONDA_PREFIX-}" ] && command -v cygpath >/dev/null 2>&1; then '
        'printf "PHXTEST|BASE_CONDA_PREFIX|%s\\n" "$(cygpath -w "$CONDA_PREFIX")"; '
        "else "
        'printf "PHXTEST|BASE_CONDA_PREFIX|%s\\n" "${CONDA_PREFIX-}"; '
        "fi"
    )
    result = subprocess.run(
        [str(_GIT_BASH), "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_ACTIVATION_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        env=_cold_subprocess_env(env_prefix),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return _parse_phtest(result.stdout).get("BASE_CONDA_PREFIX", "")


def _run_bash_activate_probe(env_prefix: str) -> dict[str, str]:
    """Run ``activate.sh`` as ``bash --init-file ... -i -c`` from a cold env.

    Args:
        env_prefix: Recorded prefix used to build the cold child environment.

    Returns:
        Parsed ``PHXTEST`` markers for ``CONDA_PREFIX`` and ``PYTHON``.
    """
    assert _GIT_BASH is not None
    activate = _to_bash_path(REAL_ACTIVATE_SH)
    # Emit Windows paths via cygpath so assertions stay path-form agnostic.
    inner = (
        'if command -v cygpath >/dev/null 2>&1; then '
        'printf "PHXTEST|CONDA_PREFIX|%s\\n" "$(cygpath -w "$CONDA_PREFIX")"; '
        'printf "PHXTEST|PYTHON|%s\\n" "$(cygpath -w "$(command -v python)")"; '
        "else "
        'printf "PHXTEST|CONDA_PREFIX|%s\\n" "$CONDA_PREFIX"; '
        'printf "PHXTEST|PYTHON|%s\\n" "$(command -v python)"; '
        "fi"
    )
    result = subprocess.run(
        [str(_GIT_BASH), "--init-file", activate, "-i", "-c", inner],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_ACTIVATION_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        env=_cold_subprocess_env(env_prefix),
        check=False,
    )
    assert result.returncode == 0, (
        f"bash activate probe failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    markers = _parse_phtest(result.stdout)
    assert "CONDA_PREFIX" in markers, (
        f"missing PHXTEST|CONDA_PREFIX in stdout:\n{result.stdout}"
    )
    assert "PYTHON" in markers, (
        f"missing PHXTEST|PYTHON in stdout:\n{result.stdout}"
    )
    return markers


# =============================================================================
# Contract 6 — .gitignore
# =============================================================================


def test_gitignore_contains_phoenx_env_entry():
    """``.gitignore`` lists ``.phoenx-env`` so the record is never committed."""
    text = REAL_GITIGNORE.read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    assert ".phoenx-env" in lines


# =============================================================================
# Contract 7 — VS Code Git Bash profile arg order
# =============================================================================


def test_vscode_git_bash_init_file_before_interactive():
    """PhoenX Git Bash profile puts ``--init-file`` before ``-i`` in args.

    ``bash -i --init-file <f>`` is rejected with ``bash: --: invalid option``;
    ``bash --init-file <f> -i`` works. Skips when settings are absent (gitignored).
    """
    if not REAL_VSCODE_SETTINGS.is_file():
        pytest.skip(".vscode/settings.json not present (machine-local / gitignored)")

    data = json.loads(REAL_VSCODE_SETTINGS.read_text(encoding="utf-8"))
    profiles = data["terminal.integrated.profiles.windows"]
    bash_profile = profiles["PhoenX Git Bash"]
    args = bash_profile["args"]
    assert isinstance(args, list)
    assert "--init-file" in args
    assert "-i" in args
    assert args.index("--init-file") < args.index("-i")


# =============================================================================
# Contract 8 — activate.sh hygiene
# =============================================================================


def test_activate_sh_lf_only_no_set_e_no_bare_exit():
    """``activate.sh`` is LF-only, has no ``set -e``, and calls no bare ``exit``.

    Used via ``bash --init-file``, so exiting or aborting would close the terminal.
    """
    raw = REAL_ACTIVATE_SH.read_bytes()
    assert b"\r" not in raw, "activate.sh must use LF-only line endings (no CR)"

    text = raw.decode("utf-8")
    for line in text.splitlines():
        code = line.split("#", 1)[0].strip()
        if not code:
            continue
        assert "set -e" not in code, f"activate.sh must not use set -e: {line!r}"
        assert not re.search(r"\bexit\b", code), (
            f"activate.sh must not call bare exit: {line!r}"
        )


# =============================================================================
# Contract 1 — encoding (LF, no BOM)
# =============================================================================


@requires_powershell
def test_record_encoding_lf_no_bom(tmp_path: Path):
    """Written ``.phoenx-env`` has no UTF-8 BOM and zero ``0x0D`` bytes."""
    script, conda_root, prefix = _make_throwaway_layout(tmp_path)
    result = _run_use_env(script, "-Prefix", str(prefix), "-Quiet")
    assert result.returncode == 0, result.stderr

    record = tmp_path / ".phoenx-env"
    raw = record.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM must not be present"
    assert b"\x0d" not in raw, "CR bytes must not appear (LF-only required)"
    assert b"\n" in raw


# =============================================================================
# Contract 3 — single-quoted values and required keys
# =============================================================================


@requires_powershell
def test_record_single_quoted_values_and_keys(tmp_path: Path):
    """Record contains all three keys with single-quoted values."""
    script, conda_root, prefix = _make_throwaway_layout(tmp_path)
    result = _run_use_env(script, "-Prefix", str(prefix), "-Quiet")
    assert result.returncode == 0, result.stderr

    text = (tmp_path / ".phoenx-env").read_text(encoding="utf-8")
    assert "PHOENX_CONDA_ROOT='" in text
    assert "PHOENX_ENV_NAME='" in text
    assert "PHOENX_ENV_PREFIX='" in text

    parsed = _parse_record(text)
    assert set(parsed) == {
        "PHOENX_CONDA_ROOT",
        "PHOENX_ENV_NAME",
        "PHOENX_ENV_PREFIX",
    }
    assert parsed["PHOENX_CONDA_ROOT"] == str(conda_root)
    assert parsed["PHOENX_ENV_NAME"] == "rl_env"
    assert parsed["PHOENX_ENV_PREFIX"] == str(prefix)


# =============================================================================
# Contract 4 — Prefix / CondaRoot+EnvName resolution
# =============================================================================


@requires_powershell
def test_prefix_derives_env_name_and_conda_root(tmp_path: Path):
    """``-Prefix`` derives EnvName from the leaf and CondaRoot from ``envs\\``.

    Args:
        tmp_path: Isolated fake repo root.
    """
    script, conda_root, prefix = _make_throwaway_layout(
        tmp_path, conda_root_name="FakeConda", env_name="my_env"
    )
    result = _run_use_env(script, "-Prefix", str(prefix), "-Quiet")
    assert result.returncode == 0, result.stderr

    parsed = _parse_record((tmp_path / ".phoenx-env").read_text(encoding="utf-8"))
    assert parsed["PHOENX_ENV_NAME"] == "my_env"
    assert parsed["PHOENX_CONDA_ROOT"] == str(conda_root)
    assert parsed["PHOENX_ENV_PREFIX"] == str(prefix)


@requires_powershell
def test_conda_root_and_env_name_produce_prefix(tmp_path: Path):
    """``-CondaRoot`` / ``-EnvName`` form writes the expected ``envs\\`` prefix.

    Args:
        tmp_path: Isolated fake repo root.
    """
    script, conda_root, prefix = _make_throwaway_layout(
        tmp_path, conda_root_name="FakeConda", env_name="named_env"
    )
    result = _run_use_env(
        script,
        "-CondaRoot",
        str(conda_root),
        "-EnvName",
        "named_env",
        "-Quiet",
    )
    assert result.returncode == 0, result.stderr

    parsed = _parse_record((tmp_path / ".phoenx-env").read_text(encoding="utf-8"))
    assert parsed["PHOENX_ENV_PREFIX"] == str(prefix)
    assert parsed["PHOENX_CONDA_ROOT"] == str(conda_root)
    assert parsed["PHOENX_ENV_NAME"] == "named_env"


# =============================================================================
# Contract 5 — validation failures
# =============================================================================


@requires_powershell
def test_nonexistent_prefix_fails_without_writing(tmp_path: Path):
    """A missing prefix directory exits non-zero and does not write a record."""
    script, conda_root, _prefix = _make_throwaway_layout(tmp_path)
    missing = conda_root / "envs" / "does_not_exist"
    result = _run_use_env(script, "-Prefix", str(missing), "-Quiet")
    assert result.returncode != 0
    assert "does not exist" in (result.stderr + result.stdout)
    assert not (tmp_path / ".phoenx-env").exists()


@requires_powershell
def test_prefix_without_python_fails_without_writing(tmp_path: Path):
    """Prefix lacking ``python.exe`` exits non-zero and does not write a record."""
    script, _conda_root, prefix = _make_throwaway_layout(
        tmp_path, with_python_exe=False
    )
    result = _run_use_env(script, "-Prefix", str(prefix), "-Quiet")
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "python.exe" in combined
    assert not (tmp_path / ".phoenx-env").exists()


@requires_powershell
def test_conda_root_without_conda_exe_fails_without_writing(tmp_path: Path):
    """Conda root lacking ``Scripts\\conda.exe`` fails and writes no record."""
    script, conda_root, _prefix = _make_throwaway_layout(
        tmp_path, with_conda_exe=False, env_name="named_env"
    )
    result = _run_use_env(
        script,
        "-CondaRoot",
        str(conda_root),
        "-EnvName",
        "named_env",
        "-Quiet",
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "conda.exe" in combined
    assert not (tmp_path / ".phoenx-env").exists()


@requires_powershell
def test_no_arguments_fails_without_writing(tmp_path: Path):
    """Invoking ``use-env.ps1`` with no arguments exits non-zero."""
    script, _conda_root, _prefix = _make_throwaway_layout(tmp_path)
    result = _run_use_env(script)
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "Usage:" in combined
    assert not (tmp_path / ".phoenx-env").exists()


# =============================================================================
# Contract 2 — bash round-trip (core regression)
# =============================================================================


@requires_powershell
@requires_git_bash
def test_bash_source_preserves_windows_backslashes(tmp_path: Path):
    """Bash ``source`` of the record keeps Windows backslashes in the prefix.

    Unquoted ``E:\\Miniconda3\\envs\\rl_env`` collapses under bash because
    backslash is the escape character; single-quoted record values must survive
    a real ``source`` round-trip byte-identical.
    """
    script, _conda_root, prefix = _make_throwaway_layout(
        tmp_path, conda_root_name="Miniconda3", env_name="rl_env"
    )
    result = _run_use_env(script, "-Prefix", str(prefix), "-Quiet")
    assert result.returncode == 0, result.stderr

    record = tmp_path / ".phoenx-env"
    expected_prefix = str(prefix)
    bash_record = _to_bash_path(record)

    bash_result = subprocess.run(
        [
            str(_GIT_BASH),
            "-c",
            'source "$1" && printf %s "$PHOENX_ENV_PREFIX"',
            "bash",
            bash_record,
        ],
        capture_output=True,
        check=False,
    )
    assert bash_result.returncode == 0, bash_result.stderr.decode(
        "utf-8", errors="replace"
    )
    assert bash_result.stdout == expected_prefix.encode("utf-8")


# =============================================================================
# Isolation confirmation
# =============================================================================


def test_repo_root_phoenx_env_byte_identical_to_module_start():
    """Repo-root ``.phoenx-env`` bytes match the snapshot taken at module setup."""
    assert _snapshot_real_record() == _BEFORE_REAL_RECORD


# =============================================================================
# Integration — real activate.ps1 / activate.sh against machine-local record
# =============================================================================


@requires_powershell
def test_activate_ps1_no_profile_lands_in_recorded_env():
    """``activate.ps1`` under ``powershell -NoProfile`` sets prefix and python.

    Regression for the bug where the script printed success while leaving
    ``CONDA_PREFIX`` on base because the conda hook was gated on
    ``Get-Command conda``. Asserts real env vars, not the success line.
    The child starts cold (conda activation state stripped) so a parent
    shell already inside the target env cannot make this a no-op.
    """
    record = _require_real_activation_record()
    expected = record["PHOENX_ENV_PREFIX"]
    markers = _run_powershell_activate_probe(
        no_profile=True, env_prefix=expected
    )
    _assert_baseline_not_target(markers, expected)
    _assert_activated(markers, expected)


@requires_powershell
def test_activate_ps1_with_profile_lands_in_recorded_env():
    """``activate.ps1`` under profile-loaded PowerShell sets prefix and python.

    Same assertions as the ``-NoProfile`` probe, but with the user profile
    loaded the way an interactive VS Code / Cursor terminal would. The child
    still starts cold so inherited parent activation cannot skip the probe.
    """
    record = _require_real_activation_record()
    expected = record["PHOENX_ENV_PREFIX"]
    markers = _run_powershell_activate_probe(
        no_profile=False, env_prefix=expected
    )
    _assert_baseline_not_target(markers, expected)
    _assert_activated(markers, expected)


@requires_git_bash
def test_activate_sh_init_file_lands_in_recorded_env():
    """``bash --init-file activate.sh -i -c ...`` lands in the recorded env.

    Mirrors the VS Code Git Bash profile arg order (long option before ``-i``)
    and asserts both ``CONDA_PREFIX`` and the resolved ``python``. Baseline
    and activation children both use a cold environment.
    """
    record = _require_real_activation_record()
    expected = record["PHOENX_ENV_PREFIX"]
    baseline = _bash_probe_baseline(expected)
    _assert_baseline_not_target(
        {"BASE_CONDA_PREFIX": baseline}, expected
    )
    markers = _run_bash_activate_probe(expected)
    _assert_activated(markers, expected)


@requires_git_bash
def test_bash_init_file_after_interactive_is_rejected():
    """``bash -i --init-file <f>`` fails with ``invalid option`` (arg order).

    Negative control that makes the settings.json ordering assertion meaningful:
    the reversed flag order is rejected outright by Git Bash.
    """
    if not REAL_ACTIVATE_SH.is_file():
        pytest.skip(f"missing {REAL_ACTIVATE_SH}")
    assert _GIT_BASH is not None
    activate = _to_bash_path(REAL_ACTIVATE_SH)
    result = subprocess.run(
        [str(_GIT_BASH), "-i", "--init-file", activate, "-c", "true"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_ACTIVATION_TIMEOUT_S,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    assert result.returncode != 0
    combined = (result.stderr or "") + (result.stdout or "")
    assert "invalid option" in combined.lower()
