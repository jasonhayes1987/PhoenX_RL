# Installation Guide

Two install modes, one command order. Copy and paste the sequence that matches
what you need. All Python dependencies — `gymnasium-robotics`, `envpool`,
`ale-py`, `ray`, and the rest — are declared in `pyproject.toml` and arrive with
the final `pip install` of PhoenX. On Windows, one build-time system tool must
be present first: SWIG from conda-forge (needed to compile `box2d-py`, which has
no cp311 Windows wheel). Use conda, not `pip install swig`: the PyPI package is
a Python console-script shim that cannot import itself under pip's build
isolation, so the `box2d-py` build fails with `ModuleNotFoundError: No module
named 'swig'`. conda-forge ships a native `swig.exe` that isolation cannot
break. Users of `setup.ps1` do not need to install SWIG themselves — the script
does it.

## Requirements

- **Python 3.11 exactly** (`requires-python = ">=3.11,<3.12"`). This matches
  Isaac Sim 5.X; Isaac Sim 6.X would move the pin to 3.12.
- Windows 10/11 or Linux
- An NVIDIA GPU with recent drivers for CUDA (hard requirement for Isaac Lab
  mode; strongly recommended for Gymnasium mode)
- **SWIG** (Windows, when installing PhoenX by hand):
  `conda install -y -c conda-forge swig`. Not required when using `setup.ps1`.

## Gymnasium mode (no Isaac, ~5 min)

```bash
conda create -n phoenx python=3.11
conda activate phoenx
pip install --upgrade pip
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge swig
pip install -e ".[dev,docs]"
```

## Isaac Lab mode (~30-60 min)

Identical to Gymnasium mode, with one extra command inserted before PyTorch.
PhoenX still installs last:

```bash
conda create -n phoenx python=3.11
conda activate phoenx
pip install --upgrade pip
pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install -y -c conda-forge swig
pip install -e ".[dev,docs]"
```

## No-clone install

Replace the final line of either sequence with:

```bash
pip install "phoenx-rl @ git+https://github.com/jasonhayes1987/PhoenX_RL.git"
```

This is not Isaac-specific — it is simply "install PhoenX". Isaac support
comes automatically when you run it inside an environment that already has
Isaac Lab installed. The SWIG step still applies on Windows before this line.

## Extras

| Extra | Packages | Required? |
|-------|----------|-----------|
| `dev` | pytest, pytest-cov, hypothesis, black, isort, pylint, notebook | No |
| `docs` | mkdocs-material, mkdocstrings | No |

Plain `pip install -e .` is enough to run agents. The sequences above include
`[dev,docs]` for a full developer setup.

## Why the order matters

- **Isaac Lab goes first** because `isaacsim` declares its own torch
  dependency, and whichever torch is installed *last* wins. Installing Isaac
  first and then force-installing the cu128 build with `-U` guarantees the
  CUDA build is the one that survives. This mirrors NVIDIA's own Isaac Lab
  pip-installation page.
- **The cu128 index is required on Windows in both modes**, because torch
  wheels on PyPI are CPU-only for Windows.
- **SWIG goes before PhoenX** because `gymnasium[box2d]` pins
  `box2d-py==2.3.5`, which has no cp311 Windows wheel and must be built from
  source. That build shells out to `swig.exe`.
- **PhoenX goes last** because its dependency constraints are deliberately
  floors (`torch>=2.5`, `numpy>=1.26`). Once Isaac has installed satisfying
  versions, pip sees the constraints already met and leaves them alone.
  Installing PhoenX earlier lets Isaac's resolver churn over PhoenX's
  resolution.
- **Isaac Sim/Lab cannot be declared in `pyproject.toml` at all**: it comes
  from NVIDIA's package index (`--extra-index-url https://pypi.nvidia.com`),
  and PEP 508 metadata has no portable way to express a custom index. That is
  why it is a documented step rather than a dependency.

## Windows shortcut: `setup.ps1`

`setup.ps1` mirrors the sequences above exactly. Step order: Miniforge
bootstrap → conda env (+ pip upgrade, interpreter assertion) → Isaac (if
requested) → CUDA PyTorch → build prerequisites (swig via conda-forge) →
PhoenX → verify. The script installs SWIG for you; the manual
`conda install … swig` step applies only to hand-run sequences.

```powershell
.\setup.ps1                          # Gymnasium + dev tooling
.\setup.ps1 -Isaac -Docs             # Everything
.\setup.ps1 -EnvName phoenx-test -NonEditable   # Throwaway env, end-user install
```

| Switch | Effect |
|--------|--------|
| `-Isaac` | Install Isaac Lab / Isaac Sim |
| `-Docs` | Install the `docs` extra |
| `-NoDev` | Skip the `dev` extra (dev is **on** by default) |
| `-NonEditable` | `pip install .` instead of `pip install -e .` |
| `-EnvName <name>` | Conda env name (default `phoenx`) |
| `-PythonVersion <ver>` | Python version (default `3.11`) |

The script installs Miniforge to `%USERPROFILE%\miniforge3` if conda is
absent, addresses the env by prefix (`-p`), and is safe to re-run.

## Verification

Run these from a directory **outside** the repo — that is the real proof the
package is installed rather than merely present on disk:

```bash
python -c "import phoenx; print(phoenx.__version__)"
phoenx-train --help
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import isaaclab"        # Isaac mode only
```

The test suite is not shipped in the package, so run it from **inside** a clone:

```bash
pytest -q -m "not isaac"           # requires the dev extra
pytest -q -m isaac                 # Isaac mode only; boots a live Isaac Sim
```

## Validated configuration

One configuration has been run end to end (not a support matrix; Linux has
not been tested):

- Windows 11, RTX 4090, driver 581.08, Python 3.11.15
- Isaac Lab 2.3.2.post1 with Isaac Sim 5.1.0.0, torch 2.7.0+cu128, CUDA available
- Tests: 410 passed / 14 deselected in a Gymnasium-only environment; the same
  410 passed in the Isaac environment; and 13 passed / 1 skipped for the
  GPU-backed `isaac`-marked integration tests (live Isaac Sim, Cartpole and
  Franka reach)

## Troubleshooting

1. **Red `NativeCommandError` mentioning `AdroitHandRelocateDense-v1`.**
   `gymnasium-robotics` prints a `DeprecationWarning` to stderr; PowerShell
   renders *any* stderr as a red error block. It is harmless and does not need
   fixing.

2. **`pip` fails with `[Errno 2] No such file or directory` during the Isaac
   install.** Cause: Windows long-path support is disabled and `isaacsim`
   extras ship file paths over 260 characters. Fix: enable `LongPathsEnabled`
   (a one-time `HKLM` registry write requiring Administrator once).
   `setup.ps1` does this automatically when `-Isaac` is passed.

3. **Isaac Sim refuses to import without the EULA accepted.**
   Fix: set `OMNI_KIT_ACCEPT_EULA=yes`.

4. **Isaac's resolver downgrades `filelock`.**
   Fix: `pip install -U "filelock>=3.20.1"` after the Isaac install.
   `setup.ps1` does this automatically.

5. **`ERROR: Failed building wheel for box2d-py` / `command 'swig.exe' failed`.**
   Cause: `gymnasium[box2d]` pins `box2d-py==2.3.5`, which has no cp311
   Windows wheel, so pip builds from source and shells out to SWIG. Fix
   (manual install): `conda install -y -c conda-forge swig`, then retry
   `pip install -e ".[dev,docs]"`. `setup.ps1` installs SWIG automatically.

6. **`ModuleNotFoundError: No module named 'swig'` during the `box2d-py`
   build.** Cause: `swig` was installed from PyPI instead of conda-forge. The
   PyPI package is a console-script shim; under pip's build isolation it cannot
   import itself. Fix: `pip uninstall swig`, then
   `conda install -y -c conda-forge swig`.

7. **pip prints "pip's dependency resolver does not currently take into
   account all the packages that are installed" after installing PhoenX into
   an Isaac environment.** The block lists conflicts where `isaacsim-core` and
   `isaacsim-kernel` 5.1.0.0 pin exact versions that PhoenX's dependency set
   has bumped — specifically `filelock`, `packaging`, `aiohttp`,
   `aiohappyeyeballs`, `aiosignal`, `click`, `coverage`, `psutil`, `requests`,
   and `typing_extensions`. These are warnings, not errors. This exact
   configuration was validated: Isaac Sim boots, and the full test suite
   passes including the GPU-backed Isaac integration tests. The `filelock`
   warning is partly intentional — `setup.ps1` deliberately restores
   `filelock>=3.20.1` after the Isaac install.

8. **Red `NativeCommandError` when running `setup.ps1` with its output
   piped** (e.g. `.\setup.ps1 | Select-Object -Last 40`). The script sets
   `$ErrorActionPreference = "Stop"`, and piping with `2>&1` turns pip's
   ordinary stderr progress output into terminating errors. Run the script
   unpiped.

A stale per-user site-packages copy of `typing_extensions` can shadow the
env's copy and produce `cannot import name 'Sentinel'`. `setup.ps1` sets
`PYTHONNOUSERSITE=1` to prevent it.
