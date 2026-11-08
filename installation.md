# Installation Guide

Two install modes, one command order. Copy and paste the sequence that matches
what you need. All Python dependencies — `gymnasium-robotics`, `envpool`,
`ale-py`, `ray`, and the rest — are declared in `pyproject.toml` and arrive with
the final `pip install` of PhoenX. On Windows, one build-time system tool must
be present first: SWIG from conda-forge (needed to compile `box2d-py`, which has
no cp312 Windows wheel). Use conda, not `pip install swig`: the PyPI package is
a Python console-script shim that cannot import itself under pip's build
isolation, so the `box2d-py` build fails with `ModuleNotFoundError: No module
named 'swig'`. conda-forge ships a native `swig.exe` that isolation cannot
break. Users of `setup.ps1` do not need to install SWIG themselves — the script
does it.

## Requirements

- **Python 3.12 exactly** (`requires-python = ">=3.12,<3.13"`). This matches
  Isaac Lab 3.0 / Isaac Sim 6.X.
- Windows 11 for Isaac Lab mode (Isaac Sim 6.2 does not support Windows 10).
  Gymnasium mode: Windows 10/11 or Linux.
- An NVIDIA GPU and driver **580.88 or newer** on Windows (hard requirement
  for Isaac Lab mode; strongly recommended for Gymnasium mode). CUDA 13.0
  PyTorch will not load on an older driver.
- **SWIG** (Windows, when installing PhoenX by hand):
  `conda install -y -c conda-forge swig`. Not required when using `setup.ps1`.

## Gymnasium mode (no Isaac, ~5 min)

`phoenx` below is only the default environment name. With `setup.ps1`, pass
`-EnvName <name>` to override it; the chosen name (and its full prefix) is
recorded in `.phoenx-env` after a successful install.

```bash
conda create -n phoenx python=3.12
conda activate phoenx
pip install --upgrade pip
pip install -U torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
conda install -y -c conda-forge swig
pip install -e ".[dev,docs]"
```

## Isaac Lab mode (~30-60 min)

Isaac Lab 3.0's `isaacsim` extra is installed with `uv`. pip cannot pass
`--overrides`, and that file is what pins torch 2.11.0, torchvision 0.26.0,
and `numpy>=2`. `--torch-backend cu130` selects the CUDA 13.0 wheels, so
there is no second torch install. PhoenX still installs last. Same default
environment name (`phoenx`); override with `setup.ps1 -EnvName <name>` when
using the script.

```bash
conda create -n phoenx python=3.12
conda activate phoenx
pip install --upgrade pip
pip install uv
uv pip install --python python "isaaclab[isaacsim,all]==3.0.0" --overrides https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt --extra-index-url https://pypi.nvidia.com --torch-backend cu130 --index-strategy unsafe-best-match
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
| `dev` | pytest, pytest-cov, hypothesis, black, isort, pylint, ruff, notebook | No |
| `docs` | mkdocs-material, mkdocstrings | No |

Plain `pip install -e .` is enough to run agents. The sequences above include
`[dev,docs]` for a full developer setup.

## Why the order matters

- **Isaac Lab goes first, through uv.** The v3.0.0 overrides file pins the
  CUDA torch build as part of that install, so a later `pip install -U torch`
  is not used in Isaac mode. Gymnasium mode still installs torch itself from
  the cu130 index, because torch wheels on PyPI are CPU-only for Windows.
- **SWIG goes before PhoenX** because `gymnasium[box2d]` pins
  `box2d-py==2.3.5`, which has no cp312 Windows wheel and must be built from
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
bootstrap → conda env (+ pip upgrade, interpreter assertion) → Isaac via uv
(if requested; this includes CUDA torch) → CUDA PyTorch (Gymnasium mode
only) → build prerequisites (swig via conda-forge) → PhoenX → verify →
write `.phoenx-env`. The script installs SWIG for you; the manual
`conda install … swig` step applies only to hand-run sequences.

```powershell
.\setup.ps1                          # Gymnasium + dev tooling
.\setup.ps1 -Isaac -Docs             # Everything
.\setup.ps1 -EnvName phoenx-test -NonEditable   # Throwaway env, end-user install
.\setup.ps1 -CondaRoot E:\Miniconda3 -EnvName rl_env   # Existing conda install
```

| Switch | Effect |
|--------|--------|
| `-Isaac` | Install Isaac Lab / Isaac Sim |
| `-Docs` | Install the `docs` extra |
| `-NoDev` | Skip the `dev` extra (dev is **on** by default) |
| `-NonEditable` | `pip install .` instead of `pip install -e .` |
| `-EnvName <name>` | Conda env name (default `phoenx`) |
| `-PythonVersion <ver>` | Python version (default `3.12`) |
| `-CondaRoot <path>` | Conda installation root (default `%USERPROFILE%\miniforge3`) |

The script installs Miniforge to `%USERPROFILE%\miniforge3` if conda is
absent at `-CondaRoot`, addresses the env by prefix (`-p`), writes
`.phoenx-env` after a successful install, and is safe to re-run.

## Activating in a new terminal

A fresh PowerShell or Git Bash session does not inherit the environment from
install. After `setup.ps1` (or after pointing an existing checkout with
`use-env.ps1`), activate from the repo root:

```powershell
.\scripts\activate.ps1
```

```bash
source scripts/activate.sh
```

Both scripts read the machine-local `.phoenx-env` record, load conda from the *recorded* root rather than whichever `conda`
happens to be first on `PATH`, and activate by full environment prefix. The
record stores a prefix rather than a name because the same environment name
can exist under multiple conda installations on one machine, and a name alone
is ambiguous.

To point an existing checkout at an environment you already have, without
reinstalling:

```powershell
.\scripts\use-env.ps1 -Prefix "E:\Miniconda3\envs\rl_env"
# or:
.\scripts\use-env.ps1 -CondaRoot "E:\Miniconda3" -EnvName rl_env
```

That writes `.phoenx-env`; subsequent `activate.ps1` / `activate.sh` calls use
it.

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

The install commands above target Isaac Lab 3.0. The only end-to-end run on
record is the previous stack (not a support matrix; Linux has not been
tested):

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

4. **`filelock` is older than 3.20.1 after the Isaac install.**
   Isaac Sim 5.1's pip resolver always did this. The 3.0 uv overrides do not
   pin `filelock`, so it may no longer happen. Fix, if
   `python -c "import importlib.metadata as m; print(m.version('filelock'))"`
   prints a version below 3.20.1 or the package is missing:
   `pip install -U "filelock>=3.20.1"`. `setup.ps1` runs that only in that case.

5. **`ERROR: Failed building wheel for box2d-py` / `command 'swig.exe' failed`.**
   Cause: `gymnasium[box2d]` pins `box2d-py==2.3.5`, which has no cp312
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
   an Isaac environment.** On Isaac Sim 5.1.0.0 that block listed exact pins
   from `isaacsim-core` and `isaacsim-kernel` (`filelock`, `packaging`,
   `aiohttp`, `aiohappyeyeballs`, `aiosignal`, `click`, `coverage`, `psutil`,
   `requests`, and `typing_extensions`) that PhoenX's dependency set had
   bumped. Those are warnings, not errors, and that 5.1 configuration was
   validated. A fresh Isaac Lab 3.0.0 / Isaac Sim 6.2.0.0 install can still
   print this warning after PhoenX. The imports succeed. `gymnasium[other]`
   used to pull `moviepy` 2.2.1, which caps Pillow below 12 and undoes
   `isaaclab`'s `pillow>=12.3.0`. That extra is no longer in
   `pyproject.toml`; Gymnasium is pinned to 1.4.0. `numpy`, `aiohttp`,
   `requests`, `urllib3`, and `yarl` can still differ from the exact
   versions named by `isaacsim-kernel`. Isaac Lab's own floors
   (`numpy>=2`, `urllib3>=2.8.0`) are the ones this install follows.
   `filelock` stayed at 3.25.2, so `setup.ps1` does not re-pin it unless a
   future install leaves it below 3.20.1.

8. **Red `NativeCommandError` when running `setup.ps1` with its output
   piped** (e.g. `.\setup.ps1 | Select-Object -Last 40`). The script sets
   `$ErrorActionPreference = "Stop"`, and piping with `2>&1` turns pip's
   ordinary stderr progress output into terminating errors. Run the script
   unpiped.

9. **Git Bash: `conda activate E:\Miniconda3\envs\rl_env` fails with
   `EnvironmentNameNotFound: Could not find conda environment:
   E:Miniconda3envsrl_env`.** Cause: backslash is bash's escape character, so
   an unquoted Windows path collapses and the separators vanish. Fix: quote
   it — `conda activate "E:\Miniconda3\envs\rl_env"` — or use a POSIX path, or
   `source scripts/activate.sh`, which quotes the recorded prefix for you.

10. **Git Bash: `conda activate <anything>` fails with `CondaError: Run
    'conda init' before 'conda activate'`.** Cause: conda was never
    initialized for bash; on a fresh Windows setup there is often no
    `~/.bashrc`, `~/.bash_profile`, or `~/.profile` at all. Fix: run
    `eval "$(conda shell.bash hook)"` first, which is exactly what
    `scripts/activate.sh` does for you.

A stale per-user site-packages copy of `typing_extensions` can shadow the
env's copy and produce `cannot import name 'Sentinel'`. `setup.ps1` and the
activation scripts set `PYTHONNOUSERSITE=1` to prevent it.
