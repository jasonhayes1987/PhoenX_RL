# The PhoenX window

The PhoenX window edits a training config, builds a sweep file, and starts
the same commands you would type in a terminal. The form, the checks, and
the running process stay in Python. The window is only the screen.

Use it when you want to change a config without hand-editing YAML, or when
you want to watch a run. Use the terminal when you are already in a script
or on a machine with no display.

## Open it from the PhoenX environment

Install PhoenX in an environment first. From that environment:

```bash
phoenx-config gui
phoenx-config gui path/to/my_run.yml
phoenx-config gui --doctor
```

The command waits until you close the window, then returns that process's
exit code. A sweep file opens on Sweep Builder. Any other file opens on
Config Builder.

`--doctor` prints two lines and does not open the window:

```text
python: C:\path\to\python.exe
window: C:\path\to\phoenx-rl.exe
```

`window: MISSING` means this environment cannot find the program. Install
the window, or set `PHOENX_GUI` to the executable. If `PHOENX_GUI` is set
and that file is missing, the search stops. It does not try another copy.

## What the installer does, and what it does not

The Windows installer is per-user. It puts the window in
`%LOCALAPPDATA%\PhoenX RL`. It does not install Python. It does not install
PhoenX. A Start Menu launch cannot train until you point it at the
`python.exe` from the environment where PhoenX is installed.

The first prompt is titled **PhoenX Python**. Choose that interpreter.
The window runs `import phoenx` with it. On success it remembers the path.
On failure it asks once more. Cancel, or a second failure, closes the
window and saves nothing.

A later launch from the same user skips the prompt. `phoenx-config gui`
does not ask: it sets `PHOENX_PYTHON` to the interpreter that ran the
command.

## Build the window from a checkout

The checkout does not contain a compiled window. You need Node.js and the
official stable Rust toolchain. From the repository root:

```bash
cd desktop
npm install
npm run tauri build
```

The program is `desktop/src-tauri/target/release/phoenx-rl.exe` on Windows
(`phoenx-rl` on other systems). The NSIS installer is under
`desktop/src-tauri/target/release/bundle/nsis`. `phoenx-config gui` uses the
release program in the checkout when `PHOENX_GUI` is unset and that file
exists. Otherwise, on Windows, it uses
`%LOCALAPPDATA%\PhoenX RL\phoenx-rl.exe`.

## Pages

| Page | What it is for |
|------|----------------|
| Dashboard | Empty. Reserved for a later summary. |
| Config Builder | The training config. **Overview** has Agent, Environment and Schedule cards (only the settings the chosen agent uses) and the roots → trunk → branches diagram; click a block to edit its layers. **All fields** is the full form. **YAML** is what Save writes. |
| Sweep Builder | The sweep file. **Phases** shows each phase as a card and the selected phase's search table, overrides and defaults; **+ Add from base config** picks fields to search. |
| Launch | Train, resume, test, or sweep, with live timestep, episodes, average reward, best checkpoint and elapsed time from the trainer's output. |
| Settings | Empty. The Python prompt is not this page. |

Ctrl+S saves the page you are looking at. Ctrl+O opens a file into that
page. Those keys do nothing on Dashboard, Launch, or Settings.

## Common mistakes

| What you see | What it means |
|--------------|----------------|
| `PhoenX window not found...` | Build or install the window, or set `PHOENX_GUI`. |
| `window: MISSING` from `--doctor` | Same problem. The command did not start a window. |
| `That Python cannot import phoenx.` | The chosen `python.exe` is not the PhoenX environment. |
| The installer is installed, but training fails | The installer never included PhoenX. Install the package in an environment and choose that Python. |

## Related pages

- [Configuration Files](configurations.md) for the form and the Launch modes.
- [Hyperparameter Sweeps](hyperparameter-sweeps.md) for Sweep Builder.
- [Installation](installation.md) for the Python package itself.
