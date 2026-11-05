"""``phoenx-config``: generate the config JSON Schema and check config files.

Subcommands:

* ``schema`` writes the JSON Schema generated from the installed PhoenX
  source (see [phoenx.config_schema][]).
* ``validate`` checks YAML configs against it, plus the cross-section rules
  the schema cannot express (currently the n-step horizon), without building
  an environment or touching torch.
* ``vscode`` writes the schema and points the VS Code YAML extension at it,
  for autocomplete, hover docs and inline errors while editing configs.
* ``gui`` opens the config editor window ([phoenx.config_gui][]).

Example:
    ``phoenx-config validate my_run.yml``

    ``phoenx-config gui my_run.yml``
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

from phoenx.config_schema import build_schema, validate_config
from phoenx.config_utils import resolve_n_step

DEFAULT_SCHEMA_NAME = "phoenx-config.schema.json"
DEFAULT_GLOBS = ["**/configs/**/*.yml", "**/fixtures/*.yml"]


def _write_schema(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_schema(), indent=2) + "\n", encoding="utf-8")
    return path


def check_file(path: Path) -> list[str]:
    """Return every problem found in one config file (empty when it is valid).

    Args:
        path: YAML training config.

    Returns:
        Error strings of the form ``path.to.key: message``.
    """
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"<yaml>: {exc}"]
    if not isinstance(config, dict):
        return ["<root>: a config must be a YAML mapping"]
    errors = validate_config(config)
    try:
        resolve_n_step(copy.deepcopy(config))
    except ValueError as exc:
        errors.append(f"agent.config.N: {exc}")
    return errors


def _cmd_schema(args: argparse.Namespace) -> int:
    if args.output == "-":
        json.dump(build_schema(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"Wrote {_write_schema(Path(args.output))}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    failed = 0
    for name in args.files:
        path = Path(name)
        errors = check_file(path)
        if errors:
            failed += 1
            print(f"FAIL {path}")
            for err in errors:
                print(f"  {err}")
        else:
            print(f"OK   {path}")
    return 1 if failed else 0


def _cmd_vscode(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    schema_path = _write_schema(root / args.schema)
    settings_path = root / ".vscode" / "settings.json"
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"{settings_path} is not plain JSON (comments?); add this by hand:")
            print(json.dumps({"yaml.schemas": {f"./{args.schema}": args.glob}}, indent=2))
            return 1
    schemas = settings.setdefault("yaml.schemas", {})
    schemas[f"./{Path(args.schema).as_posix()}"] = args.glob
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {schema_path}")
    print(f"Updated {settings_path} (yaml.schemas -> {', '.join(args.glob)})")
    print("Needs the 'YAML' extension by Red Hat (redhat.vscode-yaml).")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from phoenx.config_gui import main as gui_main
    except ImportError as exc:  # tkinter missing from this Python
        print(f"The config editor needs tkinter, which this Python lacks ({exc}).\n"
              "On Windows/macOS, reinstall Python from python.org with Tcl/Tk enabled; "
              "on Debian/Ubuntu: sudo apt install python3-tk", file=sys.stderr)
        return 1
    return gui_main(args.file)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``phoenx-config`` command."""
    parser = argparse.ArgumentParser(
        prog="phoenx-config", description="PhoenX config schema and validation tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("schema", help="Write the config JSON Schema")
    p.add_argument("-o", "--output", default=DEFAULT_SCHEMA_NAME,
                   help=f"Output file, or '-' for stdout (default: {DEFAULT_SCHEMA_NAME})")
    p.set_defaults(func=_cmd_schema)

    p = sub.add_parser("validate", help="Check YAML configs without building anything")
    p.add_argument("files", nargs="+", help="Config files to check")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("vscode", help="Enable YAML autocomplete in VS Code for a project")
    p.add_argument("--root", default=".", help="Project folder (default: current folder)")
    p.add_argument("--schema", default=DEFAULT_SCHEMA_NAME,
                   help="Schema path, relative to --root")
    p.add_argument("--glob", nargs="+", default=DEFAULT_GLOBS,
                   help=f"Config file globs (default: {' '.join(DEFAULT_GLOBS)})")
    p.set_defaults(func=_cmd_vscode)

    p = sub.add_parser("gui", help="Open the config editor window")
    p.add_argument("file", nargs="?", help="Config to open (optional)")
    p.set_defaults(func=_cmd_gui)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
