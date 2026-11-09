"""Return the YAML examples that ship with PhoenX.

Each call reads one installed folder, ``configs`` or ``sweeps``, and maps
every file's path relative to that folder to its YAML text. The mapping is
empty when the folder is not installed.
"""

from importlib import resources


def bundled_examples(folder: str = "configs") -> dict[str, str]:
    """Return the YAML examples shipped with PhoenX, keyed by relative path.

    Args:
        folder: ``"configs"`` or ``"sweeps"``.

    Returns:
        Mapping of relative name to YAML text. Empty when that folder is not installed.
    """
    out: dict[str, str] = {}
    try:
        root = resources.files("phoenx.examples").joinpath(folder)
    except ModuleNotFoundError:
        return out

    def walk(node, prefix):
        for entry in sorted(node.iterdir(), key=lambda e: e.name):
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_dir():
                walk(entry, rel)
            elif entry.name.endswith(".yml"):
                out[rel] = entry.read_text(encoding="utf-8")

    if root.is_dir():
        walk(root, "")
    return out
