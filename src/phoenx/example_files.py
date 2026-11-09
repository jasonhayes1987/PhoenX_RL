"""Return the YAML examples that ship with PhoenX.

Each call reads one installed folder, ``configs`` or ``sweeps``, and maps
every file's path relative to that folder to its YAML text. The mapping is
empty when the folder is not installed.
"""


def bundled_examples(folder: str = "configs") -> dict[str, str]:
    """Return the YAML examples shipped with PhoenX, keyed by relative path.

    Args:
        folder: ``"configs"`` or ``"sweeps"``.

    Returns:
        Mapping of relative name to YAML text. Empty when that folder is not installed.
    """
    raise NotImplementedError
