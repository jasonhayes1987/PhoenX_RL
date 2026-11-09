"""Build and describe the commands the Run tab launches.

``build_command`` turns form fields into the argument list of a PhoenX
command. ``command_text`` joins that list into one line for this platform.
``describe_run`` writes the one-paragraph summary shown beside a saved
agent folder.
"""


def build_command(mode: str, values: dict[str, str]) -> list[str]:
    """Turn Run-tab field values into the argv of a PhoenX command.

    Args:
        mode: ``"train"``, ``"resume"``, ``"test"``, or ``"sweep"``.
        values: Field text, keyed by the Run tab's field names.

    Returns:
        The argv list, starting with the Python interpreter.

    Raises:
        ValueError: The first required field is missing, or ``mode`` is unknown.
    """
    raise NotImplementedError


def describe_run(info: dict) -> str:
    """Summarize a saved run in one paragraph for the Resume and Test forms.

    Args:
        info: The dict printed by ``python -m phoenx.run_info``, or one with
            an ``error`` key.

    Returns:
        The paragraph shown next to the agent folder.
    """
    raise NotImplementedError


def command_text(argv: list[str]) -> str:
    """Join an argv list into the command line the Run tab displays.

    Args:
        argv: The list ``build_command`` returns.

    Returns:
        One command line for the current platform.
    """
    raise NotImplementedError
