from __future__ import annotations

import argparse
import re
import sys
from argparse import ArgumentParser

# Plain-text config file support for the sandcat CLIs.

CONFIG_COMMENT_CHAR = '#'
_TRUTHY = ('1', 'true', 'yes', 'y', 'on')
# key = leading run of non-space/non-delimiter chars; an optional single '='
# or ':' delimiter; the rest (with internal spaces preserved) is the value.
# This keeps ':' inside URL-like values intact.
_ENTRY_RE = re.compile(r'^([^\s=:]+)\s*[=:]?\s*(.*)$')


def add_config_arg(parser: ArgumentParser) -> None:
    """Add the shared ``-c/--config`` option to an ArgumentParser."""
    parser.add_argument(
        '-c', '--config', type=str, default=None,
        help='path to a plain-text config file ("key = value" per line) supplying '
             'argument values. Values passed on the command line override the file.',
    )


def parse_args_with_config(
        parser: ArgumentParser, argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse args, layering a ``--config`` file underneath the command line.

    Config file values are turned into CLI tokens and placed before the real
    argv, so anything also passed on the command line wins. Required arguments
    may be satisfied entirely by the config file.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _extract_config_path(argv)
    if config_path is not None:
        entries = read_config_file(config_path)
        config_argv = config_entries_to_argv(parser, entries)
        argv = config_argv + argv
    return parser.parse_args(argv)


def _extract_config_path(argv: list[str]) -> str | None:
    """Find the ``--config`` value in argv without tripping on the real parser's required args."""
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument('-c', '--config', type=str, default=None)
    known, _ = pre_parser.parse_known_args(argv)
    return known.config


def read_config_file(path: str) -> dict[str, str]:
    """Parse a plain-text "key = value" config file into a dict.

    Blank lines and lines starting with ``#`` are ignored. Each entry is
    ``key = value``, ``key: value`` or ``key value``. Keys may be written with
    or without leading dashes. Values keep internal spaces, so multi-value
    options (e.g. ``pocket_centre = 2.3 1.8 2.9``) work, and ``:`` inside a
    value (e.g. ``output_directory = storage://example/out``) is preserved.
    """
    entries: dict[str, str] = {}
    with open(path) as config_fh:
        for line_number, raw_line in enumerate(config_fh, start=1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(CONFIG_COMMENT_CHAR):
                continue
            match = _ENTRY_RE.match(stripped)
            key = match.group(1).lstrip('-').strip() if match else ''
            if not key:
                raise ValueError(
                    f'{path}:{line_number}: could not parse config line: {raw_line.rstrip()!r}',
                )
            entries[key] = match.group(2).strip()
    return entries


def config_entries_to_argv(
        parser: ArgumentParser, entries: dict[str, str],
) -> list[str]:
    """Turn parsed config entries into a list of CLI tokens for ``parser``.

    Boolean flags (store_true/store_false) are emitted only when their value is
    truthy; valued options are emitted as ``--option value [value ...]``. Keys
    that do not match any option raise a ValueError so typos fail loudly rather
    than being silently ignored.
    """
    lookup = _build_action_lookup(parser)
    argv: list[str] = []
    for key, value in entries.items():
        action = lookup.get(key)
        if action is None:
            raise ValueError(
                f"Unknown config key '{key}': not an option of "
                f"'{parser.prog}'. Use the long option name (without dashes).",
            )
        if action.dest == 'config':
            continue
        if not action.option_strings:
            raise ValueError(
                f"Config key '{key}' maps to a positional argument, which "
                f"config files do not support.",
            )
        option = action.option_strings[-1]
        if _is_flag(action):
            if _is_truthy(value):
                argv.append(option)
        else:
            argv.append(option)
            if value != '':
                argv.extend(value.split())
    return argv


def _build_action_lookup(parser: ArgumentParser) -> dict[str, argparse.Action]:
    """Map every option string and dest of a parser to its argparse Action."""
    lookup: dict[str, argparse.Action] = {}
    for action in parser._actions:
        if action.dest in (None, 'help'):
            continue
        lookup[action.dest] = action
        for option_string in action.option_strings:
            lookup[option_string.lstrip('-')] = action
    return lookup


def _is_flag(action: argparse.Action) -> bool:
    """Return True if the action takes no value (store_true/store_false/store_const)."""
    return action.nargs == 0 or isinstance(
        action,
        (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction),
    )


def _is_truthy(value: str) -> bool:
    """A flag entry is on when it is bare (no value) or set to a truthy word."""
    return value == '' or value.strip().lower() in _TRUTHY
