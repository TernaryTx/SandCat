from __future__ import annotations

from argparse import ArgumentParser

import pytest

from sandcat.config import (
    add_config_arg,
    config_entries_to_argv,
    parse_args_with_config,
    read_config_file,
)


def _make_parser():
    """A parser exercising the option shapes the real CLIs use."""
    arg_parser = ArgumentParser(prog='demo')
    arg_parser.add_argument('-d', '--output_directory', required=True)
    arg_parser.add_argument('--n_replicas', type=int, default=6)
    arg_parser.add_argument('--scale_max', type=float, default=1.35)
    arg_parser.add_argument('--pocket_centre', type=float, nargs=3, default=None)
    arg_parser.add_argument('--verbose', action='store_true', default=False)
    arg_parser.add_argument('--no_spot', dest='use_spot', action='store_false')
    arg_parser.set_defaults(use_spot=True)
    arg_parser.add_argument('--exchange_interval_ps', type=float, default=None)
    add_config_arg(arg_parser)
    return arg_parser


def _write(tmp_path, text):
    path = tmp_path / 'run.cfg'
    path.write_text(text)
    return str(path)


def test_read_config_file_separators_and_comments(tmp_path):
    """'=', ':' and bare-space separators all parse; comments and blanks are skipped."""
    path = _write(tmp_path, '\n'.join([
        '# a comment',
        '',
        'output_directory = storage://example/out/',  # ':' inside value preserved
        'n_replicas: 8',
        'scale_max 1.5',
        '--verbose',
    ]))
    entries = read_config_file(path)
    assert entries == {
        'output_directory': 'storage://example/out/',
        'n_replicas': '8',
        'scale_max': '1.5',
        'verbose': '',
    }


def test_config_only_satisfies_required_and_types(tmp_path):
    """A config file alone can supply required args, multi-value args and flags."""
    path = _write(tmp_path, '\n'.join([
        'output_directory = storage://example/out/',
        'n_replicas = 8',
        'scale_max = 1.5',
        'pocket_centre = 2.3 1.8 2.9',
        'verbose = true',
        'no_spot = true',
        'exchange_interval_ps = 20',
    ]))
    args = parse_args_with_config(_make_parser(), ['--config', path])
    assert args.output_directory == 'storage://example/out/'
    assert args.n_replicas == 8
    assert args.scale_max == pytest.approx(1.5)
    assert args.pocket_centre == [2.3, 1.8, 2.9]
    assert args.verbose is True
    assert args.use_spot is False  # no_spot=true flips the store_false dest
    assert args.exchange_interval_ps == pytest.approx(20.0)


def test_command_line_overrides_config(tmp_path):
    """Explicit command-line values win over the config file."""
    path = _write(
        tmp_path,
        'output_directory = storage://example/out/\nn_replicas = 8\n',
    )
    args = parse_args_with_config(
        _make_parser(), ['--config', path, '--n_replicas', '4'],
    )
    assert args.n_replicas == 4


def test_no_config_uses_defaults():
    """Without a config file the parser behaves exactly as before."""
    args = parse_args_with_config(_make_parser(), ['-d', 'storage://example/'])
    assert args.n_replicas == 6
    assert args.use_spot is True
    assert args.verbose is False


def test_unknown_key_raises(tmp_path):
    """A typo'd key fails loudly rather than being silently ignored."""
    path = _write(
        tmp_path,
        'output_directory = storage://example/\nnot_an_option = 5\n',
    )
    with pytest.raises(ValueError, match='Unknown config key'):
        parse_args_with_config(_make_parser(), ['--config', path])


def test_flag_only_emitted_when_truthy():
    """store_true/store_false flags are emitted only for truthy config values."""
    parser = _make_parser()
    assert config_entries_to_argv(parser, {'verbose': 'false'}) == []
    assert config_entries_to_argv(parser, {'verbose': 'true'}) == ['--verbose']
    assert config_entries_to_argv(parser, {'verbose': ''}) == ['--verbose']
    assert config_entries_to_argv(parser, {'no_spot': 'yes'}) == ['--no_spot']
