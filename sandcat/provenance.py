from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any

from sandcat import __version__


OUTPUT_PROVENANCE_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_VERSION_DISTRIBUTIONS = (
    'numpy',
    'openmm',
    'openmmtools',
    'mdtraj',
    'netCDF4',
    'mpi4py',
)


def _utc_timestamp(timestamp: float | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""
    if timestamp is None:
        value = datetime.now(timezone.utc)
    else:
        value = datetime.fromtimestamp(timestamp, timezone.utc)
    return value.isoformat()


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        while chunk := input_file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_input_file(role: str, input_path: str | Path, index: int) -> dict[str, Any]:
    """Describe and checksum one simulation input without modifying it."""
    path_text = str(input_path)
    record: dict[str, Any] = {'role': role, 'path': path_text}
    if index > 0:
        record['index'] = index
    if '://' in path_text:
        record.update({'external': True, 'exists': False})
        return record

    path = Path(input_path).expanduser()
    record['resolved_path'] = str(path.resolve())
    if not path.is_file():
        record['exists'] = False
        return record
    file_stat = path.stat()
    record.update({
        'exists': True,
        'size_bytes': file_stat.st_size,
        'modified_at_utc': _utc_timestamp(file_stat.st_mtime),
        'sha256': _sha256_file(path),
    })
    return record


def _config_paths(argv: Sequence[str]) -> list[str]:
    """Extract config-file arguments from a SandCat command line."""
    paths = []
    for argument_index, argument in enumerate(argv):
        if argument in ('-c', '--config') and argument_index + 1 < len(argv):
            paths.append(argv[argument_index + 1])
        elif argument.startswith('--config='):
            paths.append(argument.split('=', 1)[1])
    return paths


def _input_records(
        input_paths: Mapping[str, str | Path | Sequence[str | Path] | None],
        argv: Sequence[str],
) -> list[dict[str, Any]]:
    """Return ordered provenance records for user and config inputs."""
    records = []
    configured_paths = dict(input_paths)
    config_paths = _config_paths(argv)
    if config_paths and 'config' not in configured_paths:
        configured_paths['config'] = config_paths
    for role, paths in configured_paths.items():
        if paths is None:
            continue
        if isinstance(paths, (str, Path)):
            role_paths: Sequence[str | Path] = [paths]
        else:
            role_paths = paths
        for index, input_path in enumerate(role_paths):
            records.append(_describe_input_file(role, input_path, index))
    return records


def _run_git(repo_root: Path, arguments: Sequence[str]) -> str | None:
    """Run a bounded read-only Git query and return stripped stdout."""
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sandcat_source_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Return package and Git identity for this SandCat implementation."""
    root = repo_root or Path(__file__).resolve().parents[1]
    environment_commit = os.environ.get('SANDCAT_GIT_COMMIT')
    git_commit = environment_commit or _run_git(root, ['rev-parse', 'HEAD'])
    return {
        'version': __version__,
        'git_commit': git_commit,
        'git_branch': _run_git(root, ['rev-parse', '--abbrev-ref', 'HEAD']),
        'git_commit_date': _run_git(root, ['show', '-s', '--format=%cI', 'HEAD']),
        'identity_source': 'environment' if environment_commit else 'git',
    }


def _distribution_versions() -> dict[str, str | None]:
    """Return versions of the scientific packages that affect a run."""
    versions: dict[str, str | None] = {}
    for distribution_name in _VERSION_DISTRIBUTIONS:
        try:
            versions[distribution_name] = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            versions[distribution_name] = None
    return versions


def _json_safe(value: Any) -> Any:
    """Convert common scientific and path values into JSON-safe structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'tolist'):
        return _json_safe(value.tolist())
    return str(value)


def build_output_provenance(
        workflow: str,
        parameters: Mapping[str, Any],
        input_paths: Mapping[str, str | Path | Sequence[str | Path] | None],
        extra: Mapping[str, Any] | None = None,
        argv: Sequence[str] | None = None,
        repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build reproducible, versioned provenance for one fresh simulation.

    Args:
        workflow: Stable workflow identifier such as ``swish-x``.
        parameters: Fully resolved simulation and setup parameters.
        input_paths: Input-file roles mapped to one path or a sequence of paths.
        extra: Existing Hamiltonian or lifecycle metadata retained at top level.
        argv: Command arguments to record. Defaults to the current process arguments.
        repo_root: Optional SandCat checkout used for Git identity tests or overrides.

    Returns:
        JSON-serializable provenance dictionary.
    """
    command_argv = list(sys.argv if argv is None else argv)
    provenance = dict(_json_safe(extra or {}))
    provenance.update({
        'schema_version': OUTPUT_PROVENANCE_SCHEMA_VERSION,
        'created_at_utc': _utc_timestamp(),
        'workflow': workflow,
        'sandcat': _sandcat_source_identity(repo_root),
        'command': {
            'argv': command_argv,
            'rendered': shlex.join(command_argv),
            'working_directory': str(Path.cwd()),
        },
        'parameters': _json_safe(parameters),
        'inputs': _input_records(input_paths, command_argv),
        'software': {
            'python': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'platform': platform.platform(),
            'distributions': _distribution_versions(),
        },
    })
    return provenance
