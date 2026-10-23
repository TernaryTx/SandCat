from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from sandcat import __version__


def setup_tempfile_logging(log_prefix: str = 'temp_log') -> str:
    # Create a temp file for logging
    temp_log = tempfile.NamedTemporaryFile(
        delete=False, suffix='.log', prefix=log_prefix,
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(temp_log.name),
            logging.StreamHandler(),
        ],
    )
    commit_hash, branch = get_git_info()
    logging.info(f"SandCat version: {__version__}")
    logging.info(f"Git commit hash: {commit_hash}")
    logging.info(f"Git branch: {branch}")
    return str(Path(temp_log.name))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def get_git_info(script_path: str = __file__) -> tuple[str, str]:
    try:
        # Get the directory of the script
        script_dir = Path(script_path).resolve().parent
        # Run the Git command in that directory
        line = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=format:"%H on %d"'],
            cwd=script_dir,
            stderr=subprocess.DEVNULL,
        )
        items = line.decode('utf-8').strip().split()
        commit_hash, branch = items[0][1:], items[4][:-1]
        return commit_hash, branch
    except Exception:
        return 'Unknown', 'Unknown'
