from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from sandcat.analysis import build_density_maps, experiments, run_dynamics_all
from sandcat.analysis import run_prospective_analysis_pipeline as pipeline


def test_analysis_modules_import_without_local_paths_or_output_creation(tmp_path: Path) -> None:
    """Public modules import without a data tree or developer-specific paths."""
    modules = [
        'experiments', 'analyse_pocket_dynamics', 'figure_style',
        'literature_numbering', 'make_comparison_pmls', 'make_overlay_pmls',
        'plot_candidate_residues', 'plot_ranking_figure', 'plot_site_residue_figure',
        'pocket_residue_dynamics', 'build_density_maps', 'residue_motion',
        'residue_ranking', 'run_dynamics_all', 'run_prospective_analysis_pipeline',
        'snapshot_frames',
    ]
    script = (
        'import importlib\n'
        f'for name in {modules!r}:\n'
        '    importlib.import_module("sandcat.analysis." + name)\n'
    )
    output = tmp_path / 'not-created'
    result = subprocess.run(
        [sys.executable, '-c', script], check=False, capture_output=True, text=True,
        env=dict(os.environ, SANDCAT_PAPER_DIR=str(tmp_path / 'inputs'),
                 SANDCAT_ANALYSIS_DIR=str(output), MPLCONFIGDIR=str(tmp_path / 'mpl')),
    )
    assert result.returncode == 0, result.stderr
    assert not output.exists()


def test_pipeline_passes_absolute_roots_and_preserves_stage_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Child processes use portable modules and explicit data roots."""
    paper = tmp_path / 'paper'
    (paper / 'september2026').mkdir(parents=True)
    (paper / 'september2026_ref_pockets').mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['pipeline', 'output', '--paper-dir', 'paper', '--skip-density'])
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    pipeline.main()
    expected = [name for name, _ in pipeline.STAGES if name != 'build_density_maps']
    assert [call.args[0][2] for call in run.call_args_list] == [
        f'sandcat.analysis.{name}' for name in expected
    ]
    for call in run.call_args_list:
        assert call.args[0][:2] == [sys.executable, '-m']
        assert call.kwargs['env']['SANDCAT_PAPER_DIR'] == str(paper)
        assert call.kwargs['env']['SANDCAT_ANALYSIS_DIR'] == str(tmp_path / 'output')


def test_pipeline_stops_on_failed_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed subprocess cannot be treated as a successful prerequisite."""
    monkeypatch.setattr(pipeline.subprocess, 'run', Mock(return_value=subprocess.CompletedProcess([], 1)))
    with pytest.raises(SystemExit, match='failed with exit code 1'):
        pipeline.run_stage('build_density_maps', tmp_path, {})


def test_density_stage_reports_target_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even after processing independent targets, failures must reach the driver."""
    monkeypatch.setattr(sys, 'argv', ['density', str(tmp_path / 'output'), 'TEST'])
    monkeypatch.setattr(build_density_maps, 'SIMULATION_INPUTS', {'TEST': ('md', 'swish', 'apo', 1)})
    monkeypatch.setattr(build_density_maps, 'unbiased_pairs', Mock(return_value=[('top', 'dcd')]))
    monkeypatch.setattr(build_density_maps, 'run_density', Mock(side_effect=RuntimeError('invalid input')))
    monkeypatch.setattr(build_density_maps.logging, 'basicConfig', Mock())
    with pytest.raises(SystemExit, match='1 of 1 targets failed'):
        build_density_maps.main()


def test_dynamics_stage_reports_target_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale output files do not turn a failed dynamics stage into success."""
    density = tmp_path / 'TEST' / 'density'
    density.mkdir(parents=True)
    (density / run_dynamics_all.HOTSPOTS_FILE).touch()
    monkeypatch.setattr(run_dynamics_all, 'ANALYSIS', tmp_path)
    monkeypatch.setattr(run_dynamics_all, 'TARGETS', ['TEST'])
    monkeypatch.setattr(run_dynamics_all, 'target_dir', lambda target: tmp_path / target)
    monkeypatch.setattr(run_dynamics_all, 'SIMULATION_INPUTS', {'TEST': ('md', 'swish', 'apo', 1)})
    monkeypatch.setattr(run_dynamics_all, 'bundled_apo', lambda target: tmp_path / 'apo.pdb')
    monkeypatch.setattr(run_dynamics_all, 'run', Mock(side_effect=RuntimeError('invalid input')))
    with pytest.raises(SystemExit, match='failed'):
        run_dynamics_all.main()


def test_pipeline_rejects_missing_benchmark_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing public datasets are diagnosed before any stage is launched."""
    monkeypatch.setattr(sys, 'argv', ['pipeline', str(tmp_path / 'output'), '--paper-dir', str(tmp_path)])
    run = Mock()
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    with pytest.raises(SystemExit):
        pipeline.main()
    run.assert_not_called()
    assert not (tmp_path / 'output').exists()


def test_external_session_input_is_bundled_and_relative(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """General callers may supply an apo outside the portable output tree."""
    root = tmp_path / 'analysis'
    source = tmp_path / 'apo.pdb'
    source.write_text('END\n')
    monkeypatch.setenv('SANDCAT_ANALYSIS_DIR', str(root))
    bundled = experiments.bundle_into_analysis(source)
    assert bundled == root / 'inputs' / 'apo.pdb'
    assert bundled.read_bytes() == source.read_bytes()
    assert experiments.analysis_relative(bundled) == 'inputs/apo.pdb'
    assert experiments.bundle_into_analysis(bundled) == bundled
