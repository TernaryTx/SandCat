from __future__ import annotations

import hashlib
import json

import numpy as np

from sandcat.provenance import (
    OUTPUT_PROVENANCE_SCHEMA_VERSION,
    build_output_provenance,
)


def test_build_output_provenance_records_reproducibility_metadata(
        tmp_path, monkeypatch,
) -> None:
    input_path = tmp_path / 'protein.pdb'
    input_contents = b'ATOM test input\n'
    input_path.write_bytes(input_contents)
    config_path = tmp_path / 'run.cfg'
    config_path.write_text('[run]\ntime_step_fs = 4\n')
    monkeypatch.setenv('SANDCAT_GIT_COMMIT', '0123456789abcdef')

    provenance = build_output_provenance(
        workflow='swish-x',
        parameters={
            'time_step_fs': 4,
            'scale_factors': np.array([1.0, 1.35]),
        },
        input_paths={'input_protein': input_path},
        extra={
            'revision': 'benzene-ca-ca-v2',
            'opes_enabled': True,
            'swish_x_lifecycle': 'post_post_v1',
        },
        argv=['sandcat-run', '--config', str(config_path)],
        repo_root=tmp_path,
    )

    assert provenance['schema_version'] == OUTPUT_PROVENANCE_SCHEMA_VERSION
    assert provenance['created_at_utc'].endswith('+00:00')
    assert provenance['workflow'] == 'swish-x'
    assert provenance['sandcat']['git_commit'] == '0123456789abcdef'
    assert provenance['sandcat']['identity_source'] == 'environment'
    assert provenance['command']['argv'] == [
        'sandcat-run', '--config', str(config_path),
    ]
    assert provenance['command']['rendered'] == (
        f"sandcat-run --config {config_path}"
    )
    assert provenance['parameters']['scale_factors'] == [1.0, 1.35]

    inputs_by_role = {record['role']: record for record in provenance['inputs']}
    assert inputs_by_role['input_protein']['sha256'] == hashlib.sha256(
        input_contents,
    ).hexdigest()
    assert inputs_by_role['input_protein']['size_bytes'] == len(input_contents)
    assert inputs_by_role['config']['sha256'] == hashlib.sha256(
        config_path.read_bytes(),
    ).hexdigest()
    assert provenance['software']['python']
    assert 'openmm' in provenance['software']['distributions']
    json.dumps(provenance)


def test_build_output_provenance_records_missing_and_external_inputs(tmp_path) -> None:
    provenance = build_output_provenance(
        workflow='swish',
        parameters={},
        input_paths={
            'missing': tmp_path / 'missing.pdb',
            'external': 'storage://example/input.pdb',
        },
        argv=['sandcat-run'],
        repo_root=tmp_path,
    )

    inputs_by_role = {record['role']: record for record in provenance['inputs']}
    assert inputs_by_role['missing']['exists'] is False
    assert inputs_by_role['external'] == {
        'role': 'external',
        'path': 'storage://example/input.pdb',
        'external': True,
        'exists': False,
    }
