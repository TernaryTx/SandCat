from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest

import sandcat.md_utils as md_utils_module
from sandcat.md_utils import (
    SWISH_ITER_LENGTH,
    OpesReplicaExchangeSampler,
    _build_swish_correction_force,
    _validate_replica_particle_counts,
    add_contact_map_upper_wall,
    add_probe_repulsion,
    _get_swish_interaction_indices,
    _select_contact_pairs,
    backup_checkpoint,
    can_restart_swish_production,
    resolve_iter_length,
    validate_output_provenance,
    write_output_provenance,
    write_probe_hamiltonian_provenance,
)
from sandcat.opes_expanded import (
    PLUMED_GEOMETRIC_BIAS_FORCE_GROUP,
    OpesExpanded,
    _read_contact_map,
)
from sandcat.provenance import OUTPUT_PROVENANCE_SCHEMA_VERSION


def _selection_topology_and_forcefield() -> tuple[app.Topology, object]:
    topology = app.Topology()
    chain = topology.addChain()
    protein = topology.addResidue('ASP', chain)
    topology.addAtom('CA', app.element.carbon, protein)
    topology.addAtom('CB', app.element.carbon, protein)
    topology.addAtom('C', app.element.carbon, protein)
    water = topology.addResidue('HOH', chain)
    topology.addAtom('O', app.element.oxygen, water)
    probe = topology.addResidue('PRB', chain)
    topology.addAtom('C1', app.element.carbon, probe)

    templates = [
        SimpleNamespace(atoms=[
            SimpleNamespace(name='CA', type='DD'),
            SimpleNamespace(name='CB', type='CT'),
            SimpleNamespace(name='C', type='C'),
        ]),
        SimpleNamespace(atoms=[SimpleNamespace(name='O', type='OW')]),
        SimpleNamespace(atoms=[SimpleNamespace(name='C1', type='CT')]),
    ]
    forcefield = SimpleNamespace(getMatchingTemplates=lambda _: templates)
    return topology, forcefield


def test_ff_type_swish_selection_matches_explicit_cross_terms() -> None:
    topology, forcefield = _selection_topology_and_forcefield()
    water_indices, protein_indices = _get_swish_interaction_indices(
        topology, 'PRB',
        forcefield_name='desamber', forcefield=forcefield,
    )
    assert water_indices == [3]
    assert protein_indices == [1]


def test_ff19_type_swish_selection_matches_explicit_cross_terms() -> None:
    topology, forcefield = _selection_topology_and_forcefield()
    forcefield.getMatchingTemplates(topology)[0].atoms[1].type = 'protein-CT'
    water_indices, protein_indices = _get_swish_interaction_indices(
        topology, 'PRB',
        forcefield_name='ff19', forcefield=forcefield,
    )
    assert water_indices == [3]
    assert protein_indices == [1]


def test_ff14_type_swish_selection_matches_explicit_cross_terms() -> None:
    topology, forcefield = _selection_topology_and_forcefield()
    # CB atom has type 'CT' which is in the ff14 apolar set; CA has type 'DD' which is not.
    water_indices, protein_indices = _get_swish_interaction_indices(
        topology, 'PRB',
        forcefield_name='ff14', forcefield=forcefield,
    )
    assert water_indices == [3]
    assert protein_indices == [1]


def test_ff_type_swish_selection_raises_for_missing_forcefield() -> None:
    topology, forcefield = _selection_topology_and_forcefield()
    with pytest.raises(ValueError, match='No versioned'):
        _get_swish_interaction_indices(
            topology, 'PRB',
            forcefield_name='unknown_ff', forcefield=forcefield,
        )
    with pytest.raises(ValueError, match='no legacy fallback'):
        _get_swish_interaction_indices(
            topology, 'PRB',
            forcefield_name='desamber', forcefield=None,
        )


def test_probe_repulsion_corrects_probe_carbon_14_exception_without_changing_charge():
    topology = app.Topology()
    chain = topology.addChain()
    probe = topology.addResidue('PRB', chain)
    probe_atoms = [
        topology.addAtom(f'C{atom_number}', app.element.carbon, probe)
        for atom_number in range(4)
    ]
    for first_atom, second_atom in zip(probe_atoms, probe_atoms[1:]):
        topology.addBond(first_atom, second_atom)
    modeller = app.Modeller(topology, unit.Quantity(np.zeros((4, 3)), unit.nanometer))

    system = mm.System()
    for _ in probe_atoms:
        system.addParticle(12 * unit.amu)
    nonbonded = mm.NonbondedForce()
    for _ in probe_atoms:
        nonbonded.addParticle(0.1, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    nonbonded.addException(0, 1, 0.01, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    nonbonded.addException(1, 2, 0.02, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    nonbonded.addException(2, 3, 0.03, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    nonbonded.addException(0, 3, 0.04, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    system.addForce(nonbonded)

    provenance = add_probe_repulsion(system, modeller)

    assert provenance['revision'] == md_utils_module.PROBE_HAMILTONIAN_REVISION
    assert provenance['modified_probe_14_exceptions'] == 1
    _, _, charge_product, sigma, epsilon = nonbonded.getExceptionParameters(3)
    assert charge_product.value_in_unit(unit.elementary_charge ** 2) == 0.04
    assert sigma == 0.8 * unit.nanometer
    assert epsilon == 0.00075 * unit.kilojoules_per_mole


def test_probe_hamiltonian_provenance_is_written(tmp_path):
    output_prefix = str(tmp_path / 'swish_output')
    write_probe_hamiltonian_provenance(
        output_prefix,
        {'revision': 'benzene-ca-ca-v2', 'modified_probe_14_exceptions': 645},
    )
    assert (tmp_path / 'swish_output_probe_hamiltonian.json').read_text() == (
        '{\n  "modified_probe_14_exceptions": 645,\n'
        '  "revision": "benzene-ca-ca-v2"\n}\n'
    )
    assert not (tmp_path / 'swish_output_probe_hamiltonian.json.tmp').exists()


def test_probe_repulsion_suppresses_intramolecular_pairs_without_extra_exclusions() -> None:
    topology = app.Topology()
    chain = topology.addChain()
    residues = [topology.addResidue('PRB', chain) for _ in range(2)]
    atoms = [
        [topology.addAtom(f'C{number}', app.element.carbon, residue) for number in range(5)]
        for residue in residues
    ]
    modeller = app.Modeller(topology, unit.Quantity(np.zeros((10, 3)), unit.nanometer))
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    for _ in range(10):
        system.addParticle(12 * unit.amu)
        nonbonded.addParticle(0.0, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    system.addForce(nonbonded)

    provenance = add_probe_repulsion(system, modeller, correct_probe_14_exceptions=False)

    repulsion_force = system.getForce(1)
    assert repulsion_force.getNumExclusions() == 0
    first_group = repulsion_force.getParticleParameters(atoms[0][0].index)[2]
    assert all(repulsion_force.getParticleParameters(atom.index)[2] == first_group for atom in atoms[0])
    assert all(repulsion_force.getParticleParameters(atom.index)[2] != first_group for atom in atoms[1])
    assert provenance['intramolecular_probe_exclusions'] == 20


def test_probe_repulsion_accepts_multiple_ligand_residue_names() -> None:
    topology = app.Topology()
    chain = topology.addChain()
    atoms = [
        topology.addAtom('C1', app.element.carbon, topology.addResidue(name, chain))
        for name in ('P00', 'P01')
    ]
    modeller = app.Modeller(topology, unit.Quantity(np.zeros((2, 3)), unit.nanometer))
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    for _ in atoms:
        system.addParticle(12 * unit.amu)
        nonbonded.addParticle(0.0, 0.3 * unit.nanometer, 0.2 * unit.kilojoules_per_mole)
    system.addForce(nonbonded)

    provenance = add_probe_repulsion(
        system, modeller, probe_resname=['P00', 'P01'],
        correct_probe_14_exceptions=False,
    )

    assert system.getForce(1).getNumInteractionGroups() == 1
    assert provenance['probe_resnames'] == ['P00', 'P01']


def test_validate_replica_particle_counts_reports_mismatched_rank_system() -> None:
    system = mm.System()
    for _ in range(3):
        system.addParticle(1 * unit.amu)
    positions = [
        unit.Quantity(np.zeros((3, 3)), unit.nanometer),
        unit.Quantity(np.zeros((2, 3)), unit.nanometer),
    ]
    box_vectors = unit.Quantity(np.eye(3), unit.nanometer)

    with pytest.raises(ValueError, match=r'expected 3, got \[3, 2\].*MPI ranks'):
        _validate_replica_particle_counts(system, positions, [box_vectors, box_vectors], 2)


def test_output_provenance_round_trip_and_fail_closed(tmp_path):
    output_prefix = str(tmp_path / 'swish_x_output')
    provenance = {
        'revision': md_utils_module.PROBE_HAMILTONIAN_REVISION,
        'opes_enabled': True,
        'swish_x_lifecycle': md_utils_module.SWISH_X_LIFECYCLE_REVISION,
    }
    write_output_provenance(output_prefix, provenance)
    assert validate_output_provenance(output_prefix) == provenance

    provenance['schema_version'] = OUTPUT_PROVENANCE_SCHEMA_VERSION
    write_output_provenance(output_prefix, provenance)
    assert validate_output_provenance(output_prefix) == provenance

    provenance['schema_version'] = OUTPUT_PROVENANCE_SCHEMA_VERSION + 1
    write_output_provenance(output_prefix, provenance)
    with pytest.raises(RuntimeError, match='Unsupported output provenance schema'):
        validate_output_provenance(output_prefix)

    provenance['schema_version'] = OUTPUT_PROVENANCE_SCHEMA_VERSION
    provenance['swish_x_lifecycle'] = 'legacy'
    write_output_provenance(output_prefix, provenance)
    with pytest.raises(RuntimeError, match='lifecycle revision mismatch'):
        validate_output_provenance(output_prefix)

    with pytest.raises(RuntimeError, match='Missing Hamiltonian provenance'):
        validate_output_provenance(str(tmp_path / 'missing'))


def test_select_contact_pairs_filters_by_seq_separation():
    """Pairs with sequence separation below the minimum threshold are excluded."""
    # Atom 0 → res 0, atom 1 → res 1, atom 2 → res 2 (linear)
    atom_indices = [10, 20, 30]
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.1, 0.0, 0.0],
                          [0.2, 0.0, 0.0]])
    atom_to_res = {10: 0, 20: 1, 30: 2}
    # Sequence separation 2 is too small (atoms 0-2 are 2 res apart, 0-1 is 1)
    pairs = _select_contact_pairs(
        atom_indices, positions, atom_to_res,
        min_seq_separation=3, contact_cutoff_nm=1.0,
    )
    assert pairs == []


def test_select_contact_pairs_filters_by_distance():
    """Pairs farther apart than the contact cutoff are excluded."""
    atom_indices = [10, 20]
    positions = np.array([[0.0, 0.0, 0.0],
                          [10.0, 0.0, 0.0]])  # 10 nm apart, way past cutoff
    atom_to_res = {10: 0, 20: 5}
    pairs = _select_contact_pairs(
        atom_indices, positions, atom_to_res,
        min_seq_separation=3, contact_cutoff_nm=0.6,
    )
    assert pairs == []


def test_select_contact_pairs_reference_matches_switching_function():
    """A contact within cutoff gets the correct switching-function reference value."""
    # 0.4 nm apart, well within 0.6 cutoff and >= 5 residues apart.
    atom_indices = [10, 20]
    positions = np.array([[0.0, 0.0, 0.0],
                          [0.4, 0.0, 0.0]])
    atom_to_res = {10: 0, 20: 8}
    pairs = _select_contact_pairs(
        atom_indices, positions, atom_to_res,
        min_seq_separation=5, contact_cutoff_nm=0.6, r0_nm=0.55,
    )
    assert len(pairs) == 1
    atom_i, atom_j, ref = pairs[0]
    assert (atom_i, atom_j) == (10, 20)
    expected = 1.0 / (1.0 + (0.4 / 0.55) ** 6)
    assert ref == pytest.approx(expected, rel=1e-12)


def test_contact_map_upper_wall_is_available_to_colvar_reporter(monkeypatch):
    topology = app.Topology()
    chain = topology.addChain()
    first_residue = topology.addResidue('ALA', chain)
    topology.addAtom('CA', app.element.carbon, first_residue)
    second_residue = topology.addResidue('ALA', chain)
    topology.addAtom('CA', app.element.carbon, second_residue)
    positions = unit.Quantity(
        np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        unit.nanometer,
    )
    modeller = app.Modeller(topology, positions)

    system = mm.System()
    system.addParticle(12.0 * unit.amu)
    system.addParticle(12.0 * unit.amu)

    monkeypatch.setattr(
        md_utils_module.mdtraj,
        'compute_dssp',
        lambda trajectory, simplified: np.array([['C', 'C']]),
    )
    monkeypatch.setattr(
        md_utils_module,
        '_select_representative_atoms',
        lambda *args: ([0, 1], [positions[0], positions[1]]),
    )
    monkeypatch.setattr(
        md_utils_module,
        '_select_contact_pairs',
        lambda *args: [(0, 1, 0.5)],
    )

    add_contact_map_upper_wall(system, modeller)

    upper_wall = system.getForce(system.getNumForces() - 1)
    assert upper_wall.getForceGroup() == PLUMED_GEOMETRIC_BIAS_FORCE_GROUP

    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(
        system,
        integrator,
        mm.Platform.getPlatformByName('Reference'),
    )
    context.setPositions(positions)
    assert np.isfinite(_read_contact_map(context))


def test_swish_correction_force_zero_at_lambda_one():
    """delta_lam = lam - 1 → 0 at lam=1.0, so the OW-apolar correction
    must vanish identically when lam=1, regardless of geometry.
    """
    system = mm.System()
    system.setDefaultPeriodicBoxVectors((4.0, 0, 0), (0, 4.0, 0), (0, 0, 4.0))
    for _ in range(2):
        system.addParticle(18.0)
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.2 * unit.nanometer)
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3, 0.5)
    system.addForce(nonbonded)
    eps_vals = np.array([0.5, 0.5])
    sig_vals = np.array([0.3, 0.3])

    correction = _build_swish_correction_force(
        system,
        water_ox_indices=[0],
        apolar_indices=[1],
        eps_vals=eps_vals,
        sig_vals=sig_vals,
        initial_lam=1.0,
    )
    correction.setForceGroup(5)
    system.addForce(correction)

    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName('Reference'))
    context.setPositions([(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)])
    state = context.getState(getEnergy=True, groups={5})
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    assert energy == pytest.approx(0.0, abs=1e-10)


def test_swish_correction_force_scales_lj_at_lambda_above_one():
    """At lam>1 the correction adds (lam-1) * 4*sqrt(eps_i*eps_j) * LJ12-6.
    Compare numerical energy at lam=1.5 to the analytic expectation.
    """
    system = mm.System()
    system.setDefaultPeriodicBoxVectors((4.0, 0, 0), (0, 4.0, 0), (0, 0, 4.0))
    for _ in range(2):
        system.addParticle(18.0)
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(1.2 * unit.nanometer)
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.3, 0.5)
    system.addForce(nonbonded)
    eps_vals = np.array([0.4, 0.9])
    sig_vals = np.array([0.30, 0.34])

    lam = 1.5
    correction = _build_swish_correction_force(
        system,
        water_ox_indices=[0],
        apolar_indices=[1],
        eps_vals=eps_vals,
        sig_vals=sig_vals,
        initial_lam=lam,
    )
    correction.setForceGroup(5)
    system.addForce(correction)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName('Reference'))
    r = 0.4
    context.setPositions([(0.0, 0.0, 0.0), (r, 0.0, 0.0)])
    state = context.getState(getEnergy=True, groups={5})
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

    eps_ij = np.sqrt(eps_vals[0] * eps_vals[1])
    sig_ij = 0.5 * (sig_vals[0] + sig_vals[1])
    expected = (lam - 1.0) * 4.0 * eps_ij * ((sig_ij / r) ** 12 - (sig_ij / r) ** 6)
    assert energy == pytest.approx(expected, rel=1e-6)


def test_swish_iter_length_is_10_picoseconds():
    """SWISH_ITER_LENGTH equals 10 ps, matching the GROMACS nstswapcoords default."""
    # a regression here would silently change every SWISH production run
    assert SWISH_ITER_LENGTH.value_in_unit(unit.picosecond) == pytest.approx(10.0)


def test_resolve_iter_length_defaults_to_swish_constant():
    """resolve_iter_length(None) returns the 10 ps SWISH default."""
    assert resolve_iter_length() is SWISH_ITER_LENGTH
    assert resolve_iter_length(None).value_in_unit(unit.picosecond) == pytest.approx(10.0)


def test_resolve_iter_length_overrides_and_validates():
    """A positive override is honoured; non-positive values are rejected."""
    assert resolve_iter_length(25).value_in_unit(unit.picosecond) == pytest.approx(25.0)
    with pytest.raises(ValueError):
        resolve_iter_length(0)
    with pytest.raises(ValueError):
        resolve_iter_length(-5)


def test_sampler_iter_length_accessor_defaults_and_override():
    """The OPES sampler falls back to the SWISH default until an interval is set."""
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    assert sampler._iter_length_ps() == pytest.approx(
        SWISH_ITER_LENGTH.value_in_unit(unit.picosecond),
    )
    sampler.set_iter_length_ps(25 * unit.picosecond)
    assert sampler._iter_length_ps() == pytest.approx(25.0)


def test_default_exchange_path_does_not_compute_component_energies():
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    sampler._replica_thermodynamic_states = np.array([0, 1])
    sampler._gmx_boundary_energy_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    sampler._n_proposed_matrix = np.zeros((2, 2), dtype=int)
    sampler._n_accepted_matrix = np.zeros((2, 2), dtype=int)
    sampler.set_exchange_diagnostics(False)
    sampler._attempt_swap(0, 1)


def test_verbose_exchange_diagnostic_contains_all_required_fields(tmp_path):
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    sampler._replica_thermodynamic_states = np.array([0, 1])
    sampler._gmx_boundary_energy_matrix = np.array([[5.0, 0.0], [0.0, 5.0]])
    sampler._gmx_boundary_physical_energy_matrix = np.array(
        [[2.0, 0.0], [0.0, 2.0]],
    )
    sampler._n_proposed_matrix = np.zeros((2, 2), dtype=int)
    sampler._n_accepted_matrix = np.zeros((2, 2), dtype=int)
    sampler._reporter = SimpleNamespace(
        filepath=str(tmp_path / 'swish_x_output.nc'),
        close=lambda: None,
    )
    sampler._iteration = 1
    sampler.set_iter_length_ps(10 * unit.picosecond)
    sampler.set_exchange_diagnostics(True)

    sampler._attempt_swap(0, 1)

    diagnostic = json.loads(
        (tmp_path / 'swish_x_output_exchange_diagnostics.jsonl').read_text(),
    )
    assert diagnostic == {
        'acceptance_probability': 1.0,
        'accepted': True,
        'exchange_bias_mode': 'gmx-staggered',
        'iteration': 1,
        'opes_reduced_potential_difference': -6.0,
        'physical_reduced_potential_difference': -4.0,
        'replica_pair': [0, 1],
        'state_pair': [0, 1],
        'time_ps': 10.0,
        'total_exchange_exponent': 10.0,
    }


def _initialized_opes(deltaf_kj: list[float], u_ref_kj: float) -> OpesExpanded:
    opes = OpesExpanded(
        temperature=300 * unit.kelvin,
        t_min=300 * unit.kelvin,
        t_max=350 * unit.kelvin,
        n_temps=2,
        pace=10,
        observation_steps=1,
    )
    opes._deltaf_kj = np.array(deltaf_kj, dtype=float)
    opes._u_ref_kj = u_ref_kj
    opes._initialized = True
    return opes


def test_opes_mpi_sync_includes_absolute_gmx_step(monkeypatch):
    opes_states = [
        _initialized_opes([0.0, 2.0], 100.0),
        _initialized_opes([0.0, 3.0], 200.0),
    ]
    synchronized_states = [copy.deepcopy(opes.get_state()) for opes in opes_states]
    synchronized_states[0]['counter'] = 11
    synchronized_states[1]['counter'] = 12
    moves = [
        SimpleNamespace(_cumulative_steps=0, _gmx_step=0),
        SimpleNamespace(_cumulative_steps=0, _gmx_step=0),
    ]

    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    sampler._sampler_states = [None, None]
    sampler._thermodynamic_states = [
        SimpleNamespace(_composable_states=[opes]) for opes in opes_states
    ]
    sampler._mcmc_moves = moves
    entries = [
        (0, synchronized_states[0], 20, 21),
        (1, synchronized_states[1], 30, 31),
    ]
    monkeypatch.setattr(
        md_utils_module.mpiplus,
        'distribute',
        lambda *args, **kwargs: entries,
    )

    sampler._sync_opes_state_across_ranks()

    assert [opes.get_state()['counter'] for opes in opes_states] == [11, 12]
    assert [move._cumulative_steps for move in moves] == [20, 30]
    assert [move._gmx_step for move in moves] == [21, 31]


def test_restart_restores_gmx_lifecycle_state(monkeypatch, tmp_path):
    opes = _initialized_opes([0.0, 2.0], 100.0)
    restored_state = copy.deepcopy(opes.get_state())
    restored_state['counter'] = 37
    restored_state['u_ref_kj'] = 123.5

    def read_dict(path):
        if path == OpesReplicaExchangeSampler.OPES_BIAS_PATH:
            return {
                '__lifecycle_revision__': md_utils_module.SWISH_X_LIFECYCLE_REVISION,
                '__checkpoint_iteration__': 5,
                'state0': {
                    'opes': restored_state,
                    'cumulative_steps': 50,
                    'gmx_step': 51,
                },
            }
        raise KeyError(path)

    reporter = SimpleNamespace(
        filepath=str(tmp_path / 'swish_x_output.nc'),
        read_dict=read_dict,
        close=lambda: None,
    )
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    sampler._reporter = reporter
    sampler._iteration = 5
    sampler._thermodynamic_states = [
        SimpleNamespace(_composable_states=[opes]),
    ]
    sampler._mcmc_moves = [SimpleNamespace(_cumulative_steps=0, _gmx_step=0)]
    monkeypatch.setattr(
        md_utils_module.multistate.ReplicaExchangeSampler,
        '_restore_sampler_from_reporter',
        lambda self, source_reporter: None,
    )

    sampler._restore_sampler_from_reporter(reporter)

    assert opes.get_state()['counter'] == 37
    assert opes.get_state()['u_ref_kj'] == pytest.approx(123.5)
    assert sampler._mcmc_moves[0]._cumulative_steps == 50
    assert sampler._mcmc_moves[0]._gmx_step == 51


def test_mix_neighboring_replicas_alternates_deterministically():
    """The OPES sampler picks the even/odd neighbour partition by iteration parity.

    openmmtools' base scheme randomises the offset each iteration, which lets a replica
    that just swapped up the ladder swap straight back down when the same partition
    recurs. Deterministic alternation (as in GROMACS -hrex) offers disjoint partitions on
    consecutive attempts, forbidding immediate back-tracking.
    """
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    n = 6
    sampler._sampler_states = [None] * n  # n_replicas property reads len()

    def proposed_pairs(iteration):
        sampler._iteration = iteration
        sampler._replica_thermodynamic_states = np.arange(n)
        sampler._n_proposed_matrix = np.zeros((n, n))
        sampler._n_accepted_matrix = np.zeros((n, n))
        # huge off-diagonal energies -> every swap rejected -> state assignment unchanged,
        # so the proposed matrix cleanly records which neighbour pairs were attempted.
        energies = np.full((n, n), 1e6)
        np.fill_diagonal(energies, 0.0)
        sampler._energy_thermodynamic_states = energies
        sampler._mix_neighboring_replicas()
        return {(i, i + 1) for i in range(n - 1)
                if sampler._n_proposed_matrix[i, i + 1] > 0}

    even, odd = {(0, 1), (2, 3), (4, 5)}, {(1, 2), (3, 4)}
    # deterministic by parity (never random)
    assert proposed_pairs(0) == even
    assert proposed_pairs(2) == even
    assert proposed_pairs(1) == odd
    assert proposed_pairs(3) == odd
    # consecutive iterations offer disjoint partitions -> no immediate back-tracking
    assert proposed_pairs(4).isdisjoint(proposed_pairs(5))


def _write_minimal_nc(path):
    """Write a tiny valid netCDF4 file at ``path`` for backup-roundtrip tests."""
    import netCDF4
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('n', 3)
        var = ds.createVariable('x', 'f4', ('n',))
        var[:] = [1.0, 2.0, 3.0]


def test_backup_checkpoint_writes_bak_when_valid(tmp_path):
    """backup_checkpoint should produce ``.bak`` siblings for both NC files."""
    prefix = str(tmp_path / 'swish_output')
    _write_minimal_nc(f'{prefix}.nc')
    _write_minimal_nc(f'{prefix}_checkpoint.nc')

    assert backup_checkpoint(prefix) is True
    assert (tmp_path / 'swish_output.nc.bak').exists()
    assert (tmp_path / 'swish_output_checkpoint.nc.bak').exists()


def test_backup_checkpoint_skips_when_corrupt(tmp_path):
    """A corrupt live file must not overwrite a previously-good backup."""
    prefix = str(tmp_path / 'swish_output')
    _write_minimal_nc(f'{prefix}.nc')
    _write_minimal_nc(f'{prefix}_checkpoint.nc')
    assert backup_checkpoint(prefix) is True

    # Corrupt the live checkpoint and confirm the call refuses to overwrite the backup.
    with open(f'{prefix}_checkpoint.nc', 'wb') as fh:
        fh.write(b'not netcdf')
    assert backup_checkpoint(prefix) is False


def test_can_restart_falls_back_to_backup(tmp_path):
    """Restart must remain possible from a ``.bak`` when the live file is corrupt."""
    prefix = str(tmp_path / 'swish_output')
    _write_minimal_nc(f'{prefix}.nc')
    _write_minimal_nc(f'{prefix}_checkpoint.nc')
    assert backup_checkpoint(prefix) is True
    with open(f'{prefix}_checkpoint.nc', 'wb') as fh:
        fh.write(b'corrupt')
    assert can_restart_swish_production(prefix) is True
