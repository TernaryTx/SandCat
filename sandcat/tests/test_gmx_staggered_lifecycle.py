from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest
from openmmtools.states import SamplerState

import sandcat.opes_expanded as opes_expanded_module
from sandcat.md_utils import (
    _gmx_boundary_reduced_energy_matrix,
    _gmx_deo_offset,
    OpesReplicaExchangeSampler,
)
from sandcat.opes_expanded import OpesExpanded, OpesExpandedDynamicsMove
from sandcat.opes_expanded import _read_contact_map


class _AffineBias:
    """Minimal post-update OPES stand-in for boundary-algebra tests."""

    def __init__(self, slope: float, intercept: float):
        self.slope = float(slope)
        self.intercept = float(intercept)

    def compute_bias_v(self, u_kj: float) -> float:
        return self.slope * float(u_kj) + self.intercept


class _ConstantBetaThermodynamicState:
    """Only the reduced-potential contract needed by the pure matrix helper."""

    temperature = None
    pressure = None
    surface_tension = None

    def __init__(self, beta_per_kj: float):
        self.beta_per_kj = float(beta_per_kj)

    def _compute_reduced_potential(
        self, potential_energy, temperature, volume, pressure,
        area, surface_tension,
    ) -> float:
        return self.beta_per_kj * potential_energy.value_in_unit(
            unit.kilojoule_per_mole,
        )


def test_gmx_boundary_matrix_uses_pre_resident_and_post_retained_cross_bias():
    """Reproduce a hand-calculated native GREX boundary exactly.

    Rows are configurations and columns are Hamiltonians.  The physical
    energies are

        U_i(x_i)=10, U_j(x_i)=31, U_i(x_j)=14, U_j(x_j)=20.

    Native GREX keeps the cached pre-update bias on each resident diagonal,
    but evaluates a foreign post-update OPES state at the partner's retained
    resident ENERGY scalar.  It must therefore produce this kJ/mol matrix::

        [[10 + 11, 31 - 3],
         [14 + 65, 20 + 43]] = [[21, 28], [79, 63]].

    The swap delta is 23 kJ/mol (15 physical + 8 bias), giving a log
    acceptance of -11.5 at beta=0.5 mol/kJ.
    """
    engine_kj = np.array([[10.0, 31.0], [14.0, 20.0]])
    geometric_kj = np.zeros_like(engine_kj)
    resident_state_ids = np.array([0, 1])
    resident_u_kj = np.array([10.0, 20.0])
    resident_pre_bias_kj = np.array([11.0, 43.0])
    post_opes_states = [
        _AffineBias(3.0, 5.0),
        _AffineBias(-1.0, 7.0),
    ]

    matrix = _gmx_boundary_reduced_energy_matrix(
        engine_kj,
        geometric_kj,
        resident_state_ids,
        resident_u_kj,
        resident_pre_bias_kj,
        post_opes_states,
        [_ConstantBetaThermodynamicState(0.5)] * 2,
        np.ones(2),
    )

    np.testing.assert_allclose(matrix, [[10.5, 14.0], [39.5, 31.5]])
    swap_delta = matrix[0, 1] + matrix[1, 0] \
        - matrix[0, 0] - matrix[1, 1]
    assert swap_delta == pytest.approx(11.5)
    assert -swap_delta == pytest.approx(-11.5)


def test_gmx_boundary_matrix_adds_geometric_bias_outside_energy_cv():
    """The wall contributes to HREX but never changes retained OPES ENERGY."""
    engine_kj = np.array([[10.0, 31.0], [14.0, 20.0]])
    geometric_kj = np.array([[0.5, 1.5], [2.5, 3.5]])
    retained_u = np.array([10.0, 20.0])
    opes_states = [_AffineBias(3.0, 5.0), _AffineBias(-1.0, 7.0)]

    matrix = _gmx_boundary_reduced_energy_matrix(
        engine_kj,
        geometric_kj,
        np.array([0, 1]),
        retained_u,
        np.array([11.0, 43.0]),
        opes_states,
        [_ConstantBetaThermodynamicState(1.0)] * 2,
        np.ones(2),
    )

    np.testing.assert_allclose(matrix, [[21.5, 29.5], [81.5, 66.5]])
    # Cross-bias values remain post_1(10)=-3 and post_0(20)=65: the
    # geometric terms above are not fed into compute_bias_v().
    np.testing.assert_allclose(
        matrix - engine_kj - geometric_kj,
        [[11.0, -3.0], [65.0, 43.0]],
    )


def test_sampler_run_orders_gmx_boundary_lifecycle(monkeypatch) -> None:
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    events = []
    move = OpesExpandedDynamicsMove.__new__(OpesExpandedDynamicsMove)
    move.n_steps = 10
    move.pace = 10
    sampler._mcmc_moves = [move]
    sampler._sampler_states = [object()]
    sampler._replica_thermodynamic_states = np.array([0])
    sampler._iteration = 1
    sampler._number_of_iterations = 2
    sampler._gmx_boundary_engine_energy = np.zeros((1, 1))
    sampler._gmx_boundary_geometric_energy = np.zeros((1, 1))
    sampler._gmx_boundary_resident_state_ids = np.array([0])
    sampler._gmx_boundary_resident_u = np.array([0.0])
    sampler._gmx_boundary_resident_pre_bias = np.array([0.0])
    sampler._gmx_boundary_volumes = np.array([1.0])
    sampler._thermodynamic_states = [object()]
    sampler._has_opes_states = lambda: True
    sampler._is_completed = lambda limit: sampler._iteration >= limit
    sampler._exchange_diagnostics_are_enabled = lambda: False
    sampler._propagate_gmx_replicas = lambda: events.append('propagate')
    sampler._sync_opes_state_across_ranks = lambda: events.append('sync')
    sampler._find_opes = lambda state: None
    sampler._mix_replicas = lambda: events.append('mix') or np.array([0])
    sampler._compute_energies = lambda: events.append('energies')
    sampler._report_iteration = lambda: events.append('report')
    sampler._update_analysis = lambda: events.append('analysis')
    sampler._check_nan_energy = lambda: events.append('nan')
    sampler._flush_all_colvars = lambda: events.append('flush')
    monkeypatch.setattr(
        'sandcat.md_utils._gmx_boundary_reduced_energy_matrix',
        lambda *args: events.append('boundary') or np.zeros((1, 1)),
    )

    sampler.run(n_iterations=1)

    assert events == [
        'propagate', 'sync', 'boundary', 'mix', 'energies', 'report', 'analysis', 'nan',
        'flush',
    ]


@pytest.mark.parametrize(
    ("absolute_step", "expected_offset"),
    [
        (5_000, 0),
        (10_000, 1),
        (15_000, 0),
        # Native checkpoint at exchange ordinal 100 resumes on the odd
        # partition; keying to a fresh SandCat iteration would invert it.
        (500_000, 1),
    ],
)
def test_gmx_deo_partition_is_keyed_to_absolute_exchange_ordinal(
    absolute_step: int,
    expected_offset: int,
):
    boundary_index = absolute_step // 5_000
    assert _gmx_deo_offset(boundary_index) == expected_offset


class _EventIntegrator:
    def __init__(self, events):
        self.events = events

    def step(self, n_steps: int) -> None:
        self.events.append(("integrate", int(n_steps)))


class _EventOpes:
    def __init__(self, events):
        self.events = events

    def compute_bias_v(self, u_kj: float) -> float:
        self.events.append(("pre_bias", float(u_kj)))
        return float(u_kj) + 1.0

    def apply_to_context(self, context) -> None:
        self.events.append(("push_post_opes", context))


class _EventSamplerState:
    def __init__(self, events, label="advanced"):
        self.events = events
        self.label = label

    def __deepcopy__(self, memo):
        return type(self)(self.events, label="boundary")

    def apply_to_context(self, context, ignore_velocities=False) -> None:
        self.events.append(("load_advanced_state", context, ignore_velocities))

    def update_from_context(self, context, **kwargs) -> None:
        self.events.append((f"capture_{self.label}_state", context, kwargs))


class _EventCache:
    def __init__(self, events, context, integrator):
        self.events = events
        self.context = context
        self.integrator = integrator

    def get_context(self, thermodynamic_state, integrator_template):
        self.events.append(("get_context", thermodynamic_state))
        return self.context, self.integrator


def test_apply_gmx_staggered_orders_update_old_force_step_and_post_push(monkeypatch):
    """The boundary decision belongs to x(t), but mutation targets x(t+dt).

    In particular, the Python estimator must become theta+ before the one
    boundary step while the live Context remains theta-.  Pushing theta+ to
    the Context before that step would silently remove GMX's cached-old-force
    stagger.
    """
    events = []
    context = object()
    integrator = _EventIntegrator(events)
    opes = _EventOpes(events)
    sampler_state = _EventSamplerState(events)
    thermodynamic_state = object()
    cache = _EventCache(events, context, integrator)

    move = OpesExpandedDynamicsMove.__new__(OpesExpandedDynamicsMove)
    move.seed = None
    move.reassign_velocities = False
    move.pace = 10
    move._gmx_step = 0
    move._cumulative_steps = 0
    move._find_opes_state = lambda state: opes
    move._resolve_cache = lambda supplied: cache
    move._build_integrator = lambda state: object()
    move._update_opes_python = lambda op, u: events.append(
        ("update_python_opes", float(u)),
    )

    monkeypatch.setattr(
        opes_expanded_module,
        "_read_u_ff_kj",
        lambda ctx: events.append(("read_resident_energy", ctx)) or 10.0,
    )
    monkeypatch.setattr(
        opes_expanded_module,
        "_read_geometric_bias_kj",
        lambda ctx: events.append(("read_geometric_bias", ctx)) or 2.0,
    )

    boundary_state, payload = move.apply_gmx_staggered(
        thermodynamic_state,
        sampler_state,
        boundary_step=10,
        context_cache=object(),
    )

    names = [event[0] for event in events]
    assert names == [
        "get_context",
        "load_advanced_state",
        # Fresh GMX reaches x(N) before its first exchange; it does not
        # perform an artificial boundary/update at x(0).
        "integrate",
        "read_resident_energy",
        "read_geometric_bias",
        "pre_bias",
        "capture_boundary_state",
        "update_python_opes",
        "integrate",
        "push_post_opes",
        "capture_advanced_state",
    ]
    boundary_integrate = max(
        index for index, name in enumerate(names) if name == "integrate"
    )
    assert names.index("update_python_opes") < boundary_integrate
    assert boundary_integrate < names.index("push_post_opes")
    assert events[2] == ("integrate", 10)
    assert events[8] == ("integrate", 1)
    assert isinstance(boundary_state, _EventSamplerState)
    assert boundary_state.label == "boundary"
    assert payload == {
        "resident_u_kj": 10.0,
        "resident_geometric_kj": 2.0,
        "resident_pre_bias_kj": 11.0,
        "boundary_step": 10,
        "post_step": 11,
        "colvar_rows": [],
    }


def test_gmx_boundary_snapshot_omits_velocities_and_preserves_coordinates() -> None:
    system = mm.System()
    system.addParticle(12.0)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    sampler_state = SamplerState(
        positions=np.array([[0.1, 0.0, 0.0]]) * unit.nanometer,
        velocities=np.array([[1.0, 0.0, 0.0]]) * unit.nanometer / unit.picosecond,
        box_vectors=np.eye(3) * 3.0 * unit.nanometer,
    )
    events = []
    cache = _EventCache(events, context, integrator)
    move = OpesExpandedDynamicsMove(
        timestep=1.0 * unit.femtosecond,
        collision_rate=0.0 / unit.picosecond,
        n_steps_per_iteration=10,
        n_temps=2,
        pace=100,
    )
    move._find_opes_state = lambda state: _EventOpes(events)
    move._resolve_cache = lambda supplied: cache
    move._build_integrator = lambda state: integrator

    boundary_state, payload = move.apply_gmx_staggered(
        object(), sampler_state, boundary_step=10,
    )

    assert boundary_state.velocities is None
    np.testing.assert_allclose(
        boundary_state.positions.value_in_unit(unit.nanometer), [[0.11, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        sampler_state.positions.value_in_unit(unit.nanometer), [[0.111, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        boundary_state.box_vectors.value_in_unit(unit.nanometer), np.eye(3) * 3.0,
    )
    assert payload['boundary_step'] == 10
    assert payload['post_step'] == 11
    del context, integrator


def test_gmx_lifecycle_absolute_step_survives_move_pickle_roundtrip():
    """Restarting must not reset cadence to a fresh SandCat iteration."""
    move = OpesExpandedDynamicsMove(
        timestep=2.0 * unit.femtosecond,
        collision_rate=1.0 / unit.picosecond,
        n_steps_per_iteration=5_000,
        n_temps=4,
        pace=5_000,
    )
    move._cumulative_steps = 500_001
    move._gmx_step = 500_001

    restored = OpesExpandedDynamicsMove.__new__(OpesExpandedDynamicsMove)
    restored.__setstate__(move.__getstate__())

    assert restored._gmx_step == 500_001
    assert restored._cumulative_steps == 500_001


def test_colvar_rows_are_buffered_and_written_in_plumed_format(tmp_path):
    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    sampler._colvar_output_prefix = str(tmp_path / 'run')
    sampler._colvar_interval_ps = 2.0
    sampler._colvar_flush_rows = 3
    sampler._colvar_buffers = {0: []}
    path = tmp_path / 'run_COLVAR.0'

    sampler._append_colvar_rows(0, [(2.0, 29.0, 0.0, -10.0, 1.0)])
    assert not path.exists()
    sampler._append_colvar_rows(0, [
        (4.0, 30.0, 0.0, -11.0, 1.1),
        (6.0, 31.0, 2.5, -12.0, 1.2),
    ])

    lines = path.read_text().splitlines()
    assert lines[:2] == [
        '#! FIELDS time cmap uwall.bias ene opesX.bias',
        '#! SET print_stride 2',
    ]
    assert lines[2:] == [
        ' 2.000000  29.0000   0.0000 -10.0000   1.0000',
        ' 4.000000  30.0000   0.0000 -11.0000   1.1000',
        ' 6.000000  31.0000   2.5000 -12.0000   1.2000',
    ]
    data = np.loadtxt(path, comments='#!')
    np.testing.assert_allclose(data[:, [0, 1, 2]], [
        [2.0, 29.0, 0.0], [4.0, 30.0, 0.0], [6.0, 31.0, 2.5],
    ])
    assert not hasattr(OpesReplicaExchangeSampler, '_write_opes_energy_log')
    assert not (tmp_path / 'run_ENE.0.data').exists()


def test_deltaf_rows_use_plumed_general_format(tmp_path):
    opes = OpesExpanded(
        temperature=300 * unit.kelvin,
        t_min=300 * unit.kelvin,
        t_max=350 * unit.kelvin,
        n_temps=3,
    )
    opes._initialized = True
    opes._temps_k = np.array([300.0, 301.656, 350.0])
    opes._rct_kj = -11.51167394495033
    opes._deltaf_kj = np.array([
        8.653975976238964e-18,
        2819.122476856191,
        73173.21596393967,
    ])

    sampler = OpesReplicaExchangeSampler.__new__(OpesReplicaExchangeSampler)
    sampler._thermodynamic_states = [SimpleNamespace(_composable_states=[opes])]
    sampler._reporter = SimpleNamespace(
        filepath=str(tmp_path / 'run.nc'),
        close=lambda: None,
    )
    sampler._iteration = 100
    sampler.set_iter_length_ps(10 * unit.picosecond)

    sampler._write_opes_deltaf_log()

    lines = (tmp_path / 'run_DeltaFs.0.data').read_text().splitlines()
    assert lines[:2] == [
        '#! FIELDS time rct DeltaF_300 DeltaF_301.656 DeltaF_350',
        '#! SET print_stride  10',
    ]
    assert lines[2].split() == [
        '1000',
        '-11.51167394495033',
        '8.653975976238964e-18',
        '2819.122476856191',
        '73173.21596393967',
    ]
    assert lines[2].count('e') == 1


def test_contact_map_reporter_reads_live_collective_variable():
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    contact = mm.CustomBondForce('r')
    contact.addBond(0, 1)
    wall = mm.CustomCVForce('cmap')
    wall.addCollectiveVariable('cmap', contact)
    wall.setForceGroup(2)
    system.addForce(wall)
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    context = mm.Context(system, integrator)
    context.setPositions([[0, 0, 0], [0.37, 0, 0]] * unit.nanometer)

    assert _read_contact_map(context) == pytest.approx(0.37)
