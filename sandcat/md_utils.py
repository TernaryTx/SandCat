from __future__ import annotations

import copy
from collections import deque
from collections.abc import Sequence
import itertools
import json
import logging
import math
import os
import shutil
from typing import Any
import mdtraj
import mpiplus
import netCDF4
import numpy as np
from openmm import (
    app, unit,
    CustomBondForce, CustomCVForce, CustomNonbondedForce, NonbondedForce,
    LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
)

from openmmtools import states, mcmc, multistate

from sandcat.md_variables import FRICTION, PRESSURE, TEMPERATURE, get_n_steps, REPORT_INTERVAL
from sandcat.provenance import OUTPUT_PROVENANCE_SCHEMA_VERSION
from sandcat.system_setup import add_position_restraints
from sandcat.variables import (
    CHECKPOINT_BACKUP_SUFFIX,
    PROBE_RESNAME,
    SWISH_APOLAR_TYPES_BY_FORCEFIELD,
    SWISH_ITER_LENGTH,
    SWISH_OUTPUT_PREFIX,
    SWISH_X_OUTPUT_PREFIX,
    TIMESCAPES_ATOMS,
)
from sandcat.opes_expanded import (
    PLUMED_GEOMETRIC_BIAS_FORCE_GROUP,
    OpesExpanded, OpesExpandedDynamicsMove, _read_geometric_bias_kj,
    _read_u_ff_kj,
    add_opes_expanded_bias,
)

# Residue names that are never protein — excluded from position restraint selection.
# Includes water, common monatomic ions, and the SWISH probe.
_SOLVENT_RESNAMES = frozenset({'HOH', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN', PROBE_RESNAME})

PROBE_HAMILTONIAN_REVISION = 'probe-intermolecular-repulsion-v3'
SWISH_X_LIFECYCLE_REVISION = 'gmx_staggered_boundary_v1'


def _is_primary_mpi_rank() -> bool:
    """Return whether this process is rank 0, including non-MPI execution."""
    communicator = mpiplus.get_mpicomm()
    return communicator is None or communicator.rank == 0


def write_probe_hamiltonian_provenance(
        output_prefix: str,
        provenance: dict[str, int | str],
) -> None:
    """Write probe Hamiltonian provenance beside a simulation output prefix.

    Args:
        output_prefix: Prefix used for the simulation output files.
        provenance: Record returned by :func:`add_probe_repulsion`.
    """
    provenance_path = f'{output_prefix}_probe_hamiltonian.json'
    temporary_path = f'{provenance_path}.tmp'
    with open(temporary_path, 'w') as provenance_file:
        json.dump(provenance, provenance_file, indent=2, sort_keys=True)
        provenance_file.write('\n')
        provenance_file.flush()
        os.fsync(provenance_file.fileno())
    os.replace(temporary_path, provenance_path)


def write_output_provenance(
        output_prefix: str,
        provenance: dict[str, object],
) -> None:
    """Atomically write workflow and Hamiltonian provenance beside output."""
    provenance_path = f'{output_prefix}_provenance.json'
    temporary_path = f'{provenance_path}.tmp'
    with open(temporary_path, 'w') as provenance_file:
        json.dump(provenance, provenance_file, indent=2, sort_keys=True)
        provenance_file.write('\n')
        provenance_file.flush()
        os.fsync(provenance_file.fileno())
    os.replace(temporary_path, provenance_path)


def validate_output_provenance(output_prefix: str) -> dict[str, object]:
    """Fail closed when restart provenance does not match this implementation."""
    provenance_path = f'{output_prefix}_provenance.json'
    if not os.path.isfile(provenance_path):
        raise RuntimeError(
            f'Missing Hamiltonian provenance for restart: {provenance_path}',
        )
    with open(provenance_path) as provenance_file:
        provenance = json.load(provenance_file)
    schema_version = provenance.get('schema_version')
    if (schema_version is not None
            and schema_version != OUTPUT_PROVENANCE_SCHEMA_VERSION):
        raise RuntimeError(
            'Unsupported output provenance schema version: '
            f'{schema_version!r} != {OUTPUT_PROVENANCE_SCHEMA_VERSION!r}',
        )
    if provenance.get('revision') != PROBE_HAMILTONIAN_REVISION:
        raise RuntimeError(
            'Reporter Hamiltonian revision mismatch: '
            f"{provenance.get('revision')!r} != {PROBE_HAMILTONIAN_REVISION!r}",
        )
    if (provenance.get('opes_enabled')
            and provenance.get('swish_x_lifecycle') != SWISH_X_LIFECYCLE_REVISION):
        raise RuntimeError(
            'SWISH-X lifecycle revision mismatch or missing provenance: '
            f"{provenance.get('swish_x_lifecycle')!r} "
            f"!= {SWISH_X_LIFECYCLE_REVISION!r}",
        )
    return provenance

# Default MD time between replica-exchange attempts. 10 ps matches the
# nstswapcoords default in the SWISH GROMACS setup. Callers may override it
# per run via the ``exchange_interval_ps`` argument on the production helpers.
def resolve_iter_length(exchange_interval_ps: float | None = None) -> unit.Quantity:
    """Return the replica-exchange iteration length as an OpenMM Quantity.

    Falls back to SWISH_ITER_LENGTH (10 ps) when ``exchange_interval_ps`` is None.
    """
    if exchange_interval_ps is None:
        return SWISH_ITER_LENGTH
    if exchange_interval_ps <= 0:
        raise ValueError('exchange_interval_ps must be positive.')
    return exchange_interval_ps * unit.picosecond


def _get_nonbonded_force(system: app.System) -> NonbondedForce:
    """Return the NonbondedForce from a system, raising if none is found."""
    return next(
        f for f in (system.getForce(i) for i in range(system.getNumForces()))
        if isinstance(f, NonbondedForce)
    )


def _get_swish_interaction_indices(
        topology: app.Topology,
        probe_resname: str,
        forcefield_name: str | None = None,
        forcefield: app.ForceField | None = None,
) -> tuple[list[int], list[int]]:
    """Return water-oxygen and selected protein indices for SWISH scaling.

    Args:
        topology: Solvated OpenMM topology.
        probe_resname: Residue name used by the mixed-solvent probe.
        forcefield_name: Key identifying the force field used to build the system.
        forcefield: OpenMM force field used to assign atom types.

    Returns:
        Water oxygen indices and protein indices whose OW interactions are scaled.

    Raises:
        ValueError: If the requested selection data is missing or invalid.
    """
    water_ox_indices = [
        atom.index for atom in topology.atoms()
        if atom.residue.name == 'HOH'
        and atom.element is not None
        and atom.element.symbol == 'O'
    ]
    if forcefield_name not in SWISH_APOLAR_TYPES_BY_FORCEFIELD:
        raise ValueError(
            f"No versioned SWISH atom-type selection is available for "
            f"force field {forcefield_name!r}.",
        )
    atom_types = _get_forcefield_atom_types(topology, forcefield)
    selected_types = SWISH_APOLAR_TYPES_BY_FORCEFIELD[forcefield_name]
    apolar_indices = [
        atom.index for atom in topology.atoms()
        if atom.residue.name != probe_resname
        and atom_types[atom.index] in selected_types
    ]
    return water_ox_indices, apolar_indices


def _get_forcefield_atom_types(
        topology: app.Topology,
        forcefield: app.ForceField | None,
) -> dict[int, str]:
    """Map topology atom indices to OpenMM force-field atom types.

    Args:
        topology: Topology passed to ``ForceField.createSystem``.
        forcefield: Force field that parametrised the topology.

    Returns:
        Mapping from atom index to assigned atom-type name.

    Raises:
        ValueError: If the force field is unavailable or a template cannot be mapped.
    """
    if forcefield is None:
        raise ValueError(
            'SWISH atom-type selection requires the OpenMM ForceField used to '
            'build the system; no legacy fallback is performed.',
        )
    system_data_factory = getattr(forcefield, '_SystemData', None)
    match_templates = getattr(forcefield, '_matchAllResiduesToTemplates', None)
    if system_data_factory is not None and match_templates is not None:
        try:
            system_data = system_data_factory(topology)
            match_templates(system_data, topology, {}, False)
        except Exception as error:
            raise ValueError(
                'Could not assign force-field atom types for SWISH selection.',
            ) from error
        return {
            atom.index: system_data.atomType[atom]
            for atom in topology.atoms()
        }

    # Lightweight force-field doubles used by callers and tests may only expose
    # the public template API. OpenMM ForceField objects take the graph-matching
    # path above, which also handles template/topology atom-name differences.
    residues = list(topology.residues())
    try:
        templates = forcefield.getMatchingTemplates(topology)
    except Exception as error:
        raise ValueError(
            'Could not assign force-field atom types for SWISH selection.',
        ) from error
    if len(templates) != len(residues):
        raise ValueError('Force-field template count does not match topology residues.')

    atom_types: dict[int, str] = {}
    for residue, template in zip(residues, templates):
        template_types = {atom.name: atom.type for atom in template.atoms}
        for atom in residue.atoms():
            atom_type = template_types.get(atom.name)
            if atom_type is None:
                raise ValueError(
                    f'No force-field atom type for {residue.name} {atom.name} '
                    f'(index {atom.index}).',
                )
            atom_types[atom.index] = atom_type
    return atom_types


def _build_swish_correction_force(
        system: app.System,
        water_ox_indices: list[int],
        apolar_indices: list[int],
        eps_vals: np.ndarray,
        sig_vals: np.ndarray,
        initial_lam: float,
) -> CustomNonbondedForce:
    """Build the SWISH OW-apolar epsilon scaling correction: U = (lam-1)*4*eps_ij*((sig_ij/r)^12-(sig_ij/r)^6).

    lam=1.0 gives zero correction. Exclusions are copied from the system NonbondedForce.
    """
    n_particles = system.getNumParticles()
    correction = CustomNonbondedForce(
        'delta_lam * 4 * sqrt(eps1 * eps2) * ((0.5 * (sig1 + sig2) / r)^12 - (0.5 * (sig1 + sig2) / r)^6);'
        'delta_lam = lam - 1'
    )
    correction.addGlobalParameter('lam', initial_lam)
    correction.addPerParticleParameter('eps')
    correction.addPerParticleParameter('sig')
    for idx in range(n_particles):
        correction.addParticle([eps_vals[idx], sig_vals[idx]])
    correction.addInteractionGroup(water_ox_indices, apolar_indices)
    correction.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    correction.setCutoffDistance(1.2 * unit.nanometer)
    nonbonded = _get_nonbonded_force(system)
    for exc_idx in range(nonbonded.getNumExceptions()):
        p1, p2, *_ = nonbonded.getExceptionParameters(exc_idx)
        correction.addExclusion(p1, p2)
    return correction


def add_probe_repulsion(
        system: app.System,
        omm_model: app.Modeller,
        probe_resname: str | Sequence[str] = PROBE_RESNAME,
        sigma_nm: float = 0.8,
        epsilon_kj_per_mol: float = 0.0015,
        correct_probe_14_exceptions: bool = True,
) -> dict[str, int | str]:
    """Correct the probe-probe LJ term and its intramolecular 1-4 exceptions.

    The custom force applies only the difference between the target GROMACS-equivalent
    probe-carbon interaction and the ordinary OpenMM mixed probe-probe LJ term. This
    avoids double-counting while preserving probe interactions with other molecules.
    Probe-carbon exceptions separated by three bonds are updated in the existing
    NonbondedForce, retaining their charge products and using half the normal epsilon.
    This correction must be disabled for arbitrary probes because their
    intramolecular parameters come from their own force field.

    Args:
        system: OpenMM System to modify.
        omm_model: Model containing probe residues.
        probe_resname: Residue name or names assigned to probe molecules.
        sigma_nm: Target intermolecular probe repulsion sigma.
        epsilon_kj_per_mol: Target intermolecular probe repulsion epsilon.
        correct_probe_14_exceptions: Whether to apply the historical built-in-probe
            carbon 1-4 correction.

    Returns:
        A small provenance record suitable for attaching to run output metadata.
    """
    probe_resnames = (
        {probe_resname} if isinstance(probe_resname, str) else set(probe_resname)
    )
    probe_heavy_indices_by_residue = [
        [
            atom.index for atom in residue.atoms()
            if atom.element is not None and atom.element.symbol != 'H'
        ]
        for residue in omm_model.topology.residues()
        if residue.name in probe_resnames
    ]
    probe_heavy_indices = [
        atom_index
        for residue_indices in probe_heavy_indices_by_residue
        for atom_index in residue_indices
    ]
    probe_group_by_particle = [0] * system.getNumParticles()
    for group_index, residue_indices in enumerate(
            probe_heavy_indices_by_residue, start=1):
        for atom_index in residue_indices:
            probe_group_by_particle[atom_index] = group_index

    probe_carbon_indices = {
        atom.index for atom in omm_model.topology.atoms()
        if atom.residue.name in probe_resnames
        and atom.element is not None
        and atom.element.symbol == 'C'
    }
    adjacency: dict[int, set[int]] = {}
    for atom_a, atom_b in omm_model.topology.bonds():
        adjacency.setdefault(atom_a.index, set()).add(atom_b.index)
        adjacency.setdefault(atom_b.index, set()).add(atom_a.index)

    def bond_distance(first_index: int, second_index: int) -> int | None:
        queue = deque([(first_index, 0)])
        visited = {first_index}
        while queue:
            current_index, distance = queue.popleft()
            if current_index == second_index:
                return distance
            for neighbour_index in adjacency.get(current_index, set()):
                if neighbour_index not in visited:
                    visited.add(neighbour_index)
                    queue.append((neighbour_index, distance + 1))
        return None

    nonbonded_force = _get_nonbonded_force(system)
    modified_exception_count = 0
    target_sigma = sigma_nm * unit.nanometer
    target_epsilon = epsilon_kj_per_mol * unit.kilojoules_per_mole
    for exception_index in range(nonbonded_force.getNumExceptions()):
        first_index, second_index, charge_product, _, _ = (
            nonbonded_force.getExceptionParameters(exception_index)
        )
        if (correct_probe_14_exceptions
                and {first_index, second_index} <= probe_carbon_indices
                and bond_distance(first_index, second_index) == 3):
            nonbonded_force.setExceptionParameters(
                exception_index,
                first_index,
                second_index,
                charge_product,
                target_sigma,
                target_epsilon / 2,
            )
            modified_exception_count += 1

    base_sigma_values = []
    base_epsilon_values = []
    for particle_index in range(system.getNumParticles()):
        _, particle_sigma, particle_epsilon = nonbonded_force.getParticleParameters(
            particle_index,
        )
        base_sigma_values.append(particle_sigma)
        base_epsilon_values.append(particle_epsilon)

    # 'probe_sigma' not 'sigma' to avoid collision with the per-dof 'sigma' in openmmtools integrators
    repulsion_force = CustomNonbondedForce(
        '(1-delta(probe_group1-probe_group2)) * ('
        '4 * probe_epsilon * ((probe_sigma / r)^12 - (probe_sigma / r)^6)'
        ' - 4 * sqrt(base_eps1 * base_eps2)'
        ' * ((0.5 * (base_sig1 + base_sig2) / r)^12'
        ' - (0.5 * (base_sig1 + base_sig2) / r)^6))'
    )
    repulsion_force.addGlobalParameter('probe_sigma', target_sigma)
    repulsion_force.addGlobalParameter('probe_epsilon', target_epsilon)
    repulsion_force.addPerParticleParameter('base_eps')
    repulsion_force.addPerParticleParameter('base_sig')
    repulsion_force.addPerParticleParameter('probe_group')

    for particle_index in range(system.getNumParticles()):
        repulsion_force.addParticle([
            base_epsilon_values[particle_index],
            base_sigma_values[particle_index],
            probe_group_by_particle[particle_index],
        ])

    repulsion_force.addInteractionGroup(probe_heavy_indices, probe_heavy_indices)
    repulsion_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    repulsion_force.setCutoffDistance(1.2 * unit.nanometer)

    # CUDA requires every nonbonded force to carry an identical exclusion set.
    # Same-residue probe pairs are suppressed by ``probe_group`` in the energy
    # expression instead of adding exclusions that are absent from the base
    # NonbondedForce.
    excluded_pairs = set()
    for exception_index in range(nonbonded_force.getNumExceptions()):
        first_index, second_index, *_ = nonbonded_force.getExceptionParameters(
            exception_index,
        )
        excluded_pairs.add(tuple(sorted((int(first_index), int(second_index)))))
    for first_index, second_index in sorted(excluded_pairs):
        repulsion_force.addExclusion(first_index, second_index)

    repulsion_force.setName(
        f'probe_repulsion_{PROBE_HAMILTONIAN_REVISION}_'
        f'{modified_exception_count}_exceptions',
    )
    system.addForce(repulsion_force)
    return {
        'revision': PROBE_HAMILTONIAN_REVISION,
        'probe_resnames': sorted(probe_resnames),
        'intermolecular_probe_repulsion': {
            'scope': 'all distinct probe residues, including different ligand types',
            'sigma_nm': sigma_nm,
            'epsilon_kj_per_mol': epsilon_kj_per_mol,
        },
        'modified_probe_14_exceptions': modified_exception_count,
        'intramolecular_probe_exclusions': sum(
            len(residue_indices) * (len(residue_indices) - 1) // 2
            for residue_indices in probe_heavy_indices_by_residue
        ),
    }


def add_contact_map_upper_wall(
        system: app.System,
        omm_model: app.Modeller,
        pocket_centre_nm: tuple[float, float, float] | None = None,
        min_dist_from_pocket_nm: float = 0.6,
        contact_cutoff_nm: float = 0.6,
        min_seq_separation: int = 5,
        at: float = 30.0,
        kappa_kj_per_mol: float = 500.0,
) -> None:
    """Add a PLUMED-style contact map upper wall to prevent protein unfolding.

    CV = Σ(s_i(r) - ref_i)^2 where s(r) = 1/(1+(r/r0)^6), r0=0.55 nm. Pairs are selected
    from all residues (H/E/C in simplified DSSP) using Timescapes atoms, then filtered by
    contact_cutoff_nm and min_seq_separation; pocket-proximal residues are excluded so the
    pocket can open. Wall fires when CV > at.
    """
    positions_nm = np.array(omm_model.positions.value_in_unit(unit.nanometer))
    pocket_centre = np.array(pocket_centre_nm) if pocket_centre_nm is not None else None

    md_top = mdtraj.Topology.from_openmm(omm_model.topology)
    dssp = mdtraj.compute_dssp(mdtraj.Trajectory(positions_nm[np.newaxis], md_top), simplified=True)[0]

    atom_to_res_idx = {a.index: a.residue.index for a in omm_model.topology.atoms()}

    rep_atom_indices, rep_positions = _select_representative_atoms(
        omm_model.topology, positions_nm, dssp, pocket_centre, min_dist_from_pocket_nm,
    )

    if len(rep_atom_indices) < 2:
        raise ValueError(
            'Fewer than 2 representative atoms found for the contact map. '
            'Check secondary structure assignment and pocket centre parameters.'
        )

    bond_pairs = _select_contact_pairs(
        rep_atom_indices, np.array(rep_positions), atom_to_res_idx,
        min_seq_separation, contact_cutoff_nm,
    )

    if not bond_pairs:
        raise ValueError(
            'No contact pairs found. Try increasing contact_cutoff_nm or '
            'reducing min_dist_from_pocket_nm.'
        )

    r0_qty = 0.55 * unit.nanometer
    cmap_force = CustomBondForce('(1 / (1 + (r / r0)^6) - ref)^2')
    cmap_force.addPerBondParameter('r0')
    cmap_force.addPerBondParameter('ref')
    for atom_i, atom_j, ref in bond_pairs:
        cmap_force.addBond(atom_i, atom_j, [r0_qty, ref])

    upper_wall = CustomCVForce('kappa * step(cmap - at) * (cmap - at)^2')
    upper_wall.addCollectiveVariable('cmap', cmap_force)
    # 'at' is dimensionless — attaching length units would make OpenMM store it in nm
    # and fire the wall ~10x earlier than the PLUMED AT=30 reference intends.
    upper_wall.addGlobalParameter('at', at)
    upper_wall.addGlobalParameter('kappa', kappa_kj_per_mol * unit.kilojoules_per_mole)
    upper_wall.setForceGroup(PLUMED_GEOMETRIC_BIAS_FORCE_GROUP)
    system.addForce(upper_wall)


def _select_representative_atoms(
        topology: app.Topology,
        positions_nm: np.ndarray,
        dssp: np.ndarray,
        pocket_centre: np.ndarray | None,
        min_dist_from_pocket_nm: float,
) -> tuple[list[int], list[np.ndarray]]:
    """Return Timescapes atom indices and positions for all assigned residues, skipping pocket-proximal.
    """
    atom_indices: list[int] = []
    atom_positions: list[np.ndarray] = []

    for res in topology.residues():
        target_atom_name = TIMESCAPES_ATOMS.get(res.name)
        if target_atom_name is None:
            continue
        if dssp[res.index] not in ('H', 'E', 'C'):
            continue
        for atom in res.atoms():
            if atom.name != target_atom_name:
                continue
            pos = positions_nm[atom.index]
            if pocket_centre is not None and np.linalg.norm(pos - pocket_centre) < min_dist_from_pocket_nm:
                break
            atom_indices.append(atom.index)
            atom_positions.append(pos)
            break  # one representative atom per residue

    return atom_indices, atom_positions


def _select_contact_pairs(
        atom_indices: list[int],
        positions: np.ndarray,
        atom_to_res_idx: dict[int, int],
        min_seq_separation: int,
        contact_cutoff_nm: float,
        r0_nm: float = 0.55,
) -> list[tuple[int, int, float]]:
    """Return (atom_i, atom_j, ref) for pairs within contact_cutoff_nm and >= min_seq_separation apart."""
    pairs = []
    for (idx_a, atom_i), (idx_b, atom_j) in itertools.combinations(enumerate(atom_indices), 2):
        if abs(atom_to_res_idx[atom_i] - atom_to_res_idx[atom_j]) < min_seq_separation:
            continue
        dist = np.linalg.norm(positions[idx_a] - positions[idx_b])
        if dist >= contact_cutoff_nm:
            continue
        ref = 1.0 / (1.0 + (dist / r0_nm) ** 6)
        pairs.append((atom_i, atom_j, ref))
    return pairs


def _get_swish_params(
        base_system: app.System,
        omm_model: app.Modeller,
        probe_resname: str,
        n_replicas: int,
        scale_max: float,
        forcefield_name: str | None,
        forcefield: app.ForceField | None,
) -> tuple[list[float], list[int], list[int], np.ndarray, np.ndarray]:
    """Return (scale_factors, water_ox_indices, apolar_indices, eps_vals, sig_vals) for SWISH correction."""
    scale_factors = list(np.linspace(1.0, scale_max, n_replicas).round(3))
    water_ox_indices, apolar_indices = _get_swish_interaction_indices(
        topology=omm_model.topology,
        probe_resname=probe_resname,
        forcefield_name=forcefield_name,
        forcefield=forcefield,
    )

    nonbonded_force = _get_nonbonded_force(base_system)
    n_particles = base_system.getNumParticles()
    eps_vals = np.zeros(n_particles)
    sig_vals = np.zeros(n_particles)
    for idx in water_ox_indices + apolar_indices:
        _, sig, eps = nonbonded_force.getParticleParameters(idx)
        eps_vals[idx] = eps.value_in_unit(unit.kilojoules_per_mole)
        sig_vals[idx] = sig.value_in_unit(unit.nanometer)

    return scale_factors, water_ox_indices, apolar_indices, eps_vals, sig_vals


def create_static_swish_systems(
        base_system: app.System,
        omm_model: app.Modeller,
        probe_resname: str = PROBE_RESNAME,
        n_replicas: int = 6,
        scale_max: float = 1.35,
        forcefield_name: str | None = None,
        forcefield: app.ForceField | None = None,
) -> list[app.System]:
    """Return N deep-copied Systems with lambda baked in as a constant for per-replica NPT equilibration."""
    scale_factors, water_ox_indices, apolar_indices, eps_vals, sig_vals = _get_swish_params(
        base_system, omm_model, probe_resname, n_replicas, scale_max,
        forcefield_name, forcefield,
    )

    systems = []
    for lam in scale_factors:
        replica_system = copy.deepcopy(base_system)
        correction = _build_swish_correction_force(
            replica_system, water_ox_indices, apolar_indices, eps_vals, sig_vals, initial_lam=lam,
        )
        replica_system.addForce(correction)
        systems.append(replica_system)

    return systems


def create_swish_system(
        base_system: app.System,
        omm_model: app.Modeller,
        probe_resname: str = PROBE_RESNAME,
        n_replicas: int = 6,
        scale_max: float = 1.35,
        forcefield_name: str | None = None,
        forcefield: app.ForceField | None = None,
) -> tuple[app.System, list[float]]:
    """Return (swish_system, scale_factors) with lam as a free global parameter for SWISH HREMD.

    All replicas share one CUDA context via CompoundThermodynamicState + SwishLambdaState.
    """
    scale_factors, water_ox_indices, apolar_indices, eps_vals, sig_vals = _get_swish_params(
        base_system, omm_model, probe_resname, n_replicas, scale_max,
        forcefield_name, forcefield,
    )

    swish_system = copy.deepcopy(base_system)
    correction = _build_swish_correction_force(
        swish_system, water_ox_indices, apolar_indices, eps_vals, sig_vals, initial_lam=1.0,
    )
    swish_system.addForce(correction)
    return swish_system, scale_factors


def minimise_equilibrate_npt_sandcat(
        system: app.System,
        topology: app.Topology,
        positions: unit.Quantity,
        timestep: unit.Quantity = 4 * unit.femtosecond,
        equil_time: unit.Quantity = 10 * unit.nanosecond,
        additional_solvent_resnames: Sequence[str] = (),
) -> tuple[unit.Quantity, unit.Quantity]:
    """Minimise and NPT-equilibrate in two phases: restrained, then unrestrained.

    ``equil_time`` is the TOTAL equilibration time, split 50/50. The first half is
    NPT with protein heavy-atom position restraints (solvent/probe relax around a
    fixed protein); the restraints are then released and the second half is
    unrestrained NPT so the protein relaxes into its (lambda-scaled) equilibrium
    conformation *before* production.

    Restraints and barostat are removed from ``system`` in-place. Returns
    (positions, box).

    Args:
        system: System to minimise and equilibrate.
        topology: Topology corresponding to the system.
        positions: Initial positions.
        timestep: Integration timestep.
        equil_time: Total equilibration time split across both phases.
        additional_solvent_resnames: Additional non-protein residue names that
            must not receive protein position restraints.

    Returns:
        Equilibrated positions and periodic box vectors.
    """
    solvent_resnames = _SOLVENT_RESNAMES | set(additional_solvent_resnames)
    protein_heavy_idx = [
        atom.index for atom in topology.atoms()
        if atom.residue.name not in solvent_resnames
        and atom.element is not None
        and atom.element.symbol != 'H'
    ]

    # Barostat is added before the restraint so the restraint holds the higher
    # force index; releasing it after phase 1 then leaves barostat_idx valid.
    barostat_frequency = get_n_steps(1 * unit.picosecond, timestep)
    barostat = MonteCarloBarostat(PRESSURE, TEMPERATURE, barostat_frequency)
    barostat_idx = system.addForce(barostat)

    restraint_idx = add_position_restraints(system, positions, protein_heavy_idx)

    integrator = LangevinMiddleIntegrator(TEMPERATURE, FRICTION, timestep)
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        Platform.getPlatformByName('CUDA'),
        {'Precision': 'mixed'},
    )

    simulation.context.setPositions(positions)
    simulation.context.computeVirtualSites()  # no-op for TIP3P; needed for TIP4P M-sites

    simulation.minimizeEnergy()

    # Split the total equilibration 50/50 between restrained and unrestrained NPT.
    total_steps = get_n_steps(equil_time, timestep)
    restrained_steps = total_steps // 2
    unrestrained_steps = total_steps - restrained_steps

    # Phase 1: restrained NPT (protein heavy atoms pinned).
    simulation.step(restrained_steps)

    # Phase 2: unrestrained NPT. Release the protein restraints and let it settle
    # into its equilibrium conformation so production does not have to.
    system.removeForce(restraint_idx)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(unrestrained_steps)

    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)

    # The restraint was already removed in phase 2; barostat_idx is still valid
    # because the restraint held the higher index.
    system.removeForce(barostat_idx)

    return state.getPositions(), state.getPeriodicBoxVectors()


def _validate_replica_particle_counts(
        system: app.System,
        equil_positions: Sequence[unit.Quantity],
        equil_boxes: Sequence[unit.Quantity],
        n_replicas: int,
) -> None:
    """Validate replica-state dimensions before creating OpenMMTools storage.

    Args:
        system: Shared production System.
        equil_positions: Equilibrated coordinates for each replica.
        equil_boxes: Equilibrated periodic box vectors for each replica.
        n_replicas: Expected number of replicas.

    Raises:
        ValueError: If replica counts or particle counts do not match the System.
    """
    if len(equil_positions) != n_replicas or len(equil_boxes) != n_replicas:
        raise ValueError(
            'Replica state count does not match the Hamiltonian ladder: '
            f'expected {n_replicas}, got {len(equil_positions)} position sets '
            f'and {len(equil_boxes)} box sets.'
        )

    expected_particles = system.getNumParticles()
    replica_particles = [len(positions) for positions in equil_positions]
    if any(count != expected_particles for count in replica_particles):
        raise ValueError(
            'Equilibrated replicas do not match the production System particle count: '
            f'expected {expected_particles}, got {replica_particles}. '
            'All MPI ranks must construct an identical mixed-solvent system.'
        )


# Cannot stress enough how important this class is to getting MPI to run
# SWISH - do NOT remove it
class SwishLambdaState(states.GlobalParameterState):
    """Composable state controlling the lam SWISH water-apolar epsilon scaling parameter."""
    lam = states.GlobalParameterState.GlobalParameter('lam', standard_value=1.0)

def _gmx_deo_offset(boundary_index: int) -> int:
    """Return the native GMX alternating-neighbour partition for boundary q.

    GMX's first exchange boundary (q=1) proposes (0,1),(2,3),...; q=2
    proposes (1,2),(3,4),....  The absolute boundary index makes restarts
    deterministic instead of resetting the phase at the storage iteration.
    """
    if boundary_index < 1:
        raise ValueError('boundary_index must be positive')
    return 1 - (int(boundary_index) % 2)


def _gmx_boundary_reduced_energy_matrix(
        engine_energy_kj: np.ndarray,
        geometric_energy_kj: np.ndarray,
        resident_state_ids: np.ndarray,
        resident_u_kj: np.ndarray,
        resident_pre_bias_kj: np.ndarray,
        opes_states: list[OpesExpanded | None],
        thermodynamic_states: list,
        volumes_nm3: np.ndarray,
        area_xy_nm2: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the retained-ENERGY GMX/PLUMED exchange matrix.

    ``engine_energy_kj[r,k]`` and ``geometric_energy_kj[r,k]`` are evaluated
    at the retained boundary coordinate x_r.  The off-diagonal OPES term is
    evaluated using the target state k's *post-update* estimator at the
    resident engine ENERGY U_s(x_r); the diagonal retains the pre-update
    resident value sent to GREX.  The returned matrix is reduced (kT units)
    and is intentionally separate from the normal post-step reporter matrix.
    """
    engine = np.asarray(engine_energy_kj, dtype=float)
    geometric = np.asarray(geometric_energy_kj, dtype=float)
    resident_ids = np.asarray(resident_state_ids, dtype=int)
    resident_u = np.asarray(resident_u_kj, dtype=float)
    pre_bias = np.asarray(resident_pre_bias_kj, dtype=float)
    volumes = np.asarray(volumes_nm3, dtype=float)
    if engine.ndim != 2 or engine.shape[0] != len(resident_ids):
        raise ValueError('boundary engine-energy matrix has inconsistent shape')
    if geometric.shape != engine.shape:
        raise ValueError('boundary geometric-energy matrix shape mismatch')
    if not (np.isfinite(engine).all() and np.isfinite(geometric).all()
            and np.isfinite(resident_u).all() and np.isfinite(pre_bias).all()):
        raise FloatingPointError('non-finite GMX boundary energy')
    n_replicas, n_states = engine.shape
    reduced = np.empty_like(engine)
    for replica_index in range(n_replicas):
        for state_index in range(n_states):
            if state_index == resident_ids[replica_index]:
                bias = pre_bias[replica_index]
            else:
                opes = opes_states[state_index]
                bias = (
                    0.0 if opes is None
                    else opes.compute_bias_v(resident_u[replica_index])
                )
            if not np.isfinite(bias):
                raise FloatingPointError('non-finite GMX boundary OPES bias')
            total = (
                engine[replica_index, state_index]
                + geometric[replica_index, state_index]
                + bias
            ) * unit.kilojoule_per_mole
            area = None
            if area_xy_nm2 is not None:
                area = float(area_xy_nm2[replica_index]) * unit.nanometer ** 2
            reduced[replica_index, state_index] = float(
                thermodynamic_states[state_index]._compute_reduced_potential(
                total, thermodynamic_states[state_index].temperature,
                volumes[replica_index] * unit.nanometer ** 3,
                thermodynamic_states[state_index].pressure,
                area,
                thermodynamic_states[state_index].surface_tension,
            ))
    if not np.isfinite(reduced).all():
        raise FloatingPointError('non-finite reduced GMX boundary criterion')
    return reduced


def _format_plumed_colvar_row(row: Sequence[float]) -> str:
    """Format one COLVAR row like PLUMED ``PRINT FMT=%8.4f`` output."""
    if not row:
        raise ValueError('COLVAR rows must contain at least a time value.')
    time_ps, *values = row
    return f'{float(time_ps):9.6f}' + ''.join(
        f' {float(value):8.4f}' for value in values
    )


def _format_plumed_general_row(values: Sequence[float]) -> str:
    """Format a PLUMED row using its 16-significant-digit general form."""
    return ' '.join(f'{float(value):23.16g}' for value in values)


class OpesReplicaExchangeSampler(multistate.ReplicaExchangeSampler):
    """Replica-exchange sampler that persists per-state OPES_EXPANDED bias to .nc storage.

    openmmtools serialises thermodynamic states only at creation, so OPES accumulators
    mutate but never reach disk via the standard path. This subclass gathers bias from
    all MPI ranks after propagation, writes it as a sidecar dict each iteration, and
    restores it on from_storage(). No-ops for plain-SWISH .nc files.
    """

    OPES_BIAS_PATH = 'opes_bias_states'

    def set_exchange_diagnostics(self, enabled: bool) -> None:
        """Enable or disable component-level JSONL exchange diagnostics."""
        self._exchange_diagnostics_enabled = bool(enabled)

    def _exchange_diagnostics_are_enabled(self) -> bool:
        """Return whether component-level exchange diagnostics are enabled."""
        return getattr(self, '_exchange_diagnostics_enabled', False)

    def set_colvar_reporter(
            self, output_prefix: str, interval_ps: float | None,
            flush_rows: int = 100,
    ) -> None:
        """Configure buffered, per-state PLUMED-style COLVAR output."""
        if interval_ps is not None and interval_ps <= 0:
            raise ValueError('COLVAR interval must be positive or None.')
        if flush_rows <= 0:
            raise ValueError('COLVAR flush_rows must be positive.')
        self._colvar_output_prefix = output_prefix
        self._colvar_interval_ps = interval_ps
        self._colvar_flush_rows = int(flush_rows)
        self._colvar_buffers = {state_id: [] for state_id in range(self.n_replicas)}
        communicator = mpiplus.get_mpicomm()
        if communicator is None or communicator.rank == 0:
            self._truncate_colvar_files()

    def _append_colvar_rows(self, state_id: int, rows: list[tuple]) -> None:
        if not rows or getattr(self, '_colvar_interval_ps', None) is None:
            return
        buffer = self._colvar_buffers[state_id]
        buffer.extend(rows)
        if len(buffer) >= self._colvar_flush_rows:
            self._flush_colvar(state_id)

    def _flush_colvar(self, state_id: int) -> None:
        buffer = getattr(self, '_colvar_buffers', {}).get(state_id, [])
        if not buffer:
            return
        path = f'{self._colvar_output_prefix}_COLVAR.{state_id}'
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a') as handle:
            if write_header:
                handle.write('#! FIELDS time cmap uwall.bias ene opesX.bias\n')
                handle.write(f'#! SET print_stride {self._colvar_interval_ps:.9g}\n')
            for row in buffer:
                handle.write(_format_plumed_colvar_row(row) + '\n')
        buffer.clear()

    def _flush_all_colvars(self) -> None:
        for state_id in getattr(self, '_colvar_buffers', {}):
            self._flush_colvar(state_id)

    def _truncate_colvar_files(self) -> None:
        """Discard text rows newer than the restored integration clock."""
        if getattr(self, '_colvar_interval_ps', None) is None:
            return
        timestep_ps = None
        for move in self._mcmc_moves:
            if isinstance(move, OpesExpandedDynamicsMove):
                timestep_ps = move.timestep.value_in_unit(unit.picosecond)
                cutoff_ps = move._gmx_step * timestep_ps
                break
        if timestep_ps is None:
            return
        for state_id in range(self.n_replicas):
            path = f'{self._colvar_output_prefix}_COLVAR.{state_id}'
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                lines = handle.readlines()
            kept = []
            for line in lines:
                if line.startswith('#!') or not line.strip():
                    kept.append(line)
                    continue
                try:
                    if float(line.split(None, 1)[0]) <= cutoff_ps + 1e-9:
                        kept.append(line)
                except ValueError:
                    continue
            with open(path, 'w') as handle:
                handle.writelines(kept)

    def set_iter_length_ps(self, iter_length: unit.Quantity) -> None:
        """Record the replica-exchange iteration length used for ΔF log time stamps.

        The .nc storage does not persist this, so production / restart helpers set
        it explicitly; it only affects the time column written to the DeltaFs logs.
        """
        self._iter_length_ps_value = iter_length.value_in_unit(unit.picosecond)

    def _iter_length_ps(self) -> float:
        """MD time per iteration in ps; falls back to the SWISH default when unset."""
        value = getattr(self, '_iter_length_ps_value', None)
        if value is None:
            return SWISH_ITER_LENGTH.value_in_unit(unit.picosecond)
        return value

    @staticmethod
    def _find_opes(thermo_state: Any) -> OpesExpanded | None:
        for component in getattr(thermo_state, '_composable_states', []):
            if isinstance(component, OpesExpanded):
                return component
        return None

    def _has_opes_states(self) -> bool:
        return any(self._find_opes(s) is not None for s in self._thermodynamic_states)

    def run(self, n_iterations: int | None = None) -> None:
        """Run with the GMX staggered lifecycle whenever OPES is present.

        Plain SWISH continues through the openmmtools implementation.  OPES
        SWISH-X must propagate to x(t), form the retained-energy criterion,
        perform the old-force boundary step, and only then mix the mapping.
        """
        if not self._has_opes_states():
            return super().run(n_iterations=n_iterations)
        opes_moves = [move for move in self._mcmc_moves
                      if isinstance(move, OpesExpandedDynamicsMove)]
        if len(opes_moves) != self.n_replicas:
            raise RuntimeError('GMX-staggered OPES requires one native move per replica')
        if len({move.n_steps for move in opes_moves}) != 1 or len({move.pace for move in opes_moves}) != 1:
            raise RuntimeError('GMX-staggered OPES requires identical interval and PACE on every replica')
        if sorted(np.asarray(self._replica_thermodynamic_states, dtype=int)) != list(range(self.n_replicas)):
            raise RuntimeError('Replica/state mapping is not a permutation')
        if self._iteration == 0:
            self._compute_energies()
            mpiplus.run_single_node(
                0, self._reporter.write_energies, self._energy_thermodynamic_states,
                self._neighborhoods, self._energy_unsampled_states, self._iteration,
            )
            self._check_nan_energy()
        if n_iterations is None:
            iteration_limit = self.number_of_iterations
        else:
            iteration_limit = min(self._iteration + n_iterations, self.number_of_iterations)
        try:
            while not self._is_completed(iteration_limit):
                self._iteration += 1
                self._propagate_gmx_replicas()
                self._sync_opes_state_across_ranks()
                self._gmx_boundary_energy_matrix = _gmx_boundary_reduced_energy_matrix(
                    self._gmx_boundary_engine_energy,
                    self._gmx_boundary_geometric_energy,
                    self._gmx_boundary_resident_state_ids,
                    self._gmx_boundary_resident_u,
                    self._gmx_boundary_resident_pre_bias,
                    [self._find_opes(s) for s in self._thermodynamic_states],
                    self._thermodynamic_states,
                    self._gmx_boundary_volumes,
                    getattr(self, '_gmx_boundary_area_xy', None),
                )
                if self._exchange_diagnostics_are_enabled():
                    self._gmx_boundary_physical_energy_matrix = (
                        _gmx_boundary_reduced_energy_matrix(
                            self._gmx_boundary_engine_energy,
                            self._gmx_boundary_geometric_energy,
                            self._gmx_boundary_resident_state_ids,
                            self._gmx_boundary_resident_u,
                            np.zeros(self.n_replicas),
                            [None] * self.n_replicas,
                            self._thermodynamic_states,
                            self._gmx_boundary_volumes,
                            getattr(self, '_gmx_boundary_area_xy', None),
                        )
                    )
                self._replica_thermodynamic_states = self._mix_replicas()
                self._compute_energies()
                self._report_iteration()
                self._update_analysis()
                self._check_nan_energy()
        finally:
            self._flush_all_colvars()

    def _propagate_gmx_replica(self, replica_id: int) -> dict[str, Any]:
        state_id = int(self._replica_thermodynamic_states[replica_id])
        thermo = self._thermodynamic_states[state_id]
        move = self._mcmc_moves[state_id]
        sampler_state = self._sampler_states[replica_id]
        boundary_step = int(self._iteration * move.n_steps)
        boundary_state, local = move.apply_gmx_staggered(
            thermo, sampler_state, boundary_step=boundary_step,
            context_cache=self.sampler_context_cache,
        )
        # Evaluate only the small boundary observables on the worker.  The
        # full-precision x(t) never needs to cross MPI; x(t+dt) is returned by
        # the normal sampler-state serialization below.
        context, _ = self.energy_context_cache.get_context(self._thermodynamic_states[0])
        boundary_state.apply_to_context(context, ignore_velocities=True)
        engine_row = np.zeros(self.n_replicas, dtype=float)
        geometric_row = np.zeros(self.n_replicas, dtype=float)
        for target_id, target in enumerate(self._thermodynamic_states):
            target.apply_to_context(context)
            engine_row[target_id] = _read_u_ff_kj(context)
            geometric_row[target_id] = _read_geometric_bias_kj(context)
        # Keep the value read before the boundary step on the diagonal.
        engine_row[state_id] = local['resident_u_kj']
        geometric_row[state_id] = local['resident_geometric_kj']
        box = boundary_state.box_vectors
        volume = float(boundary_state.volume.value_in_unit(unit.nanometer ** 3))
        area = None
        if box is not None:
            vec = np.asarray(box.value_in_unit(unit.nanometer))
            area = float(np.linalg.norm(np.cross(vec[0], vec[1])))
        return {
            'sampler_state': sampler_state.__getstate__(ignore_velocities=False),
            'state_id': state_id,
            'engine_row': engine_row,
            'geometric_row': geometric_row,
            'resident_u_kj': local['resident_u_kj'],
            'resident_pre_bias_kj': local['resident_pre_bias_kj'],
            'volume_nm3': volume,
            'area_xy_nm2': area,
            'gmx_step': getattr(move, '_gmx_step', None),
            'colvar_rows': local.get('colvar_rows', []),
        }

    def _propagate_gmx_replicas(self) -> None:
        results, replica_ids = mpiplus.distribute(
            self._propagate_gmx_replica, range(self.n_replicas), send_results_to=0,
        )
        self._gmx_boundary_engine_energy = np.zeros((self.n_replicas, self.n_replicas))
        self._gmx_boundary_geometric_energy = np.zeros_like(self._gmx_boundary_engine_energy)
        self._gmx_boundary_resident_state_ids = np.zeros(self.n_replicas, dtype=int)
        self._gmx_boundary_resident_u = np.zeros(self.n_replicas)
        self._gmx_boundary_resident_pre_bias = np.zeros(self.n_replicas)
        self._gmx_boundary_volumes = np.zeros(self.n_replicas)
        self._gmx_boundary_area_xy = np.zeros(self.n_replicas)
        for replica_id, result in zip(replica_ids, results):
            self._sampler_states[replica_id].__setstate__(result['sampler_state'])
            self._gmx_boundary_engine_energy[replica_id] = result['engine_row']
            self._gmx_boundary_geometric_energy[replica_id] = result['geometric_row']
            self._gmx_boundary_resident_state_ids[replica_id] = result['state_id']
            self._gmx_boundary_resident_u[replica_id] = result['resident_u_kj']
            self._gmx_boundary_resident_pre_bias[replica_id] = result['resident_pre_bias_kj']
            self._gmx_boundary_volumes[replica_id] = result['volume_nm3']
            self._gmx_boundary_area_xy[replica_id] = result['area_xy_nm2'] or 0.0
            self._append_colvar_rows(result['state_id'], result.get('colvar_rows', []))
        if not np.isfinite(self._gmx_boundary_engine_energy).all():
            raise FloatingPointError('non-finite GMX boundary engine energy')

    def _propagate_replicas(self) -> None:
        super()._propagate_replicas()
        if self._has_opes_states():
            self._sync_opes_state_across_ranks()

    def _mix_neighboring_replicas(self) -> None:
        """Attempt neighbour swaps, alternating the even/odd partition deterministically.

        openmmtools' base implementation selects the partition offset at random each
        iteration (``offset = np.random.randint(2)``), so the same partition (even
        ``{(0,1),(2,3),...}`` or odd ``{(1,2),(3,4),...}``) recurs on ~50% of consecutive
        iterations. When it does, a replica that has just swapped up the ladder is offered
        the same pair again and swaps straight back down: walker motion becomes
        anti-persistent (observed reversal probability ~0.8 on TEM1 SWISH-X) and net
        transport along the lambda ladder is throttled.

        GROMACS ``-hrex`` instead alternates the two partitions deterministically (DEO), so
        the pair a walker just used is never offered on the following attempt, immediate
        back-tracking is impossible, and transport is persistent (~2.6x more end-to-end
        round trips per walker at matched per-swap acceptance). SandCat therefore always
        uses DEO, keyed to restart-safe iteration parity. This changes only the swap
        schedule, not the Metropolis acceptance criterion.
        """
        # Staged OPES has a q-boundary matrix and uses the GMX DEO phase;
        # otherwise fall back to plain iteration parity.
        if getattr(self, '_gmx_boundary_energy_matrix', None) is None:
            offset = self._iteration % 2
        else:
            offset = _gmx_deo_offset(self._iteration)
        for thermodynamic_state_i in range(offset, self.n_replicas - 1, 2):
            thermodynamic_state_j = thermodynamic_state_i + 1
            replica_i = np.where(self._replica_thermodynamic_states == thermodynamic_state_i)
            replica_j = np.where(self._replica_thermodynamic_states == thermodynamic_state_j)
            self._attempt_swap(replica_i, replica_j)

    def _attempt_swap(self, replica_i: Any, replica_j: Any) -> None:
        """Use the retained x(t) criterion for OPES; reporter energies stay post-step."""
        matrix = getattr(self, '_gmx_boundary_energy_matrix', None)
        if matrix is None:
            return super()._attempt_swap(replica_i, replica_j)
        replica_i_id = int(np.asarray(replica_i).ravel()[0])
        replica_j_id = int(np.asarray(replica_j).ravel()[0])
        state_i = int(self._replica_thermodynamic_states[replica_i_id])
        state_j = int(self._replica_thermodynamic_states[replica_j_id])
        log_p_accept = -(
            matrix[replica_i_id, state_j] + matrix[replica_j_id, state_i]
        )
        log_p_accept += (
            matrix[replica_i_id, state_i] + matrix[replica_j_id, state_j]
        )
        acceptance_probability = (
            1.0 if log_p_accept >= 0.0 else math.exp(log_p_accept)
        )
        self._n_proposed_matrix[state_i, state_j] += 1
        self._n_proposed_matrix[state_j, state_i] += 1
        accepted = log_p_accept >= 0.0 or np.random.rand() < acceptance_probability
        if accepted:
            self._replica_thermodynamic_states[replica_i_id] = state_j
            self._replica_thermodynamic_states[replica_j_id] = state_i
            self._n_accepted_matrix[state_i, state_j] += 1
            self._n_accepted_matrix[state_j, state_i] += 1
        if self._exchange_diagnostics_are_enabled():
            physical_matrix = self._gmx_boundary_physical_energy_matrix
            physical_difference = (
                physical_matrix[replica_i_id, state_j]
                + physical_matrix[replica_j_id, state_i]
                - physical_matrix[replica_i_id, state_i]
                - physical_matrix[replica_j_id, state_j]
            )
            total_difference = -float(log_p_accept)
            self._append_exchange_diagnostic({
                'iteration': int(self._iteration),
                'time_ps': self._iteration * self._iter_length_ps(),
                'state_pair': [state_i, state_j],
                'replica_pair': [replica_i_id, replica_j_id],
                'physical_reduced_potential_difference': float(physical_difference),
                'opes_reduced_potential_difference': float(
                    total_difference - physical_difference,
                ),
                'total_exchange_exponent': float(log_p_accept),
                'acceptance_probability': float(acceptance_probability),
                'accepted': bool(accepted),
                'exchange_bias_mode': 'gmx-staggered',
            })

    def _exchange_diagnostics_path(self) -> str | None:
        """Return the component-level exchange diagnostics path."""
        nc_path = getattr(self._reporter, 'filepath', None)
        if not nc_path:
            return None
        prefix = nc_path[:-3] if nc_path.endswith('.nc') else nc_path
        return f'{prefix}_exchange_diagnostics.jsonl'

    def _append_exchange_diagnostic(self, diagnostic: dict[str, Any]) -> None:
        """Append one component-level exchange diagnostic record."""
        path = self._exchange_diagnostics_path()
        if path is None:
            return
        with open(path, 'a') as diagnostic_file:
            diagnostic_file.write(json.dumps(diagnostic, sort_keys=True) + '\n')

    def _truncate_exchange_diagnostics_to_iteration(self) -> None:
        """Drop diagnostic rows newer than the restored sampler iteration."""
        path = self._exchange_diagnostics_path()
        if path is None or not os.path.exists(path):
            return
        kept_lines = []
        with open(path) as diagnostic_file:
            for line in diagnostic_file:
                try:
                    diagnostic = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(diagnostic.get('iteration', -1)) <= self._iteration:
                    kept_lines.append(json.dumps(diagnostic, sort_keys=True) + '\n')
        with open(path, 'w') as diagnostic_file:
            diagnostic_file.writelines(kept_lines)

    def _get_replica_opes_payload(
            self,
            replica_id: int,
    ) -> tuple[int, dict[str, Any], int | None, int | None] | None:
        """Return state/bias/lifecycle clocks for cross-rank synchronization."""
        state_id = int(self._replica_thermodynamic_states[replica_id])
        opes = self._find_opes(self._thermodynamic_states[state_id])
        if opes is None:
            return None
        move = self._mcmc_moves[state_id]
        return (
            state_id,
            opes.get_state(),
            getattr(move, '_cumulative_steps', None),
            getattr(move, '_gmx_step', None),
        )

    def _sync_opes_state_across_ranks(self) -> None:
        """Broadcast each rank's locally-updated OPES bias and cumulative steps to all ranks.

        Without this, rank 0 only has fresh bias for its own replica; stale bias on other
        ranks causes wrong PACE boundaries. mpiplus.distribute is a no-op when MPI is off.
        """
        results = mpiplus.distribute(
            self._get_replica_opes_payload, range(self.n_replicas),
            send_results_to='all',
        )
        for entry in results:
            if entry is None:
                continue
            state_id, bias_state, cumstep, gmx_step = entry
            opes = self._find_opes(self._thermodynamic_states[state_id])
            if opes is not None:
                opes.load_state(bias_state)
            move = self._mcmc_moves[state_id]
            if cumstep is not None and hasattr(move, '_cumulative_steps'):
                move._cumulative_steps = cumstep
            if gmx_step is not None and hasattr(move, '_gmx_step'):
                move._gmx_step = gmx_step

    def _report_iteration_items(self) -> None:
        super()._report_iteration_items()
        self._write_opes_bias_states()
        self._write_opes_deltaf_log()

    def _write_opes_bias_states(self) -> None:
        if not self._has_opes_states():
            return
        checkpoint_interval = getattr(self._reporter, 'checkpoint_interval', None)
        if (checkpoint_interval and self._iteration % int(checkpoint_interval) != 0):
            return
        bias_dict = {
            '__lifecycle_revision__': SWISH_X_LIFECYCLE_REVISION,
            '__checkpoint_iteration__': int(self._iteration),
        }
        for state_id, state in enumerate(self._thermodynamic_states):
            opes = self._find_opes(state)
            if opes is not None:
                move = self._mcmc_moves[state_id]
                bias_dict[f'state{state_id}'] = {
                    'opes': opes.get_state(),
                    'cumulative_steps': getattr(move, '_cumulative_steps', 0),
                    'gmx_step': getattr(move, '_gmx_step', 0),
                }
        if bias_dict:
            self._reporter.write_dict(self.OPES_BIAS_PATH, bias_dict)

    def _opes_deltaf_log_path(self, state_id: int) -> str | None:
        """PLUMED-compatible per-state DeltaFs log path, or None if no .nc path is available.

        Naming mirrors PLUMED's ``DeltaFs.{N}.data`` so analysis scripts that already
        parse PLUMED OPES_EXPANDED logs work unchanged on sandcat output.
        """
        nc_path = getattr(self._reporter, 'filepath', None)
        if not nc_path:
            return None
        prefix = nc_path[:-3] if nc_path.endswith('.nc') else nc_path
        return f'{prefix}_DeltaFs.{state_id}.data'

    def _write_opes_deltaf_log(self) -> None:
        """Append one per-state ΔF snapshot per iteration to PLUMED-format text logs.

        Header (written once when the file is empty):
            #! FIELDS time rct DeltaF_<T0> DeltaF_<T1> ... DeltaF_<TN-1>
            #! SET print_stride <iter_length_ps>
        Body line per iteration: ``<time_ps> <rct_kj> <ΔF0> <ΔF1> ... <ΔFN-1>``.

        Runs only after ``_sync_opes_state_across_ranks`` has gathered per-state OPES
        updates onto rank 0, so the logged ΔF reflects the freshest per-rank update for
        whichever state was propagated this iteration. States whose OPES bias is still
        in the unbiased observation phase are skipped (ΔF=0 by construction).
        """
        if not self._has_opes_states():
            return
        iter_ps = self._iter_length_ps()
        time_ps = self._iteration * iter_ps
        for state_id, state in enumerate(self._thermodynamic_states):
            opes = self._find_opes(state)
            if opes is None or not opes._initialized:
                continue
            path = self._opes_deltaf_log_path(state_id)
            if path is None:
                return
            write_header = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, 'a') as log_fh:
                if write_header:
                    temp_labels = ' '.join(
                        f'DeltaF_{temp_k:g}' for temp_k in opes._temps_k
                    )
                    log_fh.write(f'#! FIELDS time rct {temp_labels}\n')
                    log_fh.write(f'#! SET print_stride  {iter_ps:g}\n')
                log_fh.write(_format_plumed_general_row(
                    (time_ps, opes._rct_kj, *opes._deltaf_kj),
                ) + '\n')

    def _truncate_opes_deltaf_logs_to_iteration(self) -> None:
        """On restart, drop any per-state text-log rows past the restored iteration.

        Without this, a crashed run that wrote logs beyond the last .nc checkpoint will
        leave duplicate / inconsistent time rows after restart. Header lines (``#!``) are
        always preserved; body rows are kept only if their ``time`` column is at or
        before ``self._iteration * SWISH_ITER_LENGTH``.
        """
        if not self._has_opes_states():
            return
        iter_ps = self._iter_length_ps()
        cutoff_ps = self._iteration * iter_ps
        for state_id, _ in enumerate(self._thermodynamic_states):
            path = self._opes_deltaf_log_path(state_id)
            if path is None or not os.path.exists(path):
                continue
            with open(path, 'r') as log_fh:
                lines = log_fh.readlines()
            kept = []
            for line in lines:
                if line.startswith('#!') or not line.strip():
                    kept.append(line)
                    continue
                try:
                    row_time_ps = float(line.split(None, 1)[0])
                except ValueError:
                    continue  # malformed row; drop
                if row_time_ps <= cutoff_ps + 1e-6:
                    kept.append(line)
            with open(path, 'w') as log_fh:
                log_fh.writelines(kept)

    def _restore_sampler_from_reporter(self, reporter: Any) -> None:
        """Restore OPES bias from storage while the reporter is still open in 'r' mode.

        Must happen here because from_storage closes and reopens the reporter append-only
        on rank 0 after this call — read_dict on ranks 1..N-1 would fail afterwards.
        """
        super()._restore_sampler_from_reporter(reporter)
        try:
            bias_dict = reporter.read_dict(self.OPES_BIAS_PATH)
        except (KeyError, ValueError):
            if self._has_opes_states():
                raise RuntimeError('Missing OPES lifecycle checkpoint for SWISH-X restart')
            return
        if self._has_opes_states() and bias_dict.get('__lifecycle_revision__') != SWISH_X_LIFECYCLE_REVISION:
            raise RuntimeError(
                'Stored OPES state does not carry the GMX staggered lifecycle revision; '
                'refusing to mix a legacy SWISH-X restart.'
            )
        if self._has_opes_states() and int(bias_dict.get('__checkpoint_iteration__', -1)) != int(self._iteration):
            raise RuntimeError('OPES lifecycle checkpoint does not match restored sampler iteration')
        for state_id, state in enumerate(self._thermodynamic_states):
            entry = bias_dict.get(f'state{state_id}')
            if entry is None:
                continue
            opes = self._find_opes(state)
            if opes is not None:
                # Accept the old raw state only for plain/non-OPES legacy files;
                # staged OPES checkpoints always carry clocks alongside bias.
                opes.load_state(entry['opes'] if isinstance(entry, dict) and 'opes' in entry else entry)
            if isinstance(entry, dict):
                move = self._mcmc_moves[state_id]
                move._cumulative_steps = int(entry.get('cumulative_steps', move._cumulative_steps))
                move._gmx_step = int(entry.get('gmx_step', move._gmx_step))
        self._truncate_opes_deltaf_logs_to_iteration()
        self._truncate_exchange_diagnostics_to_iteration()


def _is_valid_nc(path: str) -> bool:
    """Return True if ``path`` opens cleanly as a netCDF4 dataset.

    Caller must export ``HDF5_USE_FILE_LOCKING=FALSE`` before the Python
    process starts when validating a snapshot of an openmmtools-held file —
    HDF5 reads the env var at library init and otherwise refuses the file.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with netCDF4.Dataset(path, 'r') as ds:
            list(ds.variables.keys())
    except (OSError, RuntimeError):
        return False
    return True


def _checkpoint_paths(output_prefix: str) -> tuple[str, str]:
    return f'{output_prefix}.nc', f'{output_prefix}_checkpoint.nc'


def backup_checkpoint(output_prefix: str = SWISH_OUTPUT_PREFIX) -> bool:
    """Snapshot the live .nc + _checkpoint.nc to ``.bak`` siblings if the copies validate.

    Byte-copies each live file (HDF5 locking does not block this), validates
    the copies, then atomically promotes them to ``.bak``. If either copy
    fails to open, drops both .tmp files and returns False so a previously
    good ``.bak`` is never overwritten with a corrupt snapshot.
    """
    storage_path, checkpoint_path = _checkpoint_paths(output_prefix)
    for path in (storage_path, checkpoint_path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False

    tmp_paths = [
        src + CHECKPOINT_BACKUP_SUFFIX + '.tmp'
        for src in (storage_path, checkpoint_path)
    ]
    for src, tmp in zip((storage_path, checkpoint_path), tmp_paths):
        shutil.copy2(src, tmp)

    if not all(_is_valid_nc(tmp) for tmp in tmp_paths):
        for tmp in tmp_paths:
            if os.path.exists(tmp):
                os.remove(tmp)
        return False

    for src, tmp in zip((storage_path, checkpoint_path), tmp_paths):
        os.replace(tmp, src + CHECKPOINT_BACKUP_SUFFIX)
    return True


def _restore_from_backup_if_corrupt(output_prefix: str) -> None:
    """Restore from ``.bak`` if the live file is missing, corrupt, or older.

    The bg sync uploads only .bak snapshots, so after a spot interruption the
    .bak on S3 can be newer than the live .nc — pick whichever valid file is
    more recent.
    """
    storage_path, checkpoint_path = _checkpoint_paths(output_prefix)
    for path in (storage_path, checkpoint_path):
        backup = path + CHECKPOINT_BACKUP_SUFFIX
        live_valid = _is_valid_nc(path)
        bak_valid = _is_valid_nc(backup)
        if not bak_valid:
            continue
        if not live_valid:
            logging.warning(f'Restoring {path} from {backup} (live file missing or corrupt).')
            shutil.copy2(backup, path)
        elif os.path.getmtime(backup) > os.path.getmtime(path):
            logging.info(f'Restoring {path} from {backup} (backup is newer).')
            shutil.copy2(backup, path)


def can_restart_swish_production(output_prefix: str = SWISH_OUTPUT_PREFIX) -> bool:
    """Return True if .nc and _checkpoint.nc (or their ``.bak`` siblings) are usable."""
    storage_path, checkpoint_path = _checkpoint_paths(output_prefix)
    for path in (storage_path, checkpoint_path):
        if _is_valid_nc(path):
            continue
        if _is_valid_nc(path + CHECKPOINT_BACKUP_SUFFIX):
            continue
        return False
    return True


def restart_swish_production(
        total_simtime_ns: float = 250.0,
        time_step_fs: int = 4,
        exchange_interval_ps: float | None = None,
        output_prefix: str = SWISH_OUTPUT_PREFIX,
        verbose: bool = False,
        colvar_interval_ps: float | None = 2.0,
) -> None:
    """Resume a SWISH replica exchange run from existing NC storage, extending to total_simtime_ns.

    ``exchange_interval_ps`` must match the value used for the original run; it
    controls how many MD steps make up one stored iteration. If None, the SWISH
    default of 10 ps is used.
    """
    if verbose and _is_primary_mpi_rank():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    storage_path = f'{output_prefix}.nc'
    _restore_from_backup_if_corrupt(output_prefix)
    if not can_restart_swish_production(output_prefix):
        raise FileNotFoundError(
            f'Cannot restart: {storage_path} or its checkpoint is missing or empty.'
        )

    iter_length = resolve_iter_length(exchange_interval_ps)
    timestep_qty = time_step_fs * unit.femtosecond
    steps_per_iteration = get_n_steps(iter_length, timestep_qty)
    total_steps = get_n_steps(total_simtime_ns * unit.nanosecond, timestep_qty)
    target_iterations = total_steps // steps_per_iteration

    simulation = OpesReplicaExchangeSampler.from_storage(storage_path)
    simulation.set_iter_length_ps(iter_length)
    simulation.set_exchange_diagnostics(verbose)
    colvar_stride_steps = None
    if colvar_interval_ps is not None:
        colvar_stride_steps = get_n_steps(
            colvar_interval_ps * unit.picosecond, timestep_qty,
        )
    for move in simulation._mcmc_moves:
        if isinstance(move, OpesExpandedDynamicsMove):
            move.set_colvar_stride(colvar_stride_steps)
    if simulation._has_opes_states():
        simulation.set_colvar_reporter(output_prefix, colvar_interval_ps)
    completed = simulation.iteration

    if completed >= target_iterations:
        logging.info(
            f'Restart: already at iteration {completed} >= target {target_iterations}, nothing to run.'
        )
        return

    remaining = target_iterations - completed
    logging.info(
        f'Restart: resuming from iteration {completed}/{target_iterations} '
        f'({remaining} iterations remaining).'
    )
    simulation.extend(n_iterations=remaining)


def run_swish_x_production(
        swish_system: app.System,
        scale_factors: list[float],
        equil_positions: list[unit.Quantity],
        equil_boxes: list[unit.Quantity],
        opes_t_max: unit.Quantity,
        opes_n_temps: int,
        opes_pace: int = 500,
        opes_observation_steps: int = 100,
        total_simtime_ns: float = 250.0,
        time_step_fs: int = 4,
        exchange_interval_ps: float | None = None,
        output_prefix: str = SWISH_X_OUTPUT_PREFIX,
        verbose: bool = False,
        use_barostat: bool = False,
        collision_freq: float = 1.0,
        base_seed: int | None = None,
        colvar_interval_ps: float | None = 2.0,
) -> None:
    """SWISH-X = SWISH × OPES-Multithermal (Borsatto et al, 2024).

    Per-replica OPES_EXPANDED bias targets [TEMPERATURE, opes_t_max]. The bias wrapper is a
    Force in the System, so it is included in the swap acceptance (openmmtools' reduced_potential
    sums all force groups), matching gmx+plumed -hrex.
    ``exchange_interval_ps`` sets the MD time between exchange attempts (default 10 ps).
    When ``use_barostat=True``, production runs NPT via a MonteCarloBarostat managed by the
    ThermodynamicState; all replicas share the same pressure target.
    When ``base_seed`` is set, replica ``i`` is given a fixed integrator seed ``base_seed + i``
    for a reproducible per-replica stochastic trajectory; None keeps OpenMM's OS-random seeding.
    OPES SWISH-X always uses the native GMX staggered boundary: retained
    x(t) energies and pre-update resident bias form the swap criterion, then
    one old-force step is integrated before the post-update estimator is
    installed.  This is deliberately the default lifecycle, not a diagnostic
    option.
    """

    if verbose and _is_primary_mpi_rank():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    n_replicas = len(scale_factors)
    _validate_replica_particle_counts(
        swish_system,
        equil_positions,
        equil_boxes,
        n_replicas,
    )
    if not unit.is_quantity(opes_t_max):
        opes_t_max = opes_t_max * unit.kelvin
    iter_length = resolve_iter_length(exchange_interval_ps)
    timestep_qty = time_step_fs * unit.femtosecond
    steps_per_iteration = get_n_steps(iter_length, timestep_qty)
    colvar_stride_steps = None
    if colvar_interval_ps is not None:
        colvar_stride_steps = get_n_steps(
            colvar_interval_ps * unit.picosecond, timestep_qty,
        )
        actual_interval_ps = (
            colvar_stride_steps * timestep_qty
        ).value_in_unit(unit.picosecond)
        if not math.isclose(actual_interval_ps, colvar_interval_ps, abs_tol=1e-12):
            raise ValueError(
                'COLVAR interval must be an integer multiple of the MD timestep.'
            )

    add_opes_expanded_bias(swish_system, n_temps=opes_n_temps)

    thermo_kwargs = {'system': swish_system, 'temperature': TEMPERATURE}
    if use_barostat:
        thermo_kwargs['pressure'] = PRESSURE
    base_thermo = states.ThermodynamicState(**thermo_kwargs)
    thermo_states = []
    for lam in scale_factors:
        opes_state = OpesExpanded(
            temperature=TEMPERATURE,
            t_min=TEMPERATURE,
            t_max=opes_t_max,
            n_temps=opes_n_temps,
            pace=opes_pace,
            observation_steps=opes_observation_steps,
        )
        thermo_states.append(states.CompoundThermodynamicState(
            thermodynamic_state=base_thermo,
            composable_states=[SwishLambdaState(lam=lam), opes_state],
        ))

    moves = [
        OpesExpandedDynamicsMove(
            timestep=timestep_qty,
            collision_rate=collision_freq / unit.picosecond,
            n_steps_per_iteration=steps_per_iteration,
            pace=opes_pace,
            n_temps=opes_n_temps,
            seed=(None if base_seed is None else base_seed + i),
        )
        for i in range(n_replicas)
    ]
    for move in moves:
        move.set_colvar_stride(colvar_stride_steps)

    sampler_states = [
        states.SamplerState(positions=pos, box_vectors=box)
        for pos, box in zip(equil_positions, equil_boxes)
    ]

    total_steps = get_n_steps(total_simtime_ns * unit.nanosecond, timestep_qty)
    n_iterations = total_steps // steps_per_iteration
    checkpoint_interval = get_n_steps(REPORT_INTERVAL, iter_length)

    reporter = multistate.MultiStateReporter(
        f'{output_prefix}.nc', checkpoint_interval=checkpoint_interval,
    )

    simulation = OpesReplicaExchangeSampler(
        mcmc_moves=moves, number_of_iterations=n_iterations,
        replica_mixing_scheme='swap-neighbors',
    )
    simulation.set_iter_length_ps(iter_length)
    simulation.set_exchange_diagnostics(verbose)

    simulation.create(
        thermodynamic_states=thermo_states,
        sampler_states=sampler_states,
        storage=reporter,
    )
    simulation.set_colvar_reporter(output_prefix, colvar_interval_ps)

    simulation.run()


def run_swish_production(
        swish_system: app.System,
        scale_factors: list[float],
        equil_positions: list[unit.Quantity],
        equil_boxes: list[unit.Quantity],
        total_simtime_ns: float = 250.0,
        time_step_fs: int = 4,
        exchange_interval_ps: float | None = None,
        output_prefix: str = SWISH_OUTPUT_PREFIX,
        verbose: bool = False,
        use_barostat: bool = False,
) -> None:
    """Run SWISH Hamiltonian replica exchange production.

    ``exchange_interval_ps`` sets the MD time between exchange attempts (default 10 ps).
    When ``use_barostat=True``, production runs NPT via a MonteCarloBarostat managed by the
    ThermodynamicState; all replicas share the same pressure target.
    """
    if verbose and _is_primary_mpi_rank():
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    n_replicas = len(scale_factors)
    _validate_replica_particle_counts(
        swish_system,
        equil_positions,
        equil_boxes,
        n_replicas,
    )
    iter_length = resolve_iter_length(exchange_interval_ps)
    timestep_qty = time_step_fs * unit.femtosecond
    steps_per_iteration = get_n_steps(iter_length, timestep_qty)

    move = mcmc.LangevinDynamicsMove(
        timestep=timestep_qty,
        collision_rate=FRICTION,
        n_steps=steps_per_iteration,
    )

    thermo_kwargs = {'system': swish_system, 'temperature': TEMPERATURE}
    if use_barostat:
        thermo_kwargs['pressure'] = PRESSURE
    base_thermo = states.ThermodynamicState(**thermo_kwargs)
    thermo_states = [
        states.CompoundThermodynamicState(
            thermodynamic_state=base_thermo,
            composable_states=[SwishLambdaState(lam=lam)],
        )
        for lam in scale_factors
    ]

    sampler_states = [
        states.SamplerState(positions=pos, box_vectors=box)
        for pos, box in zip(equil_positions, equil_boxes)
    ]

    total_steps = get_n_steps(total_simtime_ns * unit.nanosecond, timestep_qty)
    n_iterations = total_steps // steps_per_iteration
    checkpoint_interval = get_n_steps(REPORT_INTERVAL, iter_length)

    reporter = multistate.MultiStateReporter(
        f'{output_prefix}.nc', checkpoint_interval=checkpoint_interval,
    )

    # Use OpesReplicaExchangeSampler (a no-op superset for OPES-free systems) with
    # 'swap-neighbors' so plain SWISH mixes like SWISH-X and the GROMACS -hrex reference:
    # deterministic even/odd neighbour alternation (see _mix_neighboring_replicas). This
    # also matches the restart path, which reloads plain-SWISH storage via this class.
    simulation = OpesReplicaExchangeSampler(
        mcmc_moves=move, number_of_iterations=n_iterations,
        replica_mixing_scheme='swap-neighbors',
    )
    simulation.set_iter_length_ps(iter_length)
    simulation.set_exchange_diagnostics(verbose)

    simulation.create(
        thermodynamic_states=thermo_states,
        sampler_states=sampler_states,
        storage=reporter,
    )

    simulation.run()
