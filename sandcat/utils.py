from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from openmm import System, app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator, SystemGenerator
from scipy.spatial import cKDTree

from sandcat.ligand import LigFile
from sandcat.ligand_io import parse_ligfile_openff
from sandcat.system_setup import add_custom_metal_bonds
from sandcat.system_setup import check_for_espaloma_charges
from sandcat.system_setup import parse_box_info_json_file
from sandcat.variables import (
    DESAMBER_BENZENE_FORCEFIELD,
    FORCEFIELD_OPTIONS,
    FORCEFIELDS,
    PROBE_RESNAME,
    PROBE_SDF_MAP,
    WATER_VOLUME_L,
    uses_template_probe,
)


DEFAULT_SMALL_MOLECULE_FORCEFIELD = 'openff-2.2.0.offxml'
MAX_MULTI_LIGAND_PROBES = 100


def _add_solvent_with_seed(
        omm_model: app.Modeller,
        forcefield: app.ForceField,
        water_model: str,
        ionic_strength_molar: float,
        box_kwargs: dict[str, object],
        seed: int | None,
) -> None:
    """Add solvent while making OpenMM's ion placement reproducible.

    OpenMM selects waters for ion replacement with Python's global ``random``
    module and does not expose a seed argument. MPI ranks must make the same
    selection so they construct identical replica systems.

    Args:
        omm_model: Model to solvate.
        forcefield: Force field used to add water and ions.
        water_model: OpenMM water-model name.
        ionic_strength_molar: Salt concentration in molar.
        box_kwargs: Box arguments passed to ``Modeller.addSolvent``.
        seed: Seed used for ion placement.
    """
    random_state = random.getstate()
    try:
        random.seed(seed)
        omm_model.addSolvent(
            forcefield,
            model=water_model,
            ionicStrength=ionic_strength_molar * unit.molar,
            **box_kwargs,
        )
    finally:
        random.setstate(random_state)


def resolve_probe_sdf_path(probe_type: str) -> str:
    """Resolve a built-in probe name or user-supplied SDF path.

    Args:
        probe_type: Built-in probe name or path to an SDF containing one molecule.

    Returns:
        Path to the probe SDF.

    Raises:
        ValueError: If a custom probe path does not have an SDF extension.
        FileNotFoundError: If the resolved SDF does not exist.
    """
    probe_path = PROBE_SDF_MAP.get(probe_type, probe_type)
    if Path(probe_path).suffix.lower() != '.sdf':
        available_probes = ', '.join(PROBE_SDF_MAP)
        raise ValueError(
            f'Probe must be a built-in name ({available_probes}) or an SDF path: {probe_type}'
        )
    if not Path(probe_path).is_file():
        raise FileNotFoundError(f'Probe SDF does not exist: {probe_path}')
    return probe_path


def load_probe(probe_type: str) -> Molecule:
    """Load one probe molecule from a built-in name or user-supplied SDF.

    Built-in SDFs carry Espaloma partial charges. User-supplied SDF charges are
    retained by the OpenFF parser, and an Espaloma charge property is used when
    present.

    Args:
        probe_type: Built-in probe name or path to an SDF containing one molecule.

    Returns:
        OpenFF Molecule with one or more 3D conformers.

    Raises:
        ValueError: If the SDF does not contain exactly one molecule with a 3D conformer.
    """
    probe_path = resolve_probe_sdf_path(probe_type)
    probe_molecules = parse_ligfile_openff(LigFile(path=probe_path))
    if len(probe_molecules) != 1:
        raise ValueError(
            f'Probe SDF must contain exactly one molecule; found {len(probe_molecules)} '
            f'in {probe_path}'
        )
    probe_mol = probe_molecules[0]
    if not probe_mol.conformers:
        raise ValueError(f'Probe SDF must contain 3D coordinates: {probe_path}')
    check_for_espaloma_charges(probe_mol)
    return probe_mol


def _patch_nonstandard_residue_bonds(topology: app.Topology) -> None:
    """Add missing bonds that ``app.PDBFile`` omits for non-standard residues.

    Two independent fixes are applied:

    1. **Internal bonds**: residues with any unbonded atom have their internal
       bonds looked up from the simulation forcefields and added.  Uses "any"
       (not "all") because ``app.PDBFile`` may add the backbone N bond while
       leaving the remaining intra-residue bonds missing (e.g. SEP gets its
       incoming C→N bond but none of its 13 internal bonds).

    2. **Backbone C→N bonds**: every adjacent residue pair in each chain is
       scanned for a missing peptide bond. This catches the case where a
       non-standard residue HAS internal bonds (via CONECT records) but its
       C→N connection to the neighbouring residue was not written to the file.
    """
    bonded_atoms = {a.index for b in topology.bonds() for a in (b.atom1, b.atom2)}

    # Residues with ANY unbonded atom need internal bonds from forcefield templates.
    # "Any" (not "all") because app.PDBFile may add the backbone N bond while leaving
    # all other intra-residue bonds missing (e.g. SEP gets its incoming C→N bond but
    # none of its 13 internal bonds).
    needs_internal_patch = frozenset(
        res.name
        for chain in topology.chains()
        for res in chain.residues()
        if list(res.atoms()) and any(a.index not in bonded_atoms for a in res.atoms())
    )

    internal_bonds: dict[str, list[tuple[str, str]]] = {}
    if needs_internal_patch:
        ff = app.ForceField(*FORCEFIELDS)
        internal_bonds = {
            name: [
                (ff._templates[name].atoms[i].name, ff._templates[name].atoms[j].name)
                for i, j in ff._templates[name].bonds
            ]
            for name in needs_internal_patch
            if name in ff._templates
        }

    for chain in topology.chains():
        residues = list(chain.residues())
        existing = {(b.atom1.index, b.atom2.index) for b in topology.bonds()}
        existing |= {(b.atom2.index, b.atom1.index) for b in topology.bonds()}

        for i, res in enumerate(residues):
            atoms = {a.name: a for a in res.atoms()}

            # Internal bonds for residues with any unbonded atom
            if res.name in needs_internal_patch:
                for a1_name, a2_name in internal_bonds.get(res.name, []):
                    if a1_name in atoms and a2_name in atoms:
                        a1, a2 = atoms[a1_name], atoms[a2_name]
                        if (a1.index, a2.index) not in existing:
                            topology.addBond(a1, a2)
                            existing.add((a1.index, a2.index))
                            existing.add((a2.index, a1.index))

            # Backbone C→N bond to next residue — always check, for all residue types
            if i < len(residues) - 1 and 'C' in atoms:
                next_atoms = {a.name: a for a in residues[i + 1].atoms()}
                if 'N' in next_atoms:
                    c, n = atoms['C'], next_atoms['N']
                    if (c.index, n.index) not in existing:
                        topology.addBond(c, n)
                        existing.add((c.index, n.index))
                        existing.add((n.index, c.index))


def load_protein_pdb(pdb_path: str) -> app.PDBFile:
    """Load a protein PDB and ensure bonds are complete for all residues.

    Wraps ``app.PDBFile`` and patches any residue that has at least one
    unbonded atom — covers both fully-skipped residues and the partial case
    where ``app.PDBFile`` adds the backbone N bond but omits internal bonds.
    Safe to call on standard-residue-only structures (no-op).
    """
    pdb = app.PDBFile(pdb_path)
    _patch_nonstandard_residue_bonds(pdb.topology)
    return pdb


def create_probe_system_generator(
        forcefield: str,
        probe_type: str,
        probe_ff_path: str | None = None,
) -> tuple[Molecule, SystemGenerator]:
    """Create a SystemGenerator configured for the requested probe.

    Args:
        forcefield: Key from ``FORCEFIELD_OPTIONS``.
        probe_type: Built-in probe name or path to an SDF containing one molecule.
        probe_ff_path: Optional SMIRNOFF OFFXML for the probe. If omitted, OpenFF 2.2.0
            is used for probes that do not have a bundled template.

    Returns:
        The loaded probe molecule and configured SystemGenerator.
    """
    forcefields = list(FORCEFIELD_OPTIONS[forcefield][0])
    probe_mol = load_probe(probe_type)
    use_template = uses_template_probe(forcefield, probe_type, probe_ff_path)
    if not use_template and DESAMBER_BENZENE_FORCEFIELD in forcefields:
        forcefields.remove(DESAMBER_BENZENE_FORCEFIELD)
    register_molecules = [] if use_template else [probe_mol]
    system_generator = SystemGenerator(
        forcefields=forcefields,
        small_molecule_forcefield=probe_ff_path or DEFAULT_SMALL_MOLECULE_FORCEFIELD,
        molecules=register_molecules,
    )
    if register_molecules:
        _align_small_molecule_14_scales(system_generator)
    return probe_mol, system_generator


def create_multi_probe_system_generator(
        forcefield: str,
        probe_types: Sequence[str],
        probe_ff_paths: Sequence[str],
) -> tuple[list[Molecule], SystemGenerator]:
    """Create a SystemGenerator with one parameter source per ligand.

    Args:
        forcefield: Key from ``FORCEFIELD_OPTIONS``.
        probe_types: Ligand SDF paths.
        probe_ff_paths: Paired SMIRNOFF OFFXML paths.

    Returns:
        Loaded ligand molecules and a configured SystemGenerator.

    Raises:
        ValueError: If the ligand and force-field counts differ.
    """
    if not probe_types:
        raise ValueError('At least one ligand is required.')
    if len(probe_types) != len(probe_ff_paths):
        raise ValueError('Each ligand must have one paired probe force field.')

    forcefields = list(FORCEFIELD_OPTIONS[forcefield][0])
    if DESAMBER_BENZENE_FORCEFIELD in forcefields:
        forcefields.remove(DESAMBER_BENZENE_FORCEFIELD)
    probe_mols = [load_probe(probe_type) for probe_type in probe_types]
    _validate_unique_probes(probe_mols, probe_types)
    first_probe_ff_path = probe_ff_paths[0]
    system_generator = SystemGenerator(
        forcefields=forcefields,
        small_molecule_forcefield=first_probe_ff_path,
        molecules=[probe_mols[0]],
    )
    _align_small_molecule_14_scales(system_generator)

    template_generators = [system_generator.template_generator]
    for probe_mol, probe_ff_path in zip(
            probe_mols[1:], probe_ff_paths[1:], strict=True):
        template_generator = SMIRNOFFTemplateGenerator(
            molecules=[probe_mol],
            forcefield=probe_ff_path,
        )
        _align_template_14_scales(
            system_generator.forcefield,
            template_generator,
        )
        system_generator.forcefield.registerTemplateGenerator(
            template_generator.generator,
        )
        template_generators.append(template_generator)
    _register_multi_probe_template_matcher(
        system_generator.forcefield,
        probe_mols,
        template_generators,
    )
    return probe_mols, system_generator


def _validate_unique_probes(
        probe_mols: Sequence[Molecule],
        probe_types: Sequence[str],
) -> None:
    """Reject chemically identical duplicate ligands in one mixture.

    Residue-directed matching safely distinguishes stereoisomers and other
    graph-identical species, but including the exact same molecule twice remains
    an input error.

    Args:
        probe_mols: Loaded ligand molecules.
        probe_types: User-facing probe identifiers paired with the molecules.

    Raises:
        ValueError: If two inputs describe the same chemical species.
    """
    for first_index, first_mol in enumerate(probe_mols):
        for second_index in range(first_index + 1, len(probe_mols)):
            second_mol = probe_mols[second_index]
            isomorphic, _ = Molecule.are_isomorphic(
                first_mol,
                second_mol,
            )
            if isomorphic:
                raise ValueError(
                    'Multi-ligand MSMD received chemically identical ligands: '
                    f'{probe_types[first_index]} and {probe_types[second_index]}.',
                )


def _register_multi_probe_template_matcher(
        forcefield: app.ForceField,
        probe_mols: Sequence[Molecule],
        template_generators: Sequence[SMIRNOFFTemplateGenerator],
) -> None:
    """Route each multi-ligand residue to its paired SMIRNOFF template.

    OpenMM's default residue matching does not preserve bond order, formal
    charge, or stereochemistry. SandCat's stable ``P00``, ``P01``, ... residue
    names provide an unambiguous identity that lets graph-identical ligands use
    distinct bespoke force fields in fresh builds and PDB-based restarts.

    Args:
        forcefield: OpenMM force field receiving the matcher.
        probe_mols: Ordered ligand molecules.
        template_generators: One paired template generator per ligand.

    Raises:
        ValueError: If the molecule and template-generator counts differ.
    """
    if len(probe_mols) != len(template_generators):
        raise ValueError('Each ligand must have one paired template generator.')

    template_routes = {
        probe_resname: (
            probe_mol,
            probe_mol.to_smiles(),
            template_generator,
        )
        for probe_resname, probe_mol, template_generator in zip(
            multi_ligand_probe_resnames(len(probe_mols)),
            probe_mols,
            template_generators,
            strict=True,
        )
    }

    def match_probe_template(
            matching_forcefield: app.ForceField,
            residue: app.Topology.Residue,
            bonded_to_atom: Sequence[Sequence[int]],
            ignore_external_bonds: bool,
            ignore_extra_particles: bool,
    ) -> object | None:
        del bonded_to_atom, ignore_external_bonds, ignore_extra_particles
        template_route = template_routes.get(residue.name)
        if template_route is None:
            return None

        probe_mol, template_name, template_generator = template_route
        if template_generator._match_residue(residue, probe_mol) is None:
            raise ValueError(
                f'Residue {residue.name} does not match its paired ligand template.',
            )
        if template_name not in matching_forcefield._templates:
            if not template_generator.generator(matching_forcefield, residue):
                raise ValueError(
                    f'Could not generate the paired template for residue '
                    f'{residue.name}.',
                )
        return matching_forcefield._templates[template_name]

    forcefield.registerTemplateMatcher(match_probe_template)


def multi_ligand_probe_resnames(n_probes: int) -> list[str]:
    """Return stable three-character residue names for a ligand mixture.

    Args:
        n_probes: Number of distinct ligands in the mixture.

    Returns:
        Ordered names ``P00`` through ``P99``.

    Raises:
        ValueError: If the count is outside the supported range.
    """
    if not 1 <= n_probes <= MAX_MULTI_LIGAND_PROBES:
        raise ValueError(
            f'Multi-ligand MSMD supports 1-{MAX_MULTI_LIGAND_PROBES} ligands; '
            f'received {n_probes}.',
        )
    return [f'P{probe_index:02d}' for probe_index in range(n_probes)]


def _align_small_molecule_14_scales(
        system_generator: SystemGenerator,
        relative_tolerance: float = 5e-5,
        absolute_tolerance: float = 1e-12,
) -> None:
    """Match generated probe-template 1-4 scales to the loaded protein force field.

    OpenMM requires all ``NonbondedForce`` definitions in one ``ForceField`` to use
    exactly the same 1-4 scaling factors. Some force fields encode numerically
    equivalent values at different precision, so the generated SMIRNOFF template
    must use the representation already loaded by OpenMM.

    Args:
        system_generator: SystemGenerator containing the protein and probe force fields.
        relative_tolerance: Relative tolerance for equivalent scale representations.
        absolute_tolerance: Absolute tolerance for equivalent scale representations.

    Raises:
        ValueError: If the custom probe and protein force fields use genuinely
            different 1-4 scaling factors.
    """
    template_generator = system_generator.template_generator
    if template_generator is None:
        return
    _align_template_14_scales(
        system_generator.forcefield,
        template_generator,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )


def _align_template_14_scales(
        forcefield: app.ForceField,
        template_generator: SMIRNOFFTemplateGenerator,
        relative_tolerance: float = 5e-5,
        absolute_tolerance: float = 1e-12,
) -> None:
    """Align one SMIRNOFF template generator with protein 1-4 scales.

    Args:
        forcefield: Protein and solvent OpenMM force field.
        template_generator: Ligand template generator to align.
        relative_tolerance: Relative tolerance for equivalent scale representations.
        absolute_tolerance: Absolute tolerance for equivalent scale representations.

    Raises:
        ValueError: If ligand and protein 1-4 scales are not equivalent.
    """
    for force_generator in forcefield.getGenerators():
        if not hasattr(force_generator, 'coulomb14scale'):
            continue
        protein_coulomb_scale = float(force_generator.coulomb14scale)
        protein_lj_scale = float(force_generator.lj14scale)
        probe_coulomb_scale = float(template_generator._coulomb14scale)
        probe_lj_scale = float(template_generator._lj14scale)
        scales_match = (
            math.isclose(
                protein_coulomb_scale,
                probe_coulomb_scale,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
            and math.isclose(
                protein_lj_scale,
                probe_lj_scale,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            )
        )
        if not scales_match:
            raise ValueError(
                'Custom probe force field uses unsupported 1-4 scaling factors: '
                f'probe Coulomb/LJ={probe_coulomb_scale}/{probe_lj_scale}, '
                f'protein Coulomb/LJ={protein_coulomb_scale}/{protein_lj_scale}.'
            )
        template_generator._coulomb14scale = str(force_generator.coulomb14scale)
        template_generator._lj14scale = str(force_generator.lj14scale)
        return


def uses_historical_probe_14_correction(
        probe_type: str,
        probe_ff_path: str | None,
) -> bool:
    """Return whether SandCat's historical built-in-probe 1-4 rewrite applies.

    Args:
        probe_type: Built-in probe name or custom SDF path.
        probe_ff_path: Optional custom probe OFFXML.

    Returns:
        True only for a built-in probe without a custom force-field override.
    """
    return probe_type in PROBE_SDF_MAP and probe_ff_path is None


def _n_probes_for_concentration(n_waters: int, conc_mol_per_litre: float) -> int:
    """Calculate number of probe molecules needed to reach a target molar concentration.

    Uses the formula: n_probe = N_A x c x n_waters x V_water, where V_water is the
    volume of one water molecule. For 1 M in ~12000 waters this returns ~215, which
    matches the SWISH bootcamp reference setup for 1JWP.

    Args:
        n_waters: Number of water molecules in the solvated box.
        conc_mol_per_litre: Target probe molar concentration.

    Returns:
        Number of probe molecules to add.
    """
    avogadro = 6.02214076e23
    water_volume_litres = n_waters * WATER_VOLUME_L
    return round(avogadro * conc_mol_per_litre * water_volume_litres)


def _build_periodic_tree(
        positions_nm: np.ndarray,
        box_matrix: np.ndarray | None,
) -> tuple[cKDTree | None, np.ndarray]:
    """Build a spatial index, including neighbouring periodic images when needed.

    Args:
        positions_nm: Cartesian positions with shape ``(n_positions, 3)``.
        box_matrix: Periodic box vectors as matrix rows, or None for non-periodic data.

    Returns:
        Spatial tree and an array mapping each tree point to its original position.
        The tree is None when no positions were supplied.
    """
    positions = np.asarray(positions_nm, dtype=float).reshape(-1, 3)
    if not len(positions):
        return None, np.empty(0, dtype=int)
    if box_matrix is None:
        return cKDTree(positions), np.arange(len(positions))

    image_offsets = np.asarray(
        list(product((-1, 0, 1), repeat=3)),
        dtype=float,
    ) @ box_matrix
    image_positions = (
        positions[np.newaxis, :, :] + image_offsets[:, np.newaxis, :]
    ).reshape(-1, 3)
    source_indices = np.tile(np.arange(len(positions)), len(image_offsets))
    return cKDTree(image_positions), source_indices


def _add_probe_copies(
        omm_model: app.Modeller,
        probe_topology: app.Topology,
        probe_positions_nm: list[np.ndarray],
) -> None:
    """Add all placed probe copies to a model in one operation.

    Args:
        omm_model: Model receiving the probe copies.
        probe_topology: Topology of one probe molecule.
        probe_positions_nm: Cartesian coordinates for each placed probe.
    """
    if not probe_positions_nm:
        return

    copies_topology = app.Topology()
    copies_positions_nm = []
    for copy_positions_nm in probe_positions_nm:
        atom_map = {}
        for source_chain in probe_topology.chains():
            target_chain = copies_topology.addChain(source_chain.id)
            for source_residue in source_chain.residues():
                target_residue = copies_topology.addResidue(
                    source_residue.name,
                    target_chain,
                    id=source_residue.id,
                    insertionCode=source_residue.insertionCode,
                )
                for source_atom in source_residue.atoms():
                    atom_kwargs = {'id': source_atom.id}
                    # OpenMM added formal-charge support to Topology.Atom and
                    # Topology.addAtom together.  Omit the keyword entirely on
                    # older releases, where even ``formalCharge=None`` fails.
                    if hasattr(source_atom, 'formalCharge'):
                        atom_kwargs['formalCharge'] = source_atom.formalCharge
                    atom_map[source_atom] = copies_topology.addAtom(
                        source_atom.name,
                        source_atom.element,
                        target_residue,
                        **atom_kwargs,
                    )
        for source_bond in probe_topology.bonds():
            copies_topology.addBond(
                atom_map[source_bond[0]],
                atom_map[source_bond[1]],
                type=source_bond.type,
                order=source_bond.order,
            )
        copies_positions_nm.append(copy_positions_nm)

    omm_model.add(
        copies_topology,
        np.vstack(copies_positions_nm) * unit.nanometer,
    )


def add_mixed_solvent(
        omm_model: app.Modeller,
        probe_mol: Molecule,
        probe_resname: str = PROBE_RESNAME,
        conc_mol_per_litre: float = 1.0,
        seed: int | None = 42,
        clash_distance_nm: float = 0.2,
        solvent_clearance_nm: float = 0.25,
        preserve_single_water_replacement: bool = False,
        probe_count: int | None = None,
) -> None:
    """Replace a subset of water molecules with organic probe molecules to create a mixed
    solvent system at the target molar concentration.

    Probes are placed at water oxygen sites with random orientations. Candidate
    placements that clash with the solute or another probe are rejected, and every
    water displaced by the complete probe volume is removed. This is important for
    ligand-sized probes: deleting only the water at the probe centre leaves the rest
    of the ligand superimposed on neighbouring solvent and can produce NaN coordinates
    during minimisation.

    Args:
        omm_model: Solvated OpenMM Modeller containing protein and TIP3P water.
        probe_mol: OpenFF Molecule for the probe, with a 3D conformer already generated.
            Must be the same molecule registered with SystemGenerator.
        probe_resname: Residue name to assign to all probe copies in the topology.
        conc_mol_per_litre: Target molar concentration of probe.
        seed: Random seed for reproducible probe placement.
        clash_distance_nm: Minimum heavy-atom distance to solute and other probes.
        solvent_clearance_nm: Remove waters whose oxygen is closer than this to any
            probe heavy atom.
        preserve_single_water_replacement: Reproduce the validated bundled-probe
            placement by replacing exactly one water with each probe.
        probe_count: Exact number of copies to insert. If omitted, derive the count
            from ``conc_mol_per_litre``.

    Raises:
        ValueError: If the requested number of probes cannot be placed without clashes.
    """
    rng = np.random.default_rng(seed)

    # Collect water residues and their oxygen positions in one pass
    positions_nm = np.array(omm_model.positions.value_in_unit(unit.nanometer))
    water_oxygen_info = []
    solute_heavy_positions = []
    for res in omm_model.topology.residues():
        if res.name == 'HOH':
            for atom in res.atoms():
                if atom.element.symbol == 'O':
                    water_oxygen_info.append((res, positions_nm[atom.index]))
                    break
        else:
            for atom in res.atoms():
                if atom.element is not None and atom.element.symbol != 'H':
                    solute_heavy_positions.append(positions_nm[atom.index])

    n_waters = len(water_oxygen_info)
    n_probes = (
        _n_probes_for_concentration(n_waters, conc_mol_per_litre)
        if probe_count is None else probe_count
    )

    if n_probes > n_waters:
        raise ValueError(
            f'Requested {n_probes} probe molecules but only {n_waters} water molecules are available.'
        )

    # Build a single-residue OpenMM topology for the probe
    probe_omm_topology = probe_mol.to_topology().to_openmm()
    for chain in probe_omm_topology.chains():
        for residue in chain.residues():
            residue.name = probe_resname

    # Conformer in nm, centred at origin for clean rotation/translation
    conformer_nm = probe_mol.conformers[0].magnitude * 0.1  # Angstrom to nm
    conformer_nm -= conformer_nm.mean(axis=0)

    if preserve_single_water_replacement:
        selected_indices = rng.choice(n_waters, size=n_probes, replace=False)
        probe_centres_nm = np.array([
            water_oxygen_info[index][1] for index in selected_indices
        ])
        omm_model.delete([
            water_oxygen_info[index][0] for index in selected_indices
        ])
        placed_probe_positions = []
        for probe_centre_nm in probe_centres_nm:
            random_matrix = rng.standard_normal((3, 3))
            rotation_matrix, _ = np.linalg.qr(random_matrix)
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, 0] *= -1
            rotated_nm = (
                rotation_matrix @ conformer_nm.T
            ).T + probe_centre_nm
            placed_probe_positions.append(rotated_nm)
        _add_probe_copies(
            omm_model,
            probe_omm_topology,
            placed_probe_positions,
        )
        return

    probe_heavy_indices = np.array([
        atom.molecule_atom_index
        for atom in probe_mol.atoms
        if atom.atomic_number != 1
    ])

    box_vectors = omm_model.topology.getPeriodicBoxVectors()
    box_matrix = None
    inverse_box_matrix = None
    if box_vectors is not None:
        box_matrix = np.array([
            vector.value_in_unit(unit.nanometer) for vector in box_vectors
        ])
        inverse_box_matrix = np.linalg.inv(box_matrix)

    def minimum_distances(
            first_positions: np.ndarray,
            second_positions: np.ndarray,
    ) -> np.ndarray:
        """Return all pair distances, applying the triclinic minimum image."""
        if len(first_positions) == 0 or len(second_positions) == 0:
            return np.empty((len(first_positions), len(second_positions)))
        displacement = (
            np.asarray(first_positions)[:, np.newaxis, :]
            - np.asarray(second_positions)[np.newaxis, :, :]
        )
        if box_matrix is not None:
            fractional = displacement @ inverse_box_matrix
            fractional -= np.round(fractional)
            displacement = fractional @ box_matrix
        return np.linalg.norm(displacement, axis=2)

    water_oxygen_positions = np.array([info[1] for info in water_oxygen_info])
    fixed_heavy_positions = np.asarray(
        solute_heavy_positions,
        dtype=float,
    ).reshape(-1, 3)
    water_tree, water_source_indices = _build_periodic_tree(
        water_oxygen_positions,
        box_matrix,
    )
    fixed_heavy_tree, _ = _build_periodic_tree(
        fixed_heavy_positions,
        box_matrix,
    )
    placed_heavy_positions = np.empty((0, 3))
    placed_probe_positions = []
    displaced_water_indices = set()

    # Try water sites in random order. A water displaced by an earlier probe cannot
    # subsequently serve as the centre of another probe.
    for water_index in rng.permutation(n_waters):
        if len(placed_probe_positions) == n_probes:
            break
        if water_index in displaced_water_indices:
            continue
        rand_matrix = rng.standard_normal((3, 3))
        rot_matrix, _ = np.linalg.qr(rand_matrix)
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[:, 0] *= -1  # ensure det = +1 (proper rotation, not reflection)

        rotated_nm = (
            (rot_matrix @ conformer_nm.T).T + water_oxygen_positions[water_index]
        )
        candidate_heavy_positions = rotated_nm[probe_heavy_indices]
        fixed_collisions = (
            fixed_heavy_tree is not None
            and any(
                len(neighbours)
                for neighbours in fixed_heavy_tree.query_ball_point(
                    candidate_heavy_positions,
                    clash_distance_nm,
                )
            )
        )
        placed_collisions = (
            placed_heavy_positions.size
            and np.any(
                minimum_distances(
                    candidate_heavy_positions,
                    placed_heavy_positions,
                ) < clash_distance_nm
            )
        )
        if fixed_collisions or placed_collisions:
            continue

        water_neighbours = water_tree.query_ball_point(
            candidate_heavy_positions,
            solvent_clearance_nm,
        )
        new_displaced_indices = {
            int(water_source_indices[tree_index])
            for atom_neighbours in water_neighbours
            for tree_index in atom_neighbours
        }
        new_displaced_indices.add(int(water_index))
        displaced_water_indices.update(new_displaced_indices)
        placed_heavy_positions = np.vstack([
            placed_heavy_positions,
            candidate_heavy_positions,
        ])
        placed_probe_positions.append(rotated_nm)

    if len(placed_probe_positions) != n_probes:
        raise ValueError(
            f'Could place only {len(placed_probe_positions)} of {n_probes} probes '
            f'without heavy-atom clashes. Reduce the probe concentration or enlarge the box.'
        )

    omm_model.delete([
        water_oxygen_info[index][0] for index in sorted(displaced_water_indices)
    ])
    _add_probe_copies(
        omm_model,
        probe_omm_topology,
        placed_probe_positions,
    )


def solvent_box_kwargs(box_info_json_path: str | None) -> dict:
    """Return ``addSolvent`` kwargs for the user's box configuration.

    Wraps ``parse_box_info_json_file`` so the SandCat CLI accepts a shared
    ``-bi`` interface across workflows. When ``box_info_json_path``
    is None, returns the sandcat default ``{'padding': 1.0 nm}`` (cubic), preserving
    historical solvation behaviour. 

    Args:
        box_info_json_path: path to JSON file with box parameters, or None.

    Returns:
        dict of kwargs to splat into ``app.Modeller.addSolvent``.
    """
    if box_info_json_path is None:
        return {'padding': 1.0 * unit.nanometer}
    padding, box_shape, box_size, box_vectors = parse_box_info_json_file(box_info_json_path)
    kwargs = {
        'padding': padding,
        'boxShape': box_shape,
        'boxSize': box_size,
        'boxVectors': box_vectors,
    }
    return {k: v for k, v in kwargs.items() if v not in (None, '')}


def build_mixed_solvent_system(
        forcefield: str,
        protein_pdb_path: str,
        probe_type: str,
        conc_mol_per_litre: float,
        ionic_strength_molar: float = 0.15,
        box_info_json_path: str | None = None,
        seed: int | None = 42,
        constraints=app.HBonds,
        hydrogen_mass=4 * unit.amu,
        probe_ff_path: str | None = None,
) -> tuple[System, app.Modeller, SystemGenerator]:
    """Build a solvated, mixed-solvent OpenMM (System, Modeller) for the given force field.

    For all force fields (ff14/ff19/desamber) this is the openmmforcefields ``SystemGenerator`` +
    ``Modeller.addSolvent`` + probe-replacement path. All other sandcat machinery (probe
    repulsion, contact map, SWISH lambda scaling, equilibration) operates on the returned
    System/Modeller unchanged.

    Args:
        forcefield: key from ``ALL_FORCEFIELDS``.
        protein_pdb_path: path to the (already protonated, if requested) protein PDB.
        probe_type: Built-in probe name or path to an SDF containing one molecule.
        conc_mol_per_litre: target probe concentration.
        ionic_strength_molar: salt concentration for neutralisation.
        box_info_json_path: optional box spec.
        seed: random seed for solvent-ion and probe placement.
        constraints, hydrogen_mass: passed to ``createSystem`` (default HBonds + HMR, matching
            sandcat's 4 fs integrator).
        probe_ff_path: Optional SMIRNOFF OFFXML used to parameterize the probe.

    Returns:
        ``(system, omm_model, system_generator)`` where ``system_generator`` is the
        ``SystemGenerator`` (needed for checkpoint restart).
    """
    _, water_model = FORCEFIELD_OPTIONS[forcefield]
    probe_mol, system_generator = create_probe_system_generator(
        forcefield=forcefield,
        probe_type=probe_type,
        probe_ff_path=probe_ff_path,
    )
    protein = load_protein_pdb(protein_pdb_path)
    omm_model = app.Modeller(protein.topology, protein.positions)
    _add_solvent_with_seed(
        omm_model=omm_model,
        forcefield=system_generator.forcefield,
        water_model=water_model,
        ionic_strength_molar=ionic_strength_molar,
        box_kwargs=solvent_box_kwargs(box_info_json_path),
        seed=seed,
    )
    add_mixed_solvent(
        omm_model,
        probe_mol=probe_mol,
        conc_mol_per_litre=conc_mol_per_litre,
        seed=seed,
        preserve_single_water_replacement=probe_type in PROBE_SDF_MAP,
    )
    system = system_generator.forcefield.createSystem(
        omm_model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=constraints,
        hydrogenMass=hydrogen_mass,
    )
    add_custom_metal_bonds(system, omm_model)
    return system, omm_model, system_generator


def build_multi_ligand_solvent_system(
        forcefield: str,
        protein_pdb_path: str,
        probe_types: Sequence[str],
        probe_ff_paths: Sequence[str],
        ionic_strength_molar: float = 0.15,
        box_info_json_path: str | None = None,
        seed: int | None = 42,
        constraints=app.HBonds,
        hydrogen_mass=4 * unit.amu,
) -> tuple[System, app.Modeller, SystemGenerator]:
    """Build an MSMD system containing one copy of each supplied ligand.

    Args:
        forcefield: Key from ``FORCEFIELD_OPTIONS``.
        protein_pdb_path: Path to the input protein PDB.
        probe_types: Ligand SDF paths.
        probe_ff_paths: Paired SMIRNOFF OFFXML paths.
        ionic_strength_molar: Salt concentration for neutralisation.
        box_info_json_path: Optional box specification.
        seed: Random seed for solvent-ion and ligand placement.
        constraints: Constraints passed to ``createSystem``.
        hydrogen_mass: Hydrogen mass passed to ``createSystem``.

    Returns:
        System, solvated model, and SystemGenerator used to parameterize it.
    """
    _, water_model = FORCEFIELD_OPTIONS[forcefield]
    probe_mols, system_generator = create_multi_probe_system_generator(
        forcefield=forcefield,
        probe_types=probe_types,
        probe_ff_paths=probe_ff_paths,
    )
    probe_resnames = multi_ligand_probe_resnames(len(probe_mols))
    protein = load_protein_pdb(protein_pdb_path)
    omm_model = app.Modeller(protein.topology, protein.positions)
    _add_solvent_with_seed(
        omm_model=omm_model,
        forcefield=system_generator.forcefield,
        water_model=water_model,
        ionic_strength_molar=ionic_strength_molar,
        box_kwargs=solvent_box_kwargs(box_info_json_path),
        seed=seed,
    )
    for probe_index, (probe_mol, probe_resname) in enumerate(
            zip(probe_mols, probe_resnames, strict=True)):
        placement_seed = None if seed is None else seed + probe_index
        add_mixed_solvent(
            omm_model,
            probe_mol=probe_mol,
            probe_resname=probe_resname,
            seed=placement_seed,
            probe_count=1,
        )
    system = system_generator.forcefield.createSystem(
        omm_model.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.2 * unit.nanometer,
        constraints=constraints,
        hydrogenMass=hydrogen_mass,
    )
    add_custom_metal_bonds(system, omm_model)
    return system, omm_model, system_generator


def validate_n_replicas(n_replicas: int) -> None:
    """Raise ValueError if n_replicas is outside the supported range (2-8).

    Args:
        n_replicas (int): Number of temperature replicas to validate.
    """
    if n_replicas < 2:
        raise ValueError('replica exchange needs more than one replica to run')
    if n_replicas > 8:
        raise ValueError(
            'XGpuMedCpu machines currently have 8 cores, setting the upper replica number limit'
        )
