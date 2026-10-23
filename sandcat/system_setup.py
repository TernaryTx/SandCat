from __future__ import annotations

import json

import mdtraj as mdt
import numpy as np
from Bio.PDB.Polypeptide import standard_aa_names as STANDARD_AA_NAMES
from mdtraj import Topology as mdtraj_Topology
from mdtraj import Trajectory
from openff.toolkit import Molecule
from openff.units import unit
from openmm import CustomExternalForce
from openmm import HarmonicBondForce
from openmm import LangevinMiddleIntegrator
from openmm import NonbondedForce
from openmm import Platform
from openmm import System
from openmm import Vec3
from openmm import XmlSerializer
from openmm.app import HBonds
from openmm.app import Modeller
from openmm.app import NoCutoff
from openmm.app import PDBFile
from openmm.app import PDBxFile
from openmm.app import PME
from openmm.app import Simulation
from openmm.app.topology import Topology
from openmm.unit import amu
from openmm.unit import kilojoule_per_mole
from openmm.unit import molar
from openmm.unit import nanometer
from openmm.unit import picosecond
from openmm.unit.quantity import Quantity
from openmmforcefields.generators import SystemGenerator

from sandcat.ligand import LigFile
from sandcat.ligand_io import parse_ligfile_openff
from sandcat.md_variables import FRICTION
from sandcat.md_variables import get_n_steps
from sandcat.md_variables import SPRING_CONSTANT as default_spring_constant
from sandcat.md_variables import TEMPERATURE
from sandcat.md_variables import TIME_STEP_4FS as time_step

# hard-coded precision, as more accurate than single, but less expensive than double
OPENMM_PROPERTIES = {'Precision': 'mixed'}

TRANSITION_STATE_METAL_RESNAMES = {'ZN', 'ZN2', 'FE', 'FE2'}


def create_system(
    omm_model: Modeller,
    input_ligand: LigFile | None = None,
    ligand_ff_path: str | None = None,
    input_cofactor: LigFile | None = None,
    add_solvent: bool = True,
    system_xml_fpath: str = '',
    box_info_json_fpath: str | None = None,
) -> System:
    """ Create an OpenMM System for a protein with optional explicit
    solvent and optional ligand.

    Args:
        omm_model (Modeller): The OpenMM Modeller object with the
            protein structure.
        input_ligand (LigFile, optional): A ligand to include in the
            system. Defaults to None.
        ligand_ff_path (str): path to OFFXML file for the small
            molecule parameters
        input_cofactor (LigFile, optional): a cofactor with
            the cofactor coordinates and/or parameters
        add_solvent (bool, optional): whether to add TIP3P water
        system_xml_fpath (str, optional): path to XML file that
            contains the System parameters to load
        box_info_json_fpath (str, optional): path to json file with
            dict of 'padding', 'box_size', 'box_vectors', 'box_shape'
            or 'box_edge'.
    Returns:
        System: The solvated OpenMM System object with optional
            explicit water and ions.
    """
    # If provided, load the saved serialized OpenMM System
    if system_xml_fpath:
        with open(system_xml_fpath) as f:
            return XmlSerializer.deserialize(f.read())

    # Load force fields, first for the protein & solvent
    forcefields = [
        'amber/ff14SB.xml', 'amber/phosaa14SB.xml', 'amber14/tip3p.xml',
    ]
    off_molecules = []
    # Process cofactor if provided
    if input_cofactor:
        cofactor = add_ligfile_to_omm_model(omm_model, input_cofactor, 'COF')
        off_molecules.append(cofactor)

    # Add ligand if provided
    if input_ligand:
        ligand = add_ligfile_to_omm_model(omm_model, input_ligand, 'LIG')
        off_molecules.append(ligand)

    small_mol_ff = ligand_ff_path if ligand_ff_path else 'openff-2.2.0.offxml'
    system_generator = SystemGenerator(
        forcefields=forcefields,
        small_molecule_forcefield=small_mol_ff,
        molecules=off_molecules,
    )
    # Add solvent (water box)
    if add_solvent:
        padding, box_shape = 1.0 * nanometer, 'cube'
        box_size, box_vectors = None, None
        if box_info_json_fpath:
            box_info_tpl = parse_box_info_json_file(box_info_json_fpath)
            padding, box_shape, box_size, box_vectors = box_info_tpl
        omm_model.addSolvent(
            system_generator.forcefield,
            model='tip3p',  # Water model
            ionicStrength=0.15 * molar,  # Add ions for 0.15 M ionic conc
            padding=padding,  # Add padding around protein
            boxShape=box_shape,
            boxSize=box_size,
            boxVectors=box_vectors,
        )
    # Create OpenMM system
    nonbonded_method = PME if add_solvent else NoCutoff
    system = system_generator.forcefield.createSystem(
        omm_model.topology,
        nonbondedMethod=nonbonded_method,
        constraints=HBonds,
        hydrogenMass=4 * amu,
    )
    # if there are any transition metals (ZN or FE)
    # in the protein, add coordination bonds
    add_custom_metal_bonds(system, omm_model)

    return system


def check_partial_charges(ligand: Molecule) -> None:
    """Verify or assign partial charges to a ligand, using AM1-BCC.

    Args:
        ligand: OpenFF ``Molecule`` whose partial charges will be
            checked or assigned.

    Raises:
        ValueError: If the molecule has more than 100 atoms and
            partial charges cannot be generated, or if any existing
            partial charge is zero.
    """
    if ligand.partial_charges is None:
        # check if the molecule is too big
        if ligand.n_atoms > 100:
            raise ValueError(
                'This molecule is too large to generate partial \
                charges using AM1-BCC.',
            )
        # otherwise, generate partial charges here
        ligand.assign_partial_charges(
            'am1bcc', True, ligand.conformers,
        )
    elif np.any(
            np.isclose(
                ligand.partial_charges.m_as(unit.elementary_charge),
                0.0,
                atol=1e-5,
            ),
    ):
        raise ValueError(
            'At least one of the atoms has a partial charge of 0.00000',
        )


def _load_charges_from_rdkit_prop(ligand: Molecule, prop_name: str) -> None:
    """Load pre-stored partial charges from a named RDKit molecular property.

    Args:
        ligand: OpenFF ``Molecule`` to which stored charges will be assigned.
            If the property is absent the molecule is left unchanged.
        prop_name: Name of the RDKit property containing space-separated floats.

    Raises:
        ValueError: If the charge count does not match the number of atoms.
    """
    rdkit_mol = ligand.to_rdkit()  # off mols don't have props, rdkit mols do
    if rdkit_mol.HasProp(prop_name):
        charges = [float(x) for x in rdkit_mol.GetProp(prop_name).split()]
        if len(charges) != ligand.n_atoms:
            raise ValueError('Charge count does not match number of atoms')
        ligand.partial_charges = np.array(charges) * unit.elementary_charge


def check_for_am1bcc_charges(ligand: Molecule) -> None:
    """Load pre-stored AM1-BCC partial charges from the 'am1bcc_charges'
    molecular property, if present. This avoids expensive recomputation when
    charges have been pre-cached in the SDF file.

    Args:
        ligand: OpenFF ``Molecule`` to which stored charges will be assigned.
            If the property is absent the molecule is left unchanged.

    Raises:
        ValueError: If the charge count does not match the number of atoms.
    """
    _load_charges_from_rdkit_prop(ligand, 'am1bcc_charges')


def check_for_espaloma_charges(ligand: Molecule) -> None:
    """Assign Espaloma partial charges from an RDKit property if
    available, raising ValueError if the charge count does not match
    the number of ligand atoms.

    Args:
        ligand: OpenFF ``Molecule`` to which Espaloma charges will
            be assigned.

    Raises:
        ValueError: If the number of Espaloma charges does not match
            the number of atoms in the ligand.
    """
    _load_charges_from_rdkit_prop(ligand, 'espaloma_charges')


def add_ligfile_to_omm_model(
    omm_model: Modeller,
    input_ligand: LigFile,
    lig_resname: str = 'LIG',
) -> Molecule:
    """Add a ligand from a file to an OpenMM Modeller using OpenFF.

    Args:
        omm_model (Modeller): The OpenMM Modeller to which the
            ligand will be added.
        input_ligand (str): Path to ligand file (JSON, SDF, MOL2,
            or SMILES file).
        lig_resname (str, optional): Residue name to assign to the
            ligand. Defaults to 'LIG'.

    Returns:
        Molecule: The OpenFF Molecule object that was added.
    """
    ligand_mols = parse_ligfile_openff(input_ligand)
    ligand = ligand_mols[0]
    ligand_topology = ligand.to_topology().to_openmm()
    ligand_positions = ligand.conformers[0].to_openmm()

    # Get Espaloma charges stored in SDF (if any found)
    check_for_espaloma_charges(ligand)

    # check if espaloma charges are good, use AM1-BCC if
    # ligand < 100 atoms
    check_partial_charges(ligand)

    # Rename residue to a consistent name
    for chain in ligand_topology.chains():
        for residue in chain.residues():
            residue.name = lig_resname

    omm_model.add(ligand_topology, ligand_positions)
    return ligand


def minimise_equilibrate(
    system: System,
    omm_model: Modeller,
    run_time_ps_int: int = 10,
) -> tuple[Quantity, str]:
    """ Minimize and equilibrate a molecular system without restraints.
    It centers and wraps the system under periodic boundary conditions
    before saving the equilibrated structure.

    Args:
        system (System): The OpenMM System object representing the
            molecular system.
        omm_model (Modeller): The OpenMM Modeller object containing the
            topology and positions.
        run_time_ps_int (int): Duration of equilibration in picoseconds
            (default: 10).

    Returns:
        Tuple[Quantity, str]:
            - The equilibrated atomic positions as an OpenMM Quantity
                object.
            - Filename of the saved equilibrated structure.
    """
    # add position restraint force here
    rest_force_idx = add_position_restraints(
        system,
        omm_model.positions,
        get_ligand_heavy_atom_indices(omm_model),
    )

    simulation = create_simulation(
        omm_model.topology, system, TEMPERATURE, FRICTION, time_step,
    )

    # Set initial positions
    simulation.context.setPositions(omm_model.positions)

    # Minimization
    simulation.minimizeEnergy()
    total_steps = get_n_steps(run_time_ps_int * picosecond, time_step)
    simulation.step(total_steps)

    # center the proteins/ligands and wrap everything around it
    state = simulation.context.getState(
        getPositions=True, enforcePeriodicBox=True,
    )
    positions = state.getPositions(asNumpy=True).astype(np.float32)
    box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
    md_topology = mdt.Topology.from_openmm(simulation.topology)
    trajectory = mdt.Trajectory(positions, topology=md_topology)
    trajectory.unitcell_vectors = np.array([box_vectors]).astype(np.float32)

    trajectory.image_molecules(inplace=True)

    equil_positions = trajectory.openmm_positions(0)

    equil_fname = write_solvated_complex(
        simulation.topology, equil_positions, 'equil',
    )
    # remove the restraint force object
    system.removeForce(rest_force_idx)

    return equil_positions, equil_fname


def create_simulation(
    topology: Topology, system: System, temperature: Quantity,
    friction: Quantity, time_step: Quantity,
    platform_name: str = 'CUDA',
) -> Simulation:
    """ Create an OpenMM simulation object with a Langevin integrator.

    Args:
        topology (Topology): The topology of the system to simulate.
        system (System): The OpenMM system containing force field
            parameters.
        temperature (Quantity): The target temperature for the Langevin
            integrator (e.g., in kelvin).
        friction (Quantity): The collision frequency (friction
            coefficient) for the Langevin integrator.
        time_step (Quantity): The integration time step (e.g., in
            femtoseconds).
        platform_name (str, optional): The computing platform to use
            ('CUDA', 'CPU', 'OpenCL'). Defaults to 'CUDA'.

    Returns:
        Simulation: An OpenMM Simulation object configured with the
            specified parameters.
    """
    # Set up the integrator and simulation
    integrator = LangevinMiddleIntegrator(temperature, friction, time_step)
    # Define platform and simulation
    platform = Platform.getPlatformByName(platform_name)  # or 'CPU', 'OpenCL'
    if platform_name == 'CPU':
        return Simulation(topology, system, integrator, platform)
    else:
        return Simulation(
            topology, system, integrator, platform,
            OPENMM_PROPERTIES,
        )


def write_solvated_complex(
        topology: Topology,
        positions: list,
        outfile_basename: str = 'solvated_complex',
) -> str:
    """ Save the solvated system to a PDB or mmCIF file for visualization.

    This function writes the solvated molecular system to a PDB file if
     it contains fewer than 100,000 atoms, or to a PDBx/mmCIF file
    otherwise. This ensures compatibility with tools like MDTraj,
    which cannot handle PDB files with more than 100,000 atoms.

    Args:
        topology (Topology): The OpenMM topology of the system.
        positions (list): List of atomic positions.
        outfile_basename (str): The base name for the output file
            (default: 'solvated_complex').

    Returns:
        str: The name of the output file ('<outfile_basename>.pdb' or
            '<outfile_basename>.cif').
    """
    # TODO: make a selector for a different format e.g. XPDB, PARMED, XML
    n_atoms = len(positions)
    if n_atoms < 100000:
        sol_complex_fname = f'{outfile_basename}.pdb'
        with open(sol_complex_fname, 'w') as output:
            PDBFile.writeFile(topology, positions, output, keepIds=True)
    else:
        sol_complex_fname = f'{outfile_basename}.cif'
        with open(sol_complex_fname, 'w') as output:
            PDBxFile.writeFile(topology, positions, output, keepIds=True)

    return sol_complex_fname


def wrap_align_trajectory(
    trj_fpath: str,
    trj_topology: mdtraj_Topology,
    ref_file: str | None = None,
    remove_solvent: bool = False,
) -> Trajectory:
    """ This function centers molecules in the simulation box, wraps periodic
    images around the primary molecule, and aligns the trajectory to a
    reference structure or the first frame of the trajectory.

    Args:
        trj_fpath (str): Path to the trajectory file.
        trj_topology (mdtraj_Topology): The mdtraj Topology of the trajectory.
        ref_file (str | None, optional): Path to the reference structure file.
            If None, the trajectory is aligned to its first frame.
        remove_solvent (bool): whether to remove the solvent

    Returns:
        Trajectory: The processed mdtrj Trajectory object.
    """
    # center the proteins/ligands and wrap everything around it
    trajectory = mdt.load(trj_fpath, top=trj_topology)
    trajectory.image_molecules(inplace=True)
    # TODO: make this select the larger chain by default, now does it by sequence in the file
    trj_chain_indices = trajectory.topology.select('chainid 0 and name CA')
    # align to the starting structure
    if ref_file:
        ref_struct = mdt.load(ref_file)
        ref_chain_indices = ref_struct.topology.select('chainid 0 and name CA')
        trajectory.superpose(
            ref_struct,
            atom_indices=trj_chain_indices,
            ref_atom_indices=ref_chain_indices,
        )
    else:
        trajectory.superpose(
            trajectory,
            frame=0,
            atom_indices=trj_chain_indices,
        )
    if remove_solvent:
        # transition metals aren't part of the 'solvent' if they're
        # attached to the protein
        trajectory.remove_solvent(
            exclude=list(TRANSITION_STATE_METAL_RESNAMES),
            inplace=True,
        )

    return trajectory


def add_position_restraints(
    system: System,
    positions: Quantity,
    atom_idx_to_restrain: list,
    spring_constant: Quantity = default_spring_constant,
) -> int:
    """Adds harmonic position restraints to specified atoms in an
    OpenMM system.

    Args:
        system (System): the OpenMM System object to which the position
            restraints will be added.
        positions (Quantity): the reference positions of the atoms,
            typically obtained from a minimized or equilibrated
            structure. Should be in units of nanometers.
        atom_idx_to_restrain (list): a list of atom indices to which
            position restraints will be applied. Indices should
            correspond to those in the `positions` array.
        spring_constant (Quantity, optional): the spring constant (in
         kJ/mol/nm^2) defining the strength of the restraints. Default
         is 1000.0 kJ/mol/nm^2.
    Returns:
        This function modifies the provided `system` in place by adding
         a `CustomExternalForce` that applies position restraints to
         the specified atoms.
    """
    # add position restraints after System and before Simulation
    # Define the harmonic potential:
    # U(x, y, z) = 0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
    restraint_force = CustomExternalForce(
        '0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2',
    )
    restraint_force.addPerParticleParameter('k')  # Force constant
    restraint_force.addPerParticleParameter('x0')  # Reference x-coordinate
    restraint_force.addPerParticleParameter('y0')  # Reference y-coordinate
    restraint_force.addPerParticleParameter('z0')  # Reference z-coordinate

    # Add the restraint force to the system
    force_idx = system.addForce(restraint_force)
    restraint_force.setForceGroup(force_idx)

    # Add restrained atoms and their reference positions
    for atom_index, position in enumerate(positions):
        if atom_index in atom_idx_to_restrain:
            restraint_force.addParticle(
                atom_index, [
                    spring_constant,
                    position[0], position[1], position[2],
                ],
            )

    return force_idx


def get_ligand_heavy_atom_indices(
        omm_model: Modeller,
        lig_resname: str = 'LIG',
) -> list:
    """ Gets the indexes of the heavy atoms (non-Hs) of a given
    residue (from resname) from an openmm model. This can be used for
    restraints.
        Args:
            omm_model: Openmm model of a protein structure
            lig_resname: 3 letter HET resname of ligand to return indexes for.
        Returns:
            List of integers corresponding to specified ligand non-H atoms.
    """
    ligand_heavy_a_indices = []
    for atom in omm_model.topology.atoms():
        if atom.residue.name == lig_resname and atom.element.symbol != 'H':
            ligand_heavy_a_indices.append(atom.index)
    return ligand_heavy_a_indices


def find_atoms_within_radius(
    modeller: Modeller,
    center: float,
    radius: float,
    atom_name: str = 'CA',
) -> list:
    """Find atoms within a specified radius of a given XYZ coordinate
        in an OpenMM Modeller.

    Args:
        modeller (Modeller): The OpenMM Modeller object containing the
            system.
        center (float): The XYZ coordinate (in nanometers) around which
            to search for atoms.
        radius (float): The cutoff distance (in nanometers).

    Returns:
        list[tuple]: A list of tuples (atom_index, atom_name, distance)
         for atoms within the radius.
    """
    topology = modeller.topology
    positions = modeller.positions

    # Ensure the center is a Vec3 object
    if not isinstance(center, Vec3):
        center = Vec3(*center)

    # Convert center to unitless array
    center_unitless = np.array([center.x, center.y, center.z])
    atoms_within_radius = []

    for atom in topology.atoms():
        if atom.name != atom_name:
            continue
        atom_index = atom.index
        atom_position = positions[atom_index]

        # Convert position to unitless numpy array
        atom_position_unitless = np.array([
            atom_position.x,
            atom_position.y,
            atom_position.z,
        ])

        # Compute distance
        distance = np.linalg.norm(
            atom_position_unitless - center_unitless,
        )

        if distance <= radius:
            atoms_within_radius.append(atom_index)

    return atoms_within_radius


def add_custom_metal_bonds(
    system: System,
    omm_model: Modeller,
) -> None:
    """ Add custom bonds between metal ions and coordinating atoms, and
    define nonbonded exceptions for protein atoms within 3.5 angstrom.

    Args:
        system (System): OpenMM System object.
        omm_model (Modeller): OpenMM Modeller containing topology and
            positions.
    """
    bond_force = HarmonicBondForce()
    system.addForce(bond_force)

    # Get atomic positions
    topology = omm_model.topology
    positions = omm_model.positions

    # Find NonbondedForce in the system
    for force_index in range(system.getNumForces()):
        force = system.getForce(force_index)
        if isinstance(force, NonbondedForce):
            nonbonded_force = force
            break
    else:
        raise ValueError('No NonbondedForce found in the system!')

    # Identify metal and coordinating atoms
    coordinating_atoms = []
    protein_atoms = [
        atom for atom in topology.atoms()
        if atom.residue.name in STANDARD_AA_NAMES
    ]
    metal_atoms = [
        atom for atom in topology.atoms()
        if atom.name.upper() in TRANSITION_STATE_METAL_RESNAMES
    ]
    for atom in protein_atoms:
        if atom.element.symbol in ['N', 'S']:
            coordinating_atoms.append(atom)

    # Define bond parameters
    bond_length = 0.23 * nanometer  # xtal bond length between S and ZN
    # very high k, but necessary for systems to 'look right'
    bond_k = 300000 * kilojoule_per_mole / nanometer ** 2
    exclusion_distance = 0.5  # 5 ang exclusion distance

    # Retrieve existing exceptions to avoid duplicates
    existing_exceptions = set()
    for i in range(nonbonded_force.getNumExceptions()):
        p1, p2, *_ = nonbonded_force.getExceptionParameters(i)
        existing_exceptions.add((min(p1, p2), max(p1, p2)))

    # Create bonds
    for metal in metal_atoms:
        metal_pos = positions[metal.index].value_in_unit(nanometer)
        for atom in coordinating_atoms:
            atom_pos = positions[atom.index].value_in_unit(nanometer)
            distance = np.linalg.norm(metal_pos - atom_pos)
            if distance < 0.35:  # Coordination range
                bond_force.addBond(
                    metal.index, atom.index, bond_length, bond_k,
                )

            # Add nonbonded exceptions for nearby protein atoms
            for atom in protein_atoms:
                atom_pos = positions[atom.index].value_in_unit(nanometer)
                distance = np.linalg.norm(metal_pos - atom_pos)
                if distance < exclusion_distance:
                    pair = (
                        min(metal.index, atom.index),
                        max(metal.index, atom.index),
                    )
                    if pair not in existing_exceptions:
                        nonbonded_force.addException(
                            metal.index, atom.index, 0.0, 1.0, 0.0,
                        )
                        # Track the new exception
                        existing_exceptions.add(pair)


def parse_box_info_json_file(box_info_json_file: str) -> tuple:
    """ Parse openmm simulation box configuration parameters from a JSON file.

    The JSON may define one (and only one) of: `padding`, `box_size`/`box_edge`,
    or `box_vectors`. Values are converted to OpenMM `Quantity` objects where
    appropriate.

    Args:
        box_info_json_file (str): path to a JSON file containing box definitions.
    Returns:
        padding, box_shape, box_size, box_vectors (tuple): where unused entries are
            returned as None.
    Raises:
        ValueError If mutually exclusive box parameters are simultaneously defined.
    """
    # define default values
    padding, box_shape = 1.0 * nanometer, 'cube'
    box_size, box_vectors = None, None

    # load json as dict
    with open(box_info_json_file) as f:
        box_info_dict = json.load(f)

    if 'padding' in box_info_dict and box_info_dict['padding'] is not None:
        # openmm.unit.Quantity
        padding = box_info_dict['padding'] * nanometer

    if 'box_shape' in box_info_dict and box_info_dict['box_shape'] is not None:
        box_shape = box_info_dict['box_shape']  # str

    if 'box_size' in box_info_dict and box_info_dict['box_size'] is not None:
        # openmm.unit.Quantity
        box_size = box_info_dict['box_size'] * nanometer

    if 'box_edge' in box_info_dict and box_info_dict['box_edge'] is not None:
        box_edge = box_info_dict['box_edge']  # float, in nanometers
        if box_size:
            raise ValueError('Only one of these can be specified in \
box_info_json_file: box_size or box_edge')
        # openmm.unit.Quantity
        box_size = Vec3(box_edge, box_edge, box_edge) * nanometer

    if 'box_vectors' in box_info_dict and box_info_dict['box_vectors'] is not None:
        box_vectors = []  # initialise as a list
        box_vec_str = box_info_dict['box_vectors']
        vec_str_lst = box_vec_str.split('Vec3(')[1:]
        for vec in vec_str_lst:
            cleaned_vec = vec.strip('), ')
            components = cleaned_vec.split(', ')
            cleaned_comps = []
            for vec_c in components:
                cleaned_comps.append(float(vec_c.split('=')[-1]))
            box_vectors.append(
                Vec3(
                    x=cleaned_comps[0],
                    y=cleaned_comps[1],
                    z=cleaned_comps[2],
                ),
            )
        # convert to tuple, add units
        box_vectors = tuple(box_vectors) * nanometer

    if box_size is not None and box_vectors is not None:
        raise ValueError('At most one of the following arguments may be \
specified: boxSize, boxVectors, padding')
    if box_size or box_vectors:
        padding = None
        box_shape = ''

    return padding, box_shape, box_size, box_vectors
