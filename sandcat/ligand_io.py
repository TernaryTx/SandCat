from __future__ import annotations

import gzip
import json
import os
from collections import Counter

from openff.toolkit.topology import Molecule as openff_Molecule
from rdkit.Chem import ForwardSDMolSupplier
from rdkit.Chem import rdchem
from rdkit.Chem import SDMolSupplier

from sandcat.ligand import LigFile


def rename_duplicate_mol_names(mols: list[openff_Molecule]) -> list[openff_Molecule]:
    """ Rename duplicate molecule names in a list by appending a unique
     index.

    Args:
        mols (list): List of Molecule objects with a 'name' attribute.
    Returns:
        list: Modified list with unique molecule names.
    """
    name_counts = Counter(mol.name for mol in mols)
    running_counts: dict[str, int] = {}
    for mol in mols:
        if name_counts[mol.name] > 1:
            suffix = running_counts.get(mol.name, 0) + 1
            running_counts[mol.name] = suffix
            mol.name = f"{mol.name}_{suffix}"
    return mols


def parse_ligfile_openff(ligfile: LigFile) -> list[openff_Molecule]:
    """ Load a molecule from a file and return it as an openff `Molecule`
    object. Supports SDF (`.sdf`), SDF GZ (`.sdf.gz`) and JSON
    (`.json`) file formats.

    Args:
        ligfile (LigFile): a LigFile object with the path to the SDF or JSON
            file.
    Returns:
        A list of openff.toolkit.topology.Molecule objects
    """
    fname = ligfile.path
    if fname.endswith('.sdf.gz'):
        # Open the .sdf.gz file using gzip
        with gzip.open(fname, 'rb') as f:
            suppl = ForwardSDMolSupplier(f, removeHs=False)
            return [
                openff_Molecule.from_rdkit(mol) for mol in suppl if mol is not None
            ]
    elif fname.endswith('.sdf'):
        output = openff_Molecule.from_file(fname)
    elif fname.endswith('.json'):
        with open(fname) as json_file:
            ligand_dict = json.load(json_file)
        output = openff_Molecule.from_json(ligand_dict)
    else:
        raise ValueError('Input file must be in .sdf, .sdf.gz or .json \
        format.')
    # make sure to return a list
    if isinstance(output, openff_Molecule):
        return [output]
    return output


def parse_ligfile_rdkit(ligfile: LigFile) -> list[rdchem.Mol]:
    """Parse molecules from a LigFile into RDKit molecule objects.
    Args:
        ligfile (LigFile): Ligand file to parse. Must be in SDF or SDF.GZ format
    Returns:
        list[Mol]: List of RDKit molecules.
    """
    if ligfile.path.endswith('.gz'):
        with gzip.open(ligfile.path) as sdf_gz_file:
            supplier = ForwardSDMolSupplier(sdf_gz_file)
            return [mol for mol in supplier]
    supplier = SDMolSupplier(ligfile.path)
    return [mol for mol in supplier]


def split_sdf(
    ligfile: LigFile,
    output_dir: str | None = None,
) -> None:
    """ Read an SDF file and return individual SDF files for each molecule,
     named according to the header in the SDF file.

    Args:
         ligfile (LigFile): a LigFile object with the path to the SDF file.
        output_dir (Optional[str]): path to the output directory (or defaults
            to writing to same directory as input)
    """
    if output_dir is None:
        output_dir = os.path.dirname(ligfile.path)

    mols = parse_ligfile_openff(ligfile)
    # check if all the names are different
    mols = rename_duplicate_mol_names(mols)
    for mol in mols:
        header = mol.name
        mol.to_file(os.path.join(output_dir, f'{header}.sdf'), 'SDF')
