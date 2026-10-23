from __future__ import annotations

import logging
import os
import string
from pathlib import Path
from tempfile import mkdtemp

from Bio.PDB import MMCIFParser
from Bio.PDB import PDBIO
from Bio.PDB import PDBList
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

from sandcat.logging_utils import setup_logging
from sandcat.structure import PdbFile
from sandcat.structure import PocketFile


def read_pdb(pdb: PdbFile | PocketFile) -> Structure:
    """ Reads a PDB/MMCIF file and returns a Biopython Structure object
    Args:
        pdb (PdbFile | PocketFile): PdbFile / PocketFile object whose path points to the input file
    Returns:
        Bio.PDB.Structure: output Biopython Structure object
    """
    fname = pdb.path
    if fname.endswith('.pdb'):
        parser = PDBParser()
    else:
        parser = MMCIFParser()
    return parser.get_structure(Path(fname).stem, fname)


def write_pdb(struct: Structure, pdb_fname: str, select_class=None) -> None:
    """ Writes a PDB file from a Biopython Structure object
    Args:
        struct (Bio.PDB.Structure): input Biopython Structure object
        pdb_fname (str): path to the output PDB file
        select_class: Biopython select class, which will be used to modify saved output
    """
    io = PDBIO()
    io.set_structure(struct)
    for residue in struct.get_residues():
        if len(residue.resname) > 3:
            residue.resname = residue.resname[:3]
    if select_class:
        io.save(pdb_fname, select=select_class)
    else:
        io.save(pdb_fname)


def download_pdb(pdb_id: str, output_directory: str = mkdtemp()) -> PdbFile:
    """ Download mmCIF file from RCSB's PDB website
    Args:
        pdb_id: 4 letter PDB accession code
        output_directory: Where to save output file, defaults to temp directory.
    Returns:
        PdbFile object of downloaded file
    """
    downloader = PDBList()
    filename = downloader.retrieve_pdb_file(
        pdb_code=pdb_id, pdir=output_directory,
    )
    return PdbFile(filename, use_in_pipeline=True)


def download_pdb_list(pdb_ids: list[str], output_directory: str = mkdtemp(), num_threads: int = 12) -> list[PdbFile]:
    """ Download a list of mmCIF files from RCSB's PDB website using multiple threads

    Args:
        pdb_ids: List of 4 letter PDB accession code
        output_directory: Where to save output file, defaults to temp directory.
        num_threads: Number of threads to use in parallel downloading
    Returns:
        list[PdbFile] containing downloaded files
    """
    # Download files
    setup_logging()
    downloader = PDBList()
    # This function doesn't return filenames so we'll need to piece them together ourselves
    downloader.download_pdb_files(
        pdb_codes=pdb_ids, pdir=output_directory, max_num_threads=num_threads,
    )

    # Check which files have been successfully downloaded
    pdb_files, missing_ids = [], []
    for pdb_id in pdb_ids:
        filename = os.path.join(output_directory, f"{pdb_id.lower()}.cif")
        if os.path.exists(filename):
            pdb_files.append(PdbFile(filename, use_in_pipeline=True))
        else:
            missing_ids.append(pdb_id)

    # Log any IDs which didn't download properly
    if missing_ids:
        logging.warning(
            f"Missing output files for PDB IDs: {', '.join(missing_ids)}",
        )

    return pdb_files


def download_single_bio_assembly(pdb_id: str, assembly_id: int, output_dir: str) -> str | None:
    """Download a specific biological assembly mmCIF file from RCSB.

    Args:
        pdb_id: Four-character PDB identifier.
        assembly_id: Biological assembly number.
        output_dir: Directory to write the downloaded file into.
    Returns:
        Path to the downloaded file, or None if the download failed.
    """
    downloader = PDBList(verbose=False)
    filename = downloader.retrieve_assembly_file(
        pdb_id.lower(), assembly_id, pdir=output_dir, file_format='mmcif',
    )
    if filename and os.path.exists(filename):
        return filename
    return None


def download_all_bio_assemblies(
        pdb_ids: list[str],
        output_directory: str,
        num_threads: int = 12,
) -> list[PdbFile]:
    """Download all biological assembly mmCIF files from RCSB for the given PDB IDs.

    The asymmetric unit file is downloaded first (required before assembly retrieval),
    then all biological assemblies for each entry are fetched. Already-present files
    are skipped without re-downloading.

    Args:
        pdb_ids: PDB IDs to download.
        output_directory: Directory to write .cif files into.
        num_threads: Number of parallel download threads (default: 12).
    Returns:
        List of PdbFile objects for all successfully downloaded assemblies.
    """
    os.makedirs(output_directory, exist_ok=True)
    downloader = PDBList(verbose=False)
    downloader.download_pdb_files(
        pdb_ids, pdir=output_directory, file_format='mmCif', max_num_threads=num_threads,
    )
    pdb_ids_lower = {pdb_id.lower() for pdb_id in pdb_ids}
    relevant_assemblies = [
        (code, num) for code, num in downloader.get_all_assemblies()
        if code in pdb_ids_lower
    ]
    pdb_files = []
    for pdb_code, assembly_num in relevant_assemblies:
        filename = downloader.retrieve_assembly_file(
            pdb_code, assembly_num, pdir=output_directory, file_format='mmcif',
        )
        if filename and os.path.exists(filename):
            pdb_files.append(PdbFile(filename, use_in_pipeline=True))
    return pdb_files


def merge_structures(
    struct1: Structure,
    struct2: Structure,
) -> Structure:
    """ Merges two Biopython Structure objects into one. Must only have
    one model in each.

    Args:
        struct1: First Biopython Structure object.
        struct2: Second Biopython Structure object.

    Returns:
        A new Structure object containing all chains, residues, and
        atoms from both input structures.
    """
    # Assert only one model per structure
    if not len(struct1) == len(struct2) == 1:
        raise ValueError('One or both structures have more than 1 model')

    # Create a new Structure object with a Model
    merged_structure = Structure(f'{struct1.id}_{struct2.id}')
    merged_model = Model(0)  # Create a new Model with ID 0
    merged_structure.add(merged_model)

    # Get an alphabet str to reassign chain IDs
    alphabet_str = string.ascii_uppercase

    # Keep track of chain IDs used
    used_ids = {chain.id for chain in struct1.get_chains()}

    # Add struct1 chains
    for chain in struct1.get_chains():
        merged_model.add(chain)

    # Add struct2 chains, renaming if needed
    for chain in struct2.get_chains():
        if chain.id in used_ids:
            chain.id = next(c for c in alphabet_str if c not in used_ids)

        used_ids.add(chain.id)
        merged_model.add(chain)

    return merged_structure
