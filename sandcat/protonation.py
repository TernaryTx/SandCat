from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from propka.run import single as propka_single_run

from sandcat.ligand import LigFile
from sandcat.structure import PdbFile
from sandcat.structure import Processor
from sandcat.pdb_io import read_pdb
from sandcat.pdb_io import write_pdb

# Dict mapping residue names to their two different microstates
# Tuple is predominant state below and above pKa value
RESNAME_TO_MICROSTATE_NAMES: dict[str, tuple[str, str]] = {
    'HIS': ('HIP', 'HIS'),
    'ASP': ('ASH', 'ASP'),
    'GLU': ('GLH', 'GLU'),
    'LYS': ('LYS', 'LYN'),
    'ARG': ('ARG', 'ARN'),
    'CYS': ('CYS', 'CYM'),
    'TYR': ('TYR', 'TYD'),
}

# from openmmforcefields/ffxml/amber/phosaa14SB.xml
PHOSPHORYLATED_RESNAMAES = [
    'SEP', 'S1P', 'TPO', 'T1P', 'Y1P', 'PTR',
    'H1D', 'H2D', 'H1E', 'H2E',
]


class ProteinProtonater(Processor):
    def __init__(
            self,
            ph: float = 7.4,
            cap_termini: bool = False,
            output_dir: str | None = None,
    ):
        """Args:
            ph: What pH to run propka calculations on, defaults to 7.4
            cap_termini: If True, prepend ACE and append NME caps to every
                protein chain in the input before tleap adds hydrogens.
        """
        self.ph = ph
        self.cap_termini = cap_termini
        super().__init__(output_dir)

    def _run_propka(self, input_file: PdbFile, ph=7.4) -> dict[str, float]:
        """Run propka on a single protein structure file and returns a pka dict
        Args:
            ph: pH to calculate protonation state, default 7.4
            input_file: PdbFile object representing the protein structure.
        Returns:
            A dictionary of residue label to pKa value.
        """
        mol = propka_single_run(
            input_file.path, write_pka=False,
            optargs=[f'--pH={ph}', '--log-level=CRITICAL'],
        )
        conf = mol.conformations['AVR']
        res_groups = conf.get_groups_for_calculations()
        pka_dict = {group.label: group.pka_value for group in res_groups}
        return pka_dict

    def _run_tleap_add_hydrogens(self, input_pdb: str, output_pdb: str) -> None:
        """Calls tleap from AmberTools to add missing hydrogens, according to protonation state, and adds correct hydrogens to N-termini.
        Args:
            input_pdb: Path to PDB with updated protonation state residue names from propka.
            output_pdb: Path to final PDB with hydrogens added.
        """
        tleap_script = f"""
        source leaprc.protein.ff14SB
        mol = loadPDB "{input_pdb}"
        savePDB mol "{output_pdb}"
        quit
        """
        # Run tleap
        subprocess.run(
            ['tleap', '-f', '/dev/stdin'],
            input=tleap_script.encode(),
            capture_output=True,
            check=True,
        )

    @staticmethod
    def _build_ace_residue(first_res: Residue) -> Residue:
        """Build an ACE residue (CH3, C, O heavy atoms) extrapolated from the first residue's backbone.

        Places the carbonyl C 1.33 Å beyond N along the CA→N direction,
        with O and CH3 at ~120° in the residue plane. tleap fills in the
        three methyl Hs from these heavy atoms.
        """
        n_pos = np.array(first_res['N'].coord, dtype=float)
        ca_pos = np.array(first_res['CA'].coord, dtype=float)
        out = n_pos - ca_pos
        out /= np.linalg.norm(out)
        if 'C' in first_res:
            helper = np.array(first_res['C'].coord, dtype=float) - ca_pos
        else:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, out)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
        perp = np.cross(np.cross(out, helper), out)
        perp /= np.linalg.norm(perp)

        c_coord = n_pos + 1.33 * out
        o_coord = c_coord + 1.23 * (0.5 * out + 0.866 * perp)
        ch3_coord = c_coord + 1.51 * (0.5 * out - 0.866 * perp)

        ace = Residue((' ', 0, ' '), 'ACE', ' ')
        for name, coord, element in (('CH3', ch3_coord, 'C'), ('C', c_coord, 'C'), ('O', o_coord, 'O')):
            ace.add(
                Atom(
                    name, coord, 0.0, 1.0,
                    ' ', f' {name:<3}', 0, element,
                ),
            )
        return ace

    @staticmethod
    def _build_nme_residue(last_res: Residue) -> Residue:
        """Build an NME residue (N, CH3 heavy atoms) extrapolated from the last residue's backbone.

        Places the amide N 1.33 Å beyond C along the CA→C direction, then
        CH3 1.45 Å further out. tleap fills in the amide H and three
        methyl Hs from these heavy atoms.
        """
        c_pos = np.array(last_res['C'].coord, dtype=float)
        ca_pos = np.array(last_res['CA'].coord, dtype=float)
        out = c_pos - ca_pos
        out /= np.linalg.norm(out)
        if 'O' in last_res:
            helper = np.array(last_res['O'].coord, dtype=float) - ca_pos
        else:
            helper = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(helper, out)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0])
        perp = np.cross(np.cross(out, helper), out)
        perp /= np.linalg.norm(perp)

        n_coord = c_pos + 1.33 * out
        ch3_coord = n_coord + 1.45 * (0.5 * out + 0.866 * perp)

        # tleap's NME library names the methyl carbon ``C`` (no second carbon
        # to disambiguate from, unlike ACE).
        nme = Residue((' ', 0, ' '), 'NME', ' ')
        for name, coord, element in (('N', n_coord, 'N'), ('C', ch3_coord, 'C')):
            nme.add(
                Atom(
                    name, coord, 0.0, 1.0,
                    ' ', f' {name:<3}', 0, element,
                ),
            )
        return nme

    @staticmethod
    def _rename_cap_atoms_for_openmm(structure: Structure) -> Structure:
        """Rename ACE / NME atoms in ``structure`` to match OpenMM amber14-all.xml.

        tleap emits cap methyl Hs as H1/H2/H3 and NME's methyl carbon as C;
        amber14-all.xml expects HH31/HH32/HH33 and CH3. Without these
        renames the capped PDB fails ``ForceField.createSystem`` with
        "atoms match X, but bonds are different".
        """
        cap_renames = {
            'ACE': {'H1': 'HH31', 'H2': 'HH32', 'H3': 'HH33'},
            'NME': {'C': 'CH3', 'H1': 'HH31', 'H2': 'HH32', 'H3': 'HH33'},
        }
        for res in structure.get_residues():
            mapping = cap_renames.get(res.resname)
            if mapping is None:
                continue
            for old_name, new_name in mapping.items():
                if old_name not in res:
                    continue
                atom = res[old_name]
                res.detach_child(old_name)
                atom.name = new_name
                atom.id = new_name
                atom.fullname = f' {new_name:<3}'
                res.add(atom)
        return structure

    def _insert_ace_nme_caps(self, structure: Structure) -> Structure:
        """Insert ACE / NME caps around every protein chain in ``structure``.

        Mutates ``structure`` in place. For each chain that contains any
        ATOM (non-HETATM) residues, prepends an ACE and appends an NME
        with extrapolated peptide-bond geometry, strips terminus-only
        atoms (free-N-terminus H1/H2/H3 and C-terminal OXT), and
        renumbers the chain sequentially from 1. HETATM-only chains
        (e.g. ligand or water chains) are left untouched.
        """
        for chain in structure.get_chains():
            residues = [res for res in chain if res.id[0] == ' ']
            if not residues:
                continue
            first, last = residues[0], residues[-1]
            for needed in ('N', 'CA'):
                if needed not in first:
                    raise ValueError(
                        f'Chain {chain.id} first residue missing {needed!r}; cannot place ACE.',
                    )
            for needed in ('C', 'CA'):
                if needed not in last:
                    raise ValueError(
                        f'Chain {chain.id} last residue missing {needed!r}; cannot place NME.',
                    )

            ace = self._build_ace_residue(first)
            nme = self._build_nme_residue(last)

            for atom_name in ('H1', 'H2', 'H3'):
                if atom_name in first:
                    first.detach_child(atom_name)
            if 'OXT' in last:
                last.detach_child('OXT')

            existing = list(chain.child_list)
            max_seq = max(res.id[1] for res in existing)
            ace.id = (' ', max_seq + 1, ' ')
            nme.id = (' ', max_seq + 2, ' ')

            for res in existing:
                chain.detach_child(res.id)
            chain.add(ace)
            for res in existing:
                chain.add(res)
            chain.add(nme)

            for serial, res in enumerate(chain.child_list, start=1):
                res.id = (' ', serial, res.id[2])
        return structure

    def _modify_resname_prot_state(self, pka_dict: dict[str, float], structure: Structure, ph: float = 7.4) -> Structure:
        """Modifies residue names of a Biopython Structure object for charged states based on pka_dict.
        Args:
            pka_dict: A dictionary mapping residue labels to pKa values from propka.
            structure: Biopython Structure object of uncharged PDB.
            ph: Reference pH to determine protonation state (default: 7.4).
        Returns:
            The Biopython Structure object with charged residues.
        """
        for res in structure.get_residues():
            pdb_string = f"{res.resname}{res.id[1]:>4}{res.parent.id:>2}"
            pka = pka_dict.get(pdb_string, None)
            residue_microstates = RESNAME_TO_MICROSTATE_NAMES.get(res.resname)
            if pka is not None and residue_microstates:
                res.resname = residue_microstates[0] if pka > ph else residue_microstates[1]
                if abs(pka - ph) < 0.5:
                    logging.warning(
                        f"Protonation state of residue {pdb_string} with a pka of {pka}"
                        f" is very close to the target pH of {ph}",
                    )
        return structure

    def _reconstruct_chains(self, template_structure: Structure, flat_structure: Structure) -> Structure:
        """Reconstructs the Chain/Residue hierarchy of a flat structure (from tleap)
        using a template structure (original input) as the guide.

        Args:
            template_structure: The original structure with correct Chain IDs and Residue IDs.
            flat_structure: The structure from tleap (linear, missing correct Chain/Residue IDs).

        Returns:
            A new Structure object with the atoms/residues from flat_structure mapped
            onto the hierarchy of template_structure.
        """
        # Create a new structure to hold the final result
        reconstructed_structure = Structure(flat_structure.id)
        reconstructed_model = Model(0)
        reconstructed_structure.add(reconstructed_model)

        # Get a linear iterator of the protonated residues
        flat_residues = list(flat_structure.get_residues())
        flat_res_iter = iter(flat_residues)

        # Validate counts before starting
        template_res_count = len(list(template_structure.get_residues()))
        if len(flat_residues) != template_res_count:
            raise ValueError(
                f'Residue count mismatch: Template has {template_res_count}, '
                f'but tleap output has {len(flat_residues)}',
            )

        for template_chain in template_structure.get_chains():
            new_chain = Chain(template_chain.id)
            reconstructed_model.add(new_chain)

            for template_res in template_chain.get_residues():
                # Get the next protonated residue from the flat list and restore original Residue ID
                protonated_res = next(flat_res_iter)
                protonated_res.id = template_res.id

                # Detach from the old parent and add to our new chain
                protonated_res.detach_parent()
                new_chain.add(protonated_res)

        return reconstructed_structure

    def _check_for_phosphoresidues(self, structure: Structure) -> None:
        """Checks for phosphorylated residues in a structure and raises an error if any are found.
        Raises:
            ValueError: If any known phosphorylated residue is detected."""
        if any([res.resname.upper() in PHOSPHORYLATED_RESNAMAES for res in structure.get_residues()]):
            raise ValueError(
                'Phosphorylated residue detected. Prepare this protein separately.',
            )

    def process_one(self, input_file: PdbFile, ligand_input_file: LigFile | None = None) -> list[PdbFile]:
        """Protonates protein file using propka 3.5 and reduce. This will calculate protonation state with
            respect to any ligands. tleap is used to add the correct hydrogens to N-termini and to residues with
            modified protonation states as calculated by propka. NOTE: This processor only works on PdbFiles containing a single
            model (`Bio.PDB.Model.Model` object)
            TODO: Fix tauteromerization of histidine states
        Args:
            input_file: Input PdbFile type
            ligand_input_file: Not needed for this method
        Returns:
            List containing a single protonated PdbFile
        """
        pka_dict = self._run_propka(input_file, ph=self.ph)
        input_structure = read_pdb(input_file)
        self._check_for_phosphoresidues(input_structure)

        # Rename residues according to assigned protonation state
        modified_structure = self._modify_resname_prot_state(
            pka_dict, input_structure, ph=self.ph,
        )

        if self.cap_termini:
            # Insert heavy-atom ACE and NME residues with peptide-bond geometry
            # so tleap reads them as part of each chain (rather than as
            # separate units after `combine`, which leaves caps at the origin).
            modified_structure = self._insert_ace_nme_caps(modified_structure)

        # Write PDB with modified protonation states (and caps if requested)
        temp_filename = os.path.join(
            self.output_dir, f"{Path(input_file.path).stem}_pretleap_tmp.pdb",
        )
        write_pdb(modified_structure, temp_filename)

        # Run tleap to add correct hydrogens - adds to N-termini without needing explicit renumbering
        final_filename = os.path.join(
            self.output_dir,
            f"{Path(input_file.path).stem}_protein_H_atoms.pdb",
        )
        self._run_tleap_add_hydrogens(temp_filename, final_filename)
        tleap_structure = read_pdb(PdbFile(final_filename))

        # Reconstruct chain layout from the template — tleap's savePDB flattens
        # multi-chain inputs into one chain. modified_structure already has any
        # ACE/NME caps inserted, so the residue counts still match for capped
        # runs.
        final_structure = self._reconstruct_chains(
            template_structure=modified_structure,
            flat_structure=tleap_structure,
        )
        if self.cap_termini:
            final_structure = self._rename_cap_atoms_for_openmm(
                final_structure,
            )

        write_pdb(final_structure, final_filename)

        try:
            os.remove(temp_filename)  # Remove temp file
        except OSError:
            logging.warning(f"Could not delete temporary file {temp_filename}")

        return [PdbFile(final_filename)]
