"""Residue numbering as the source papers write it, for figure labels only.

The simulated constructs are not numbered the way the papers that characterised
each cryptic site number them, and on three targets the difference is large
enough to make a figure unreadable against its own reference. CRBN's W264 is
residue 164 in the simulation, IL-2's R38 is 33, and Bcl-xL's protomer B repeats
protomer A at an offset of 140, so Y45 appears as both 45 and 185.

Every figure label therefore goes through here, and every selection, file name,
CSV column and PyMOL `resi` string stays in simulation numbering, because those
have to address the actual files. The mapping is the one tabulated in
tracked_residues.csv.

Bcl-xL is the one target where the offset depends on the chain rather than only
on the target, since its two protomers map onto the same literature numbers. Its
labels carry the chain letter so the two copies of a residue stay distinct on a
panel that happens to show both.
"""
from __future__ import annotations

from pathlib import Path

from sandcat.analysis.experiments import NON_PROTEIN  # noqa: E402
OFFSET = {'CRBN': 100, 'IL2': 5}
CHAIN_OFFSET = {'BCLXL': {'A': 0, 'B': -140}}


def shift_for(target: str, chain: str | None = None) -> int:
    """How much to add to a simulation residue number to get the paper's.

    Args:
        target: Target name, as used throughout the analysis.
        chain: Chain letter, needed only for targets in CHAIN_OFFSET whose
            protomers map onto the same literature numbers. Defaults to chain A
            for those targets and is ignored for every other target.

    Returns:
        The offset to add, zero for a target with no recorded difference.
    """
    if target in CHAIN_OFFSET:
        return CHAIN_OFFSET[target].get(chain or 'A', 0)
    return OFFSET.get(target, 0)


def label(target: str, resname: str, resseq: int,
          chain: str | None = None) -> str:
    """A figure label such as TRP264, in the numbering the paper uses.

    Args:
        target: Target name.
        resname: Three-letter residue name, any case.
        resseq: Residue number in the simulation's own numbering.
        chain: Chain letter. Appended to the label for an oligomer whose
            protomers share literature numbers, so the two copies stay distinct.

    Returns:
        The label, for example 'TRP264' or 'TYR45 B'.
    """
    suffix = f' {chain}' if target in CHAIN_OFFSET and chain else ''
    return f'{resname.upper()}{resseq + shift_for(target, chain)}{suffix}'


def labels_from_topology(target: str, topology) -> dict[int, str]:
    """Label per simulation resSeq, taking the chain from an mdtraj topology.

    mdtraj numbers chains in index order, so chain index 0 is written as A.

    Args:
        target: Target name.
        topology: An mdtraj topology whose residues carry resSeq and chain.

    Returns:
        Simulation resSeq mapped to its literature label, solvent and probe
        residues excluded.
    """
    out = {}
    for residue in topology.residues:
        if residue.name in NON_PROTEIN:
            continue
        chain = chr(ord('A') + residue.chain.index)
        out[residue.resSeq] = label(target, residue.name, residue.resSeq, chain)
    return out


def labels_from_apo(target: str, apo_pdb: str) -> dict[int, str]:
    """The same mapping, reading chain ids straight out of a PDB.

    Args:
        target: Target name.
        apo_pdb: Path to the apo input structure.

    Returns:
        Simulation resSeq mapped to its literature label, solvent and probe
        residues excluded.
    """
    out = {}
    for line in Path(apo_pdb).read_text().splitlines():
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        resseq, name, chain = int(line[22:26]), line[17:20].strip(), line[21]
        if name in NON_PROTEIN:
            continue
        out[resseq] = label(target, name, resseq, chain)
    return out
