"""Per-residue motion on the lambda = 1 ensemble, independent of any candidate.

Proximity alone does not distinguish a gate from a buried rigid residue beside
a density blob: both sit close to a candidate centre and both score as
recovered. This module supplies a motion flag per residue, so recovery can
require proximity and movement together.

Two things are measured and they catch different mechanisms.

Rotamer switching counts transitions between populated states of chi1, chi2 and
chi3. chi2 is folded onto 180 degrees for the twofold-symmetric groups, the
Phe and Tyr rings and the Asp carboxylate, which is correct but means a ring
reorientation driven by chi1 registers as nothing in chi2, so chi1 is always
tested on the full 360 degree period. chi3 is included because it is the dihedral that moves the
terminal group of a long side chain, and TEM1's ARG222 switches chi3 while chi1
and chi2 stay put.

Displacement catches what rotamers miss. TEM-1's main site is gated by backbone
movement and its rank-1 hotspot has twenty residues within 7 A of which none is
aromatic, so a dihedral analysis has nothing to measure there. Displacement is
the largest separation between any two positions of the side-chain centroid,
which needs no site centre and is therefore a property of the residue rather
than of a candidate.

Everything is computed on state 0 alone, the only state whose Hamiltonian is
unscaled. Aggregating across the ladder instead would be dominated by
lambda = 1.35, where TEM1's LEU221 spans 32.8 A against 8.8 A at lambda = 1, and
would measure how far a residue moves when the Hamiltonian is heated.

State 0 is not a canonical ensemble: like every state it carries an OPES
Multithermal bias over 300-350 K. Every quantity here is an ensemble property
rather than a rate, so the bias does not affect them, and the choice of
state 0 rests on the Hamiltonian being unscaled rather than on the ensemble
being Boltzmann.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import mdtraj
import numpy as np
import pandas as pd


from sandcat.analysis.experiments import (NON_HEAVY_ELEMENTS, NON_PROTEIN,
                                          PRODUCTION_STATE)
from sandcat.analysis.pocket_residue_dynamics import (SYMMETRIC_CHI2,
                                                      SYMMETRIC_CHI3,
                                                      chi_angles,
                                                      rotamer_switches)

warnings.filterwarnings('ignore')

BACKBONE = ('N', 'CA', 'C', 'O', 'OXT')

# Motion is scored by rank within a target, not by an absolute cut: over 251
# frames of lambda = 1 sampling 94 per cent of residues travel at least 2 A and
# 75 per cent switch rotamer at least once, so no fixed threshold separates a
# gate from the bulk.


def side_chain_span(trajectory: mdtraj.Trajectory, atoms: list[int]) -> float:
    """Diameter of the region the side-chain centroid explores.

    Measured as the largest distance between any two centroid positions, which
    needs no reference structure and no reference frame. That matters on a
    demultiplexed trajectory, where frame 0 is whichever state held the
    lambda = 1 slot at the first checkpoint rather than a defined reference. It
    is a spatial extent rather than a radial one, so a side chain sweeping an
    arc at constant distance from a reference point counts as having moved.

    Args:
        trajectory: Aligned trajectory to measure over.
        atoms: Atom indices of the side chain.

    Returns:
        Maximum pairwise separation of the centroid, in Angstroms.
    """
    if not atoms:
        return 0.0
    centroid = trajectory.xyz[:, atoms, :].mean(axis=1) * 10.0
    separations = np.linalg.norm(
        centroid[:, None, :] - centroid[None, :, :], axis=-1,
    )
    return float(separations.max())


def motion_table(density_dir: Path, state: int = PRODUCTION_STATE) -> pd.DataFrame:
    """One row per residue: rotamer switches, side-chain travel, moving flag.
    Args:
        density_dir: Density directory holding the aligned trajectories.
        state: Thermodynamic state to read. 0 is lambda = 1.

    Returns:
        One row per residue with resSeq, resname, switch counts and span_a.
    """
    top = density_dir / f'density_swish_{state}_top.pdb'
    dcd = density_dir / f'swish_{state}_aligned.dcd'
    if not (top.exists() and dcd.exists()):
        return pd.DataFrame()
    trajectory = mdtraj.load(str(dcd), top=str(top))

    # NON_PROTEIN, not a list of its own: the capping groups carry a resSeq
    # and their methyl is not a backbone atom, so a local list that forgot
    # them gave ACE and NME a side-chain span and the top mobility rank.
    residues = [r for r in trajectory.topology.residues
                if r.name not in NON_PROTEIN]
    indices = [r.index for r in residues]
    angles = chi_angles(trajectory, indices)

    rows = []
    for residue in residues:
        chi = angles[angles.residue_index == residue.index]
        if len(chi):
            # The period is chosen per residue, not fixed: chi_angles folds
            # chi2 onto [0, 180) only for the twofold-symmetric Phe and Tyr
            # rings and the Asp carboxylate, and chi3 only for glutamate's.
            # Passing 180 for an asymmetric dihedral merges a rotamer with the
            # one 180 degrees away, so a side chain holding 60 then -120
            # registers no switch at all. describe_candidate reads the same
            # two constants, so both counters bin a dihedral the same way.
            chi2_period = 180.0 if residue.name in SYMMETRIC_CHI2 else 360.0
            chi3_period = 180.0 if residue.name in SYMMETRIC_CHI3 else 360.0
            flips = (rotamer_switches(chi.chi1.to_numpy(), 360.0)
                     + rotamer_switches(chi.chi2.to_numpy(), chi2_period)
                     + rotamer_switches(chi.chi3.to_numpy(), chi3_period))
        else:
            flips = 0
        side = [a.index for a in residue.atoms
                if a.element.symbol not in NON_HEAVY_ELEMENTS
                and a.name not in BACKBONE]
        span = side_chain_span(trajectory, side)
        rows.append({'resSeq': residue.resSeq, 'resname': residue.name,
                     'flips': int(flips), 'span_a': round(span, 2)})

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # one row per resSeq: a residue number can repeat across chains, and the
    # question is whether that residue moves anywhere in the assembly
    table = (table.groupby(['resSeq', 'resname'], as_index=False)
             .agg(flips=('flips', 'max'), span_a=('span_a', 'max')))
    table['span_rank'] = table.span_a.rank(ascending=False, method='min')
    table['flip_rank'] = table.flips.rank(ascending=False, method='min')
    # Ranked on each measure separately. The classification that reaches the
    # paper splits at the median of the residue's own protein, in
    # residue_ranking.motion_class; nothing here applies a cut of its own.
    return table.sort_values(['span_rank', 'flip_rank']).reset_index(drop=True)
