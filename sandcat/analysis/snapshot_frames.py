"""PyMOL sessions showing the open state of each target's matched candidate.

The structures are not chosen here. run_dynamics_all writes one open frame per
candidate, picked from the top movers at that candidate with no reference to any
answer key, and this reads the one belonging to the candidate that matches the
held-out pocket. So <TARGET>_open.pdb is the same structure as
dynamics/lambda1/<TARGET>/candidate<rank>_open.pdb, by construction rather than
by coincidence: the two cannot drift because there is only one implementation.

The closed state is the apo input, loaded straight from the simulation inputs.
It is the structure the run started from, it does not move if the equilibration
is repeated, and it is what a reader can download. No closed frame is written.

What does use the source papers' residues is the session rather than the
science: which side chains get sticks, and where the camera points. Choosing
which candidate to show uses the reference too, since that is the definition of
the matched one. Both make these validation figures rather than prospective
output, which is why the blind per-candidate set exists alongside them.

Everything comes from state 0, the unscaled Hamiltonian, and the candidate
list is the lambda = 1 difference detector's.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import mdtraj
import numpy as np
import pandas as pd


from sandcat.analysis.experiments import (target_dir,
                         HOTSPOTS_FILE, STATE_TOPOLOGY,
                         closest_approach,
                         STATE_TRAJECTORY,
                         ANALYSIS, SIM_APO, TARGET_RESIDUES, analysis_relative,
                         residue_distance_traces,
                         apo_residues, best_candidate, bundled_apo,
                         pml_preamble)

warnings.filterwarnings('ignore')

OUT = ANALYSIS / 'snapshots'
TEAL, LIGHT_TEAL, RED, GREY = '0x0DBFBF', '0x87DEDE', '0xF56170', '0x9aa0a6'


def snapshots_for(target: str) -> dict | None:
    """The blind open frame for the candidate matching this target's pocket.

    The frame itself is not chosen here. run_dynamics_all writes one per
    candidate, picked from the top movers at that candidate with no reference to
    any answer key, and this reads the one belonging to the matched candidate.
    Choosing which candidate to read is the only thing that consults the
    reference, which is what makes these sessions validation figures rather than
    prospective output.

    Args:
        target: Target name.

    Returns:
        A dict carrying the target, the matched candidate rank, the open frame
        and its index, the apo path, the residues to draw and how far each one
        moved; or None if no candidate matched or the blind frame is missing.
    """
    best = best_candidate(target)
    if best is None:
        return None
    _, rank, _ = best
    folder = target_dir(target) / 'density'
    row = pd.read_csv(folder / HOTSPOTS_FILE).query('rank == @rank')
    centre = row[['x', 'y', 'z']].to_numpy(float)[0]

    blind = (ANALYSIS / 'dynamics' / 'lambda1' / target
             / f'candidate{rank}_open.pdb')
    if not blind.exists():
        print(f'  {target}: no blind open frame for candidate {rank}; '
              f'run run_dynamics_all first')
        return None

    named = sorted(TARGET_RESIDUES.get(target, []))
    if not named:
        return None

    traj = mdtraj.load(str(folder / STATE_TRAJECTORY),
                       top=str(folder / STATE_TOPOLOGY))
    traces = residue_distance_traces(traj, centre, set(named))
    # mdtraj writes chain ids in index order, so chain 0 is A
    chains = {r.resSeq: chr(ord('A') + r.chain.index)
              for r in traj.topology.residues if r.resSeq in traces}
    if not traces:
        return None

    # which frame the dynamics stage picked, recovered by matching coordinates
    # rather than assumed, so a mismatch is visible instead of silent
    protein = traj.atom_slice(traj.topology.select('protein'))
    reference = mdtraj.load(str(blind))
    deltas = np.abs(protein.xyz - reference.xyz[0]).max(axis=(1, 2))
    opened = int(deltas.argmin())
    if deltas[opened] > 1e-4:
        print(f'  {target}: candidate {rank} open frame does not match any '
              f'frame of state 0 (closest {deltas[opened]:.2e} nm)')

    OUT.mkdir(exist_ok=True)
    protein[opened].save_pdb(str(OUT / f'{target}_open.pdb'))

    # the closed reference is the apo input, the structure the simulation
    # started from and the one a reader can download, not a frame picked out of
    # the same trajectory
    apo = apo_residues(str(SIM_APO[target]))

    moves = {}
    for q, v in traces.items():
        start = (closest_approach(apo[q], centre)
                 if q in apo else float('nan'))
        moves[q] = (float(v[opened]) - start, start, float(v[opened]))
    return {'target': target, 'rank': rank, 'opened': opened,
            # the copy inside the analysis directory, so the session stays
            # loadable once the directory is moved or published
            'apo': str(bundled_apo(target)),
            'n_frames': traj.n_frames, 'moves': moves, 'residues': sorted(traces),
            'chains': chains}


def write_pml(info: dict) -> None:
    """Write the session overlaying one target's open frame on its apo.

    Args:
        info: The dict snapshots_for returns, carrying the target name, the
            candidate rank, the chosen frame and its index, the apo path,
            the residues to draw as sticks and how far each one moved.
    """
    target, resi = info['target'], '+'.join(str(r) for r in info['residues'])
    lines = [
        '# ' + '=' * 68,
        f'#  {target}: closed and open snapshots from state 0 (lambda = 1)',
        '#',
        f'#  Open: frame {info["opened"]} of {info["n_frames"]}, the blind open',
        f'#  frame the dynamics stage chose for candidate {info["rank"]} from the',
        '#  top movers at that candidate, with no reference to the residues any',
        '#  paper names. It is the same file as',
        f'#  dynamics/lambda1/{info["target"]}/candidate{info["rank"]}_open.pdb.',
        '#  Closed: the apo input structure the simulation started from.',
        f'#  Candidate {info["rank"]} of the lambda = 1 difference map.',
        '#',
        '#  LIGHT TEAL  open, TEAL the carbons of its target residues',
        '#  GREY        apo, dark grey the carbons of its target residues',
        '#  BLUE / RED / YELLOW  N, O and S on both, so a side chain reads as',
        '#  chemistry while its carbons still say which structure it came from',
        '# ' + '=' * 68,
        '',
        *pml_preamble(OUT / f'{target}_snapshots.pml'),
        'set_color ttx_teal,       [0.05, 0.75, 0.75]',
        'set_color ttx_light_teal, [0.53, 0.87, 0.87]',
        'set_color ttx_blue,       [0.25, 0.56, 0.77]',
        'set_color ttx_red,        [0.96, 0.38, 0.44]',
        'set_color ttx_yellow,     [1.00, 0.78, 0.40]',
        '',
        f'load {analysis_relative(OUT / f"{target}_open.pdb")},   {target}_open',
        f'load {analysis_relative(info["apo"])}, {target}_apo',
        'hide everything',
        f'show cartoon, {target}_open or {target}_apo',
        'hide everything, elem H',
        f'align polymer and {target}_apo, polymer and {target}_open',
        '',
        f'color ttx_light_teal, {target}_open',
        f'color gray70, {target}_apo',
        '',
        f'select open_res,   {target}_open and resi {resi}',
        f'select closed_res, {target}_apo and resi {resi}',
        'show sticks, (open_res or closed_res) and not elem H',
        '# carbons carry the identity of their own structure, so a stick can be',
        '# traced back to the open or the apo copy at a glance',
        'color ttx_teal, open_res and elem C',
        'color gray40, closed_res and elem C',
        '# heteroatoms are shared between the two, by element rather than by',
        '# structure: colouring an oxygen teal or grey would read as a carbon',
        'color ttx_blue,   (open_res or closed_res) and elem N',
        'color ttx_red,    (open_res or closed_res) and elem O',
        'color ttx_yellow, (open_res or closed_res) and elem S',
        'set stick_radius, 0.22, open_res or closed_res',
        'deselect',
        '',
        'bg_color white',
        'set ray_shadows, 0',
        'set cartoon_fancy_helices, 1',
        'set cartoon_transparency, 0.25',
        f'orient {target}_open and resi {resi}',
    ]
    (OUT / f'{target}_snapshots.pml').write_text('\n'.join(lines) + '\n')
    write_protomer_pml(info, lines)


def write_protomer_pml(info: dict, lines: list[str]) -> None:
    """For a homo-oligomer, the same session restricted to one protomer.

    The candidate sits on one chain, so the other contributes a second copy of
    the protein and a second set of sticks some 45 A away on Bcl-xL, which the
    camera then has to frame around. Which chain to keep is read off the data
    rather than named: it is the chain of the residue closest to the candidate
    in the apo structure.

    Args:
        info: The dict snapshots_for returns.
        lines: The full session, which this trims to one protomer.
    """
    target, chains = info['target'], info.get('chains', {})
    if len(set(chains.values())) < 2:
        return
    # moves[q] = (change, apo distance, open distance); the apo distance says
    # which protomer the candidate actually sits on
    nearest = min(info['moves'], key=lambda q: info['moves'][q][1])
    keep = chains[nearest]
    residues = sorted(q for q in info['residues'] if chains.get(q) == keep)
    resi = '+'.join(str(r) for r in residues)

    out = []
    for line in lines:
        if line.startswith('orient '):
            line = f'orient {target}_open and chain {keep} and resi {resi}'
        elif line.startswith('show cartoon,'):
            line = (f'show cartoon, ({target}_open or {target}_apo) '
                    f'and chain {keep}')
        elif line.startswith('select open_res,'):
            line = f'select open_res,   {target}_open and chain {keep} and resi {resi}'
        elif line.startswith('select closed_res,'):
            line = f'select closed_res, {target}_apo and chain {keep} and resi {resi}'
        elif line.startswith('align polymer'):
            line = (f'align polymer and {target}_apo and chain {keep}, '
                    f'polymer and {target}_open and chain {keep}')
        elif line.startswith(f'#  {target}:'):
            line = f'#  {target}: protomer {keep} only, closed and open snapshots'
        out.append(line)
    # everything on the other protomer is hidden rather than unloaded, so the
    # file still documents that the second copy exists
    index = next(i for i, l in enumerate(out) if l.startswith('bg_color'))
    out[index:index] = [
        '# the other protomer is hidden, not unloaded: the candidate is on',
        f'# chain {keep} and the second copy only adds a distant duplicate',
        f'hide everything, not chain {keep}',
        '']
    (OUT / f'{target}_snapshots_protomer{keep}.pml').write_text(
        '\n'.join(out) + '\n')
    print(f'    {target}: protomer {keep} session, residues {resi}')


def main() -> None:
    """Write a session per characterised target, and the movement table.

    Targets with no candidate matching their reference are skipped, which
    is every target with no known residues and ubiquitin, the negative
    control, which has no candidates at all.
    """
    rows = []
    for target in TARGET_RESIDUES:
        info = snapshots_for(target)
        if info is None:
            print(f'  {target:9s} no candidate or no target residues')
            continue
        write_pml(info)
        widest = max(info['moves'].items(), key=lambda kv: abs(kv[1][0]))
        print(f'  {info["target"]:9s} open frame {info["opened"]:3d} of '
              f'{info["n_frames"]}, largest change {widest[0]} '
              f'{widest[1][1]:.1f} -> {widest[1][2]:.1f} A (apo -> open)')
        for resseq, (delta, closed, opened) in sorted(info['moves'].items()):
            rows.append({'target': info['target'], 'resSeq': resseq,
                         'apo_a': round(closed, 2), 'open_a': round(opened, 2),
                         'change_a': round(delta, 2)})
    if rows:
        pd.DataFrame(rows).to_csv(ANALYSIS / 'snapshot_residue_changes.csv',
                                  index=False)
        print(f'\nwritten: {OUT}/ and snapshot_residue_changes.csv')


if __name__ == '__main__':
    main()
