"""Run the pocket-dynamics stage on every target, for the lambda = 1 candidates.

Separate from the density stage because it reads the aligned SWISH trajectories
that stage writes, and because a failure on one target should not lose the rest.
It produces, per target, pocket_dynamics.csv and one open frame per candidate
with a session to view it against the apo.

Only the lambda = 1 detector is characterised. State 0 is the only rung of
the ladder whose Hamiltonian is unscaled, so it is the only candidate list the
analysis may use; running the pooled or unbiased lists here as well would
produce several sets of open frames per target that disagree with one another.

This stage is blind. No residue list from a source paper reaches it: each
candidate's open frame is driven by the top movers at that candidate, so the
output is what the method produces on a target nobody has characterised.
snapshot_frames reads these frames back for the one candidate that matches a
held-out reference, which is the only place prior knowledge enters.

The residues eligible to drive a candidate's opening coordinate are those
within GATING_RADIUS_A of its centre in the apo input, a different and tighter
test than the RESIDUE_A median over the trajectory that decides which residues
the figure stages report at that candidate. Those residues are ranked by how
far they move, so the apo fixes the pool before any motion is measured. The
protomer each candidate is assigned to also comes from the apo, since that
superposition must not move between runs.

N_CANDIDATES bounds how far down the ranked list each target is
characterised, and is imported from experiments so that this stage and the
matched-site search cannot disagree about how deep the list goes. It has to be
set generously: a site can sit well below rank 1, and on this benchmark Bcl-xL's
is candidate 4 and PLK1's candidate 6, so a smaller value silently produces a
table with no rows for them.

MDM2 was run with five SWISH replicas on purpose; every other target has six.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path


from sandcat.analysis.analyse_pocket_dynamics import run

from sandcat.analysis.build_density_maps import SIMULATION_INPUTS
from sandcat.analysis.experiments import (target_dir,
                         ANALYSIS, GATING_RADIUS_A, HOTSPOTS_FILE,
                         N_CANDIDATES,
                         POCKET_REFERENCES, bundled_apo)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# How many thermodynamic states each target ran, taken from the density
# stage's own table rather than declared again here: MDM2 ran five and
# every other target six, and two copies of that fact can disagree.
# the one target set, from experiments, so adding a target is a single edit
TARGETS = tuple(sorted(POCKET_REFERENCES))


def main() -> None:
    """Characterise every target's candidates, writing one directory each.

    A target whose density stage produced no candidate table is skipped with a
    warning. A target that raises is recorded and stepped over, since the
    targets are independent and finishing the rest is useful, but any failure
    makes the stage exit non-zero at the end: the stages after this one read the
    frames it writes, and on a reused output directory a silent failure here
    leaves them consuming an earlier run's output.

    Raises:
        SystemExit: If any target failed, after every target has been attempted.
    """
    failed = []
    for target in TARGETS:
        density = target_dir(target) / 'density'
        if not (density / HOTSPOTS_FILE).exists():
            logging.warning(f'{target}: no {HOTSPOTS_FILE}, skipped')
            continue
        logging.info(f'===== {target} =====')
        try:
            run(str(density), str(ANALYSIS / 'dynamics' / 'lambda1' / target),
                n_candidates=N_CANDIDATES, radius_a=GATING_RADIUS_A,
                n_replicas=SIMULATION_INPUTS[target][3],
                hotspots_file=HOTSPOTS_FILE,
                # the copy inside the analysis directory, so the sessions this
                # writes stay loadable once the directory is moved or published
                apo_pdb=str(bundled_apo(target)))
        except Exception:
            failed.append(target)
            logging.error(f'{target} failed:\n{traceback.format_exc()}')

    if failed:
        raise SystemExit(
            f'{len(failed)} target(s) failed: {", ".join(failed)}. Exiting '
            f'non-zero so snapshot_frames and the figure stages do not run '
            f'against missing or stale per-candidate frames.')
    print('DYNAMICS DONE')


if __name__ == '__main__':
    main()
