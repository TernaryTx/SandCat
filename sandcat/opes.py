from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from typing import Iterable

import numpy as np
import openmm as mm
import openmm.unit as unit


METAD = 'metad'
EXPLORE = 'explore'


def _standardize(quantity):
    """Strip OpenMM units, returning the numeric value in the MD unit system."""
    if unit.is_quantity(quantity):
        return quantity.value_in_unit_system(unit.md_unit_system)
    return quantity


class OpesVariable:
    """A collective variable for OPES, wrapping an OpenMM Force.

    The Force's potential energy IS the value of the CV (same convention as
    openmm.app.metadynamics.BiasVariable). ``min_value`` / ``max_value`` set
    the tabulation grid; the bias is zero outside.

    Args:
        force: OpenMM Force whose potential energy is the CV value.
        min_value, max_value: Tabulation grid extents (units allowed).
        periodic: Whether the CV wraps at the boundary (e.g. dihedrals).
        grid_width: Number of grid bins. If None, chosen for ~5 bins per
            initial sigma (KDE heuristic).
        sigma_hint: Initial bandwidth used only as a hint for the
            grid_width default; bandwidth itself lives on the OPES driver.
    """

    def __init__(
            self,
            force: mm.Force,
            min_value: float | unit.Quantity,
            max_value: float | unit.Quantity,
            periodic: bool = False,
            grid_width: int | None = None,
            sigma_hint: float | unit.Quantity | None = None,
    ):
        self.force = force
        self.min_value = _standardize(min_value)
        self.max_value = _standardize(max_value)
        self.periodic = periodic
        if grid_width is None:
            sigma_for_grid = _standardize(sigma_hint) if sigma_hint is not None else (
                (self.max_value - self.min_value) / 100.0
            )
            grid_width = max(50, int(math.ceil(5 * (self.max_value - self.min_value) / sigma_for_grid)))
        self.grid_width = grid_width


@dataclass
class _Kernel:
    """A single OPES kernel (matches OPESmetad.cpp's ``struct kernel``).

    Height absorbs the per-step weight w_n and the sigma_0/sigma rescaling
    for Silverman shrinkage; the 1/sqrt(2*pi)*sigma_0 norm is dropped
    because it cancels in prob/Zed (OPESmetad.cpp:1104-1106).
    """
    height: float
    center: np.ndarray
    sigma: np.ndarray


def _wrap_periodic(diff: np.ndarray, span: np.ndarray, periodic: np.ndarray) -> np.ndarray:
    """Wrap differences for periodic CVs into [-span/2, +span/2]."""
    out = diff.copy()
    if periodic.any():
        # mask is 1 where periodic, 0 where not — vectorised wrap
        per_diff = out[periodic]
        per_span = span[periodic]
        per_diff = per_diff - per_span * np.round(per_diff / per_span)
        out[periodic] = per_diff
    return out


def _resolve_bias_factors(
        biasfactor: float | None, mode: str, barrier_kj: float, kbt: float,
) -> tuple[float, float]:
    """Default biasfactor and derive bias_prefactor (OPESmetad.cpp:399-415)."""
    if biasfactor is None:
        biasfactor = barrier_kj / kbt
    if math.isinf(biasfactor):
        if mode == EXPLORE:
            raise ValueError('BIASFACTOR=inf is not compatible with EXPLORE mode')
        return biasfactor, 1.0
    if biasfactor <= 1.0:
        raise ValueError('BIASFACTOR must be > 1 (or inf for METAD)')
    bias_prefactor = 1.0 - 1.0 / biasfactor if mode == METAD else biasfactor - 1.0
    return biasfactor, bias_prefactor


def _resolve_epsilon_cutoff(
        epsilon: float | None,
        kernel_cutoff: float | None,
        compression_threshold: float,
        mode: str,
        barrier_kj: float,
        kbt: float,
        bias_prefactor: float,
) -> tuple[float, float, float, float]:
    """Default epsilon / kernel cutoff / compression threshold (OPESmetad.cpp:453-470).

    Returns (epsilon, cutoff2, val_at_cutoff, threshold2).
    """
    if epsilon is None:
        epsilon = math.exp(-barrier_kj / bias_prefactor / kbt)
    if epsilon <= 0:
        raise ValueError('EPSILON must be > 0; is BARRIER too high?')

    if kernel_cutoff is None:
        if mode == METAD:
            kernel_cutoff = math.sqrt(2.0 * barrier_kj / bias_prefactor / kbt)
        else:
            kernel_cutoff = math.sqrt(2.0 * barrier_kj / kbt)
    cutoff2 = kernel_cutoff * kernel_cutoff
    val_at_cutoff = math.exp(-0.5 * cutoff2)

    if compression_threshold < 0:
        raise ValueError('COMPRESSION_THRESHOLD must be >= 0')
    if compression_threshold > 0 and compression_threshold ** 2 >= cutoff2:
        raise ValueError('COMPRESSION_THRESHOLD cannot be >= KERNEL_CUTOFF')
    threshold2 = compression_threshold * compression_threshold

    return epsilon, cutoff2, val_at_cutoff, threshold2


def _resolve_initial_sigma(
        sigma: list[float | unit.Quantity] | None,
        ncv: int,
        adaptive_sigma_stride: int | None,
        pace: int,
) -> tuple[bool, np.ndarray, int]:
    """Resolve initial sigma vs adaptive sigma (OPESmetad.cpp:418-450).

    Returns (adaptive_sigma_flag, sigma0, adaptive_sigma_stride).
    """
    if sigma is None:
        if adaptive_sigma_stride is None:
            adaptive_sigma_stride = 10 * pace
        if adaptive_sigma_stride < pace:
            raise ValueError('ADAPTIVE_SIGMA_STRIDE must be >= PACE')
        return True, np.zeros(ncv), adaptive_sigma_stride

    if len(sigma) != ncv:
        raise ValueError('sigma must have one entry per CV')
    if adaptive_sigma_stride is not None and adaptive_sigma_stride > 0:
        raise ValueError('cannot set ADAPTIVE_SIGMA_STRIDE when sigma is fixed')
    sigma0 = np.array([_standardize(s) for s in sigma], dtype=float)
    return False, sigma0, 0


def _build_tabulated_function(
        ncv: int,
        mins: np.ndarray,
        maxs: np.ndarray,
        grid_widths: list[int],
        all_periodic: bool,
) -> mm.TabulatedFunction:
    """Build a zero-valued Continuous{1,2,3}DFunction matching the CV grid.

    SWIG's overload resolution rejects numpy.float64 for double args, so each
    numeric extent is wrapped in ``float()`` / ``int()`` explicitly.
    """
    flat_zeros = np.zeros(int(np.prod(grid_widths))).tolist()
    extents = [(float(mins[i]), float(maxs[i])) for i in range(ncv)]
    if ncv == 1:
        return mm.Continuous1DFunction(flat_zeros, *extents[0], bool(all_periodic))
    if ncv == 2:
        return mm.Continuous2DFunction(
            int(grid_widths[0]), int(grid_widths[1]),
            flat_zeros, *extents[0], *extents[1], bool(all_periodic),
        )
    return mm.Continuous3DFunction(
        int(grid_widths[0]), int(grid_widths[1]), int(grid_widths[2]),
        flat_zeros, *extents[0], *extents[1], *extents[2], bool(all_periodic),
    )


def _pick_free_force_group(system: mm.System) -> int:
    """Highest unused force-group index (raise if all 32 are taken)."""
    free_groups = set(range(32)) - {f.getForceGroup() for f in system.getForces()}
    if not free_groups:
        raise RuntimeError('No free force groups available for the OPES bias')
    return max(free_groups)


class OPES:
    """Drive an OpenMM simulation with an OPES_METAD or OPES_METAD_EXPLORE bias.

    On-the-fly probability enhanced sampling following Invernizzi & Parrinello
    (J. Phys. Chem. Lett. 11, 2731, 2020) for METAD and Invernizzi et al
    (J. Chem. Theory Comput. 18, 3988, 2022) for EXPLORE. Ported from PLUMED
    2.9 ``OPESmetad.cpp``; line numbers in inline comments refer there.

    The bias is a compressed list of Gaussian kernels in CV space, rasterised
    onto a grid every PACE steps and applied as a CustomCVForce wrapping a
    Continuous{1,2,3}DFunction. ``mode`` selects OPES_METAD (reweighting to a
    well-tempered target, default) or OPES_METAD_EXPLORE (uniform-weight KDE
    of the sampled distribution, more robust to suboptimal CVs). The two
    differ only in kernel deposition (weight, KDE normaliser, sigma factor);
    per-step bias evaluation is identical (OPESmetad.cpp:984).

    Args:
        system: System the bias will be added to (in-place).
        variables: 1, 2, or 3 OpesVariable objects.
        temperature: Used for kT and the well-tempered prefactor.
        barrier: Largest free-energy barrier the bias should handle.
            Defaults BIASFACTOR, EPSILON, and the kernel cutoff if those
            are not given.
        pace: Kernel deposition stride in MD steps.
        sigma: Initial kernel bandwidth per CV. If None, adaptive sigma
            uses a 10*pace-step Welford warmup (OPESmetad.cpp:418-433).
        mode: ``METAD`` or ``EXPLORE``.
        biasfactor: gamma. Default = barrier / kT. Use float('inf') for a
            uniform target (METAD only).
        epsilon: Log regulariser. Default exp(-barrier/bias_prefactor/kT)
            caps bias depth at ``barrier`` (OPESmetad.cpp:453).
        kernel_cutoff: Truncation distance in sigma units. Default
            sqrt(2*barrier/bias_prefactor/kT) for METAD,
            sqrt(2*barrier/kT) for EXPLORE.
        compression_threshold: Mahalanobis merging threshold in sigma
            units. 0 disables merging.
        no_zed: Hold the explored-volume normaliser Z=1 (debugging).
        recursive_merge: Cascade merging until no neighbour qualifies.
        fixed_sigma: Disable Silverman bandwidth shrinkage.
        adaptive_sigma_stride: Welford warmup window in MD steps. Default
            10*pace.
        sigma_min: Per-CV floor on sigma (recommended with adaptive sigma).
        welford_stride: Sample CV every N MD steps for the adaptive Welford
            accumulator (default 1, faithful to PLUMED). Larger values batch
            ``simulation.step(N)`` and avoid per-step Python overhead;
            statistically equivalent once the warmup window contains enough
            independent samples.
    """

    def __init__(
            self,
            system: mm.System,
            variables: list[OpesVariable],
            temperature: unit.Quantity,
            barrier: unit.Quantity,
            pace: int,
            sigma: list[float | unit.Quantity] | None = None,
            mode: str = METAD,
            biasfactor: float | None = None,
            epsilon: float | None = None,
            kernel_cutoff: float | None = None,
            compression_threshold: float = 1.0,
            no_zed: bool = False,
            recursive_merge: bool = True,
            fixed_sigma: bool = False,
            adaptive_sigma_stride: int | None = None,
            sigma_min: list[float | unit.Quantity] | None = None,
            welford_stride: int = 1,
    ):
        if mode not in (METAD, EXPLORE):
            raise ValueError(f"mode must be '{METAD}' or '{EXPLORE}', got '{mode}'")
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(barrier):
            barrier = barrier * unit.kilojoules_per_mole
        if pace <= 0:
            raise ValueError('PACE must be positive')
        if not 1 <= len(variables) <= 3:
            raise ValueError('OPES requires 1, 2, or 3 collective variables')

        self.variables = variables
        self.mode = mode
        self.pace = pace
        self.temperature = temperature
        self.barrier = barrier
        self.recursive_merge = recursive_merge
        self.fixed_sigma = fixed_sigma
        self.no_zed = no_zed
        if welford_stride < 1:
            raise ValueError('welford_stride must be >= 1')
        if pace % welford_stride != 0:
            raise ValueError('PACE must be a multiple of welford_stride')
        self._welford_stride = welford_stride

        kbt_qty = unit.MOLAR_GAS_CONSTANT_R * temperature
        self._kbt = kbt_qty.value_in_unit(unit.kilojoules_per_mole)
        barrier_kj = barrier.value_in_unit(unit.kilojoules_per_mole)
        self._barrier = barrier_kj
        self._ncv = len(variables)

        self._biasfactor, self._bias_prefactor = _resolve_bias_factors(
            biasfactor, mode, barrier_kj, self._kbt,
        )
        self._epsilon, self._cutoff2, self._val_at_cutoff, self._threshold2 = (
            _resolve_epsilon_cutoff(
                epsilon, kernel_cutoff, compression_threshold,
                mode, barrier_kj, self._kbt, self._bias_prefactor,
            )
        )
        self._adaptive_sigma, self._sigma0, self._adaptive_sigma_stride = (
            _resolve_initial_sigma(sigma, self._ncv, adaptive_sigma_stride, pace)
        )

        if sigma_min is None:
            self._sigma_min = None
        else:
            if len(sigma_min) != self._ncv:
                raise ValueError('sigma_min must have one entry per CV')
            self._sigma_min = np.array([_standardize(s) for s in sigma_min], dtype=float)

        # Welford accumulators for adaptive sigma (running EWMA mean and M2).
        # _adaptive_counter counts CV samples and sets the EWMA weight 1/tau;
        # _adaptive_counter_md_steps counts MD steps and gates the warmup.
        self._adaptive_counter = 0
        self._adaptive_counter_md_steps = 0
        self._av_cv = np.zeros(self._ncv)
        self._av_m2 = np.zeros(self._ncv)

        # A phantom zero kernel keeps the first log argument finite (OPESmetad.cpp:380, 456).
        self._counter = 1
        self._sum_weights = self._epsilon ** self._bias_prefactor
        self._sum_weights2 = self._sum_weights * self._sum_weights
        if no_zed:
            self._sum_weights = 1.0
            self._sum_weights2 = 1.0
        self._zed = 1.0
        self._kde_norm = self._counter if mode == EXPLORE else self._sum_weights
        self._is_first_step = True

        self._kernels: list[_Kernel] = []

        # CV-space periodicity and grid extents.
        self._periodic = np.array([v.periodic for v in variables], dtype=bool)
        self._mins = np.array([v.min_value for v in variables], dtype=float)
        self._maxs = np.array([v.max_value for v in variables], dtype=float)
        self._spans = self._maxs - self._mins
        self._grid_widths = [v.grid_width for v in variables]

        n_periodic = self._periodic.sum()
        if n_periodic not in (0, self._ncv):
            raise ValueError('OPES cannot mix periodic and non-periodic CVs')
        self._all_periodic = n_periodic == self._ncv

        # Pre-build the grid for tabulated-function evaluation (recomputed each PACE).
        self._grid_axes = [
            np.linspace(self._mins[i], self._maxs[i], self._grid_widths[i])
            for i in range(self._ncv)
        ]

        # Build the CustomCVForce. The energy expression is just `table(cv0,cv1,...)`,
        # exactly as openmm.app.metadynamics does (see metadynamics.py:128-146).
        var_names = [f'cv{i}' for i in range(self._ncv)]
        self._force = mm.CustomCVForce(f"table({', '.join(var_names)})")
        for name, var in zip(var_names, variables):
            self._force.addCollectiveVariable(name, var.force)

        self._table = _build_tabulated_function(
            self._ncv, self._mins, self._maxs, self._grid_widths, self._all_periodic,
        )
        self._force.addTabulatedFunction('table', self._table)

        self._force.setForceGroup(_pick_free_force_group(system))
        system.addForce(self._force)

    def step(self, simulation, n_steps: int) -> None:
        """Advance the simulation, depositing kernels every PACE steps.

        OPES analogue of openmm.app.metadynamics.Metadynamics.step. Deposition
        fires when ``currentStep % pace == 0`` after the first MD step
        (the ``isFirstStep_`` guard in OPESmetad.cpp:993).
        """
        steps_to_go = n_steps
        force_group = self._force.getForceGroup()
        while steps_to_go > 0:
            current_in_cycle = simulation.currentStep % self.pace
            if current_in_cycle == 0 and not self._is_first_step:
                # We are exactly at a deposition step (after starting MD).
                # Fall through to deposit, then take min(pace, remaining).
                next_steps = min(steps_to_go, self.pace)
            else:
                # Run up to the next deposition boundary (or remaining).
                next_steps = min(steps_to_go, self.pace - current_in_cycle)

            self._step_chunk(simulation, next_steps, force_group)
            steps_to_go -= next_steps

    def _step_chunk(self, simulation, n_steps: int, force_group: int) -> None:
        """Run n_steps of MD; deposit a kernel if PACE-aligned at the end.

        Batches MD into runs of ``welford_stride`` steps between Welford
        samples. stride=1 is a per-step Python loop (faithful to PLUMED);
        larger strides skip Python overhead at a negligible statistical cost.
        """
        stride = self._welford_stride
        steps_remaining = n_steps
        while steps_remaining > 0:
            chunk = min(stride, steps_remaining)
            simulation.step(chunk)
            self._is_first_step = False
            steps_remaining -= chunk
            if self._adaptive_sigma:
                self._update_adaptive_sigma(simulation, count_steps=chunk)
            if simulation.currentStep % self.pace == 0:
                self._deposit_kernel(simulation, force_group)

    def get_free_energy(self) -> np.ndarray:
        """Return the current FES estimate (kJ/mol, min at zero).

        For METAD: F(s) = -V(s) / (1 - 1/gamma). For EXPLORE the bias is on
        the well-tempered distribution; F(s) = -gamma/(gamma-1) V(s). For an
        unbiased EXPLORE FES, reweight the trajectory by exp(beta V) instead.

        Returns:
            Array of shape (grid_width_1, ..., grid_width_N).
        """
        bias_grid = self._compute_bias_grid()
        if self.mode == METAD:
            # F(s) ~= -V(s) / (1 - 1/gamma)  (well-tempered FES from bias)
            fes = -bias_grid / self._bias_prefactor
        else:
            # EXPLORE bias is on the WT distribution; F(s) ~= -gamma/(gamma-1) V(s).
            fes = -bias_grid / self._bias_prefactor * self._biasfactor
        fes -= fes.min()
        return fes

    def get_collective_variables(self, simulation):
        """Read the current values of the CVs through the OPES CustomCVForce."""
        return self._force.getCollectiveVariableValues(simulation.context)

    def get_diagnostics(self) -> dict:
        """Return a dict of OPES diagnostic scalars (rct, neff, zed, nker)."""
        rct = self._kbt * math.log(self._sum_weights / self._counter)
        neff = (1.0 + self._sum_weights) ** 2 / (1.0 + self._sum_weights2)
        return {
            'rct_kj_per_mol': rct,
            'neff': neff,
            'zed': self._zed,
            'nker': len(self._kernels),
            'sum_weights': self._sum_weights,
            'counter': self._counter,
            'adaptive_counter': self._adaptive_counter,
            'sigma0': self._sigma0.copy() if self._sigma0.size else None,
        }

    def _update_adaptive_sigma(self, simulation, count_steps: int = 1) -> None:
        """Read the current CV and fold it into the Welford EWMA accumulator.

        EWMA mean update weights by 1/n_samples (stride-invariant); sigma
        uses n_md_steps so its dimensional answer matches PLUMED's stride=1.
        """
        if not self._adaptive_sigma:
            return
        self._adaptive_counter += 1
        self._adaptive_counter_md_steps += count_steps
        warmup_samples = max(1, self._adaptive_sigma_stride // self._welford_stride)
        tau = min(self._adaptive_counter, warmup_samples)

        cv_now = np.array(self._force.getCollectiveVariableValues(simulation.context))
        diff = cv_now - self._av_cv
        if self._all_periodic:
            diff = _wrap_periodic(diff, self._spans, self._periodic)
        self._av_cv = self._av_cv + diff / tau
        if self._all_periodic:
            # Wrap the running mean back into the canonical interval if a periodic
            # CV's mean drifted past the boundary.
            self._av_cv = self._mins + np.mod(self._av_cv - self._mins, self._spans)
        diff_after = cv_now - self._av_cv
        if self._all_periodic:
            diff_after = _wrap_periodic(diff_after, self._spans, self._periodic)
        self._av_m2 = self._av_m2 + diff * diff_after

    def _deposit_kernel(self, simulation, force_group: int) -> None:
        """Add a kernel at the current CV location and refresh the bias grid.

        Mirrors OPESmetad.cpp ``update()``: read current bias, compute weight
        (METAD: exp(beta V); EXPLORE: 1), update sum_weights / counter /
        KDEnorm, rescale sigma (Silverman + adaptive), rescale height
        (sigma0/sigma), merge, recompute Z, push the new grid.
        """
        # Skip deposition until enough MD steps have fed the Welford variance
        # (OPESmetad.cpp:1012). Counting MD steps keeps the window independent of welford_stride.
        if (self._adaptive_sigma
                and self._adaptive_counter_md_steps < self._adaptive_sigma_stride
                and self._counter == 1):
            return

        cv_now = np.array(self._force.getCollectiveVariableValues(simulation.context))

        bias_state = simulation.context.getState(getEnergy=True, groups={force_group})
        current_bias = bias_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        log_weight = current_bias / self._kbt
        height = math.exp(log_weight)

        self._counter += 1
        self._sum_weights += height
        self._sum_weights2 += height * height
        neff = (1.0 + self._sum_weights) ** 2 / (1.0 + self._sum_weights2)

        if self.mode == EXPLORE:
            self._kde_norm = self._counter
            height = 1.0  # plain KDE, OPESmetad.cpp:1049
        else:
            self._kde_norm = self._sum_weights

        sigma = self._sigma0.copy()
        if self._adaptive_sigma:
            factor = 1.0 if self.mode == EXPLORE else self._biasfactor
            # Divide by the sample count; equals PLUMED's per-step Welford at welford_stride=1.
            n_samples = self._adaptive_counter
            if self._counter == 2:  # first deposition only — OPESmetad.cpp:1058
                self._av_m2 = self._av_m2 * self._biasfactor
                self._sigma0 = np.sqrt(self._av_m2 / n_samples / factor)
                if self._sigma_min is not None:
                    self._sigma0 = np.maximum(self._sigma0, self._sigma_min)
                else:
                    if (self._sigma0 < 1e-6).any():
                        # Match the C++ warning by clamping to 1e-6.
                        self._sigma0 = np.maximum(self._sigma0, 1e-6)
            sigma = np.sqrt(self._av_m2 / n_samples / factor)
            if self._sigma_min is not None:
                sigma = np.maximum(sigma, self._sigma_min)
            else:
                sigma = np.maximum(sigma, 1e-6)

        if not self.fixed_sigma:
            size = self._counter if self.mode == EXPLORE else neff
            silverman = (size * (self._ncv + 2.0) / 4.0) ** (-1.0 / (4 + self._ncv))
            sigma = sigma * silverman
            if self._sigma_min is not None:
                sigma = np.maximum(sigma, self._sigma_min)

        # height *= sigma0 / sigma per CV — see OPESmetad.cpp:1107-1109.
        height *= float(np.prod(self._sigma0 / sigma))

        new_kernel = _Kernel(height=height, center=cv_now.copy(), sigma=sigma.copy())
        self._add_kernel_with_compression(new_kernel)

        if not self.no_zed:
            self._update_zed()

        self._refresh_bias_grid(simulation.context)

    def _add_kernel_with_compression(self, candidate: _Kernel) -> None:
        """Merge candidate into the closest in-threshold kernel, cascading
        recursively if the merged kernel has another near neighbour.

        Mirrors OPESmetad.cpp:1322-1371. Mahalanobis distance uses the
        existing kernel's sigma (line 1406); merged kernel inherits
        height-weighted moments.
        """
        if self._threshold2 == 0 or not self._kernels:
            self._kernels.append(candidate)
            return

        taker = self._find_mergeable(candidate.center, exclude_idx=-1)
        if taker is None:
            self._kernels.append(candidate)
            return

        self._merge_into(self._kernels[taker], candidate)
        if not self.recursive_merge:
            return

        # Cascade: the merged kernel may now overlap another existing kernel.
        # See OPESmetad.cpp:1331-1361.
        giver = taker
        next_taker = self._find_mergeable(self._kernels[giver].center, exclude_idx=giver)
        while next_taker is not None:
            if next_taker > giver:
                next_taker, giver = giver, next_taker
            self._merge_into(self._kernels[next_taker], self._kernels[giver])
            del self._kernels[giver]
            giver = next_taker
            next_taker = self._find_mergeable(self._kernels[giver].center, exclude_idx=giver)

    def _find_mergeable(self, center: np.ndarray, exclude_idx: int) -> int | None:
        """Index of the nearest in-threshold kernel, or None.

        Distance is Mahalanobis using the existing-kernel sigma
        (OPESmetad.cpp:1406).
        """
        best_norm2 = self._threshold2
        best_idx: int | None = None
        for k_idx, kernel in enumerate(self._kernels):
            if k_idx == exclude_idx:
                continue
            diff = center - kernel.center
            if self._all_periodic:
                diff = _wrap_periodic(diff, self._spans, self._periodic)
            norm2 = float(np.sum((diff / kernel.sigma) ** 2))
            if norm2 < best_norm2:
                best_norm2 = norm2
                best_idx = k_idx
        return best_idx

    def _merge_into(self, k1: _Kernel, k2: _Kernel) -> None:
        """Height-weighted moment-preserving merge of k2 into k1
        (OPESmetad.cpp:1693-1712). Periodic CVs are unwrapped before and
        rewrapped after the centroid computation.
        """
        h = k1.height + k2.height
        c = np.empty_like(k1.center)
        ss = np.empty_like(k1.sigma)
        for i in range(self._ncv):
            c1 = k1.center[i]
            c2 = k2.center[i]
            if self._periodic[i]:
                # Bring k1 next to k2 across the PBC for moment computation.
                diff = c1 - c2
                diff -= self._spans[i] * round(diff / self._spans[i])
                c1_unwrapped = c2 + diff
            else:
                c1_unwrapped = c1
            c_i = (k1.height * c1_unwrapped + k2.height * c2) / h
            ss_k1 = k1.height * (k1.sigma[i] ** 2 + c1_unwrapped ** 2)
            ss_k2 = k2.height * (k2.sigma[i] ** 2 + c2 ** 2)
            ss_i = (ss_k1 + ss_k2) / h - c_i * c_i
            if self._periodic[i]:
                # Bring centroid back into [min, max].
                c_i = self._mins[i] + ((c_i - self._mins[i]) % self._spans[i])
            c[i] = c_i
            ss[i] = max(ss_i, 1e-30)  # numerical floor; merged variance must be positive
        k1.center = c
        k1.sigma = np.sqrt(ss)
        k1.height = h

    def _update_zed(self) -> None:
        """Recompute the explored-volume normaliser Z from scratch.

        Z is the average unnormalised KDE evaluated at kernel centres
        (OPESmetad.cpp:1175-1240; Invernizzi thesis Eq. 2.72). Vectorised
        as O(n^2 * ncv); comfortable up to n ~ a few thousand.
        """
        n = len(self._kernels)
        if n == 0:
            self._zed = 1.0
            return
        # Stack kernel state into arrays of shape (n, ncv) and (n,).
        centers = np.stack([k.center for k in self._kernels], axis=0)
        sigmas = np.stack([k.sigma for k in self._kernels], axis=0)
        heights = np.array([k.height for k in self._kernels])

        # Pairwise differences shape (n, n, ncv): row index = "evaluated-at" kernel,
        # col index = "evaluating" kernel. Use the col kernel's sigma for distance,
        # mirroring evaluateKernel's convention (kernel-centred).
        diff = centers[:, None, :] - centers[None, :, :]
        if self._all_periodic:
            spans = self._spans
            for i in range(self._ncv):
                if self._periodic[i]:
                    diff[..., i] -= spans[i] * np.round(diff[..., i] / spans[i])
        # norm2[r, c] = ||centers[r] - centers[c]||^2 in units of sigmas[c]^2.
        scaled = diff / sigmas[None, :, :]
        norm2 = (scaled ** 2).sum(axis=-1)

        kernel_vals = np.where(
            norm2 < self._cutoff2,
            heights[None, :] * (np.exp(-0.5 * norm2) - self._val_at_cutoff),
            0.0,
        )
        sum_uprob = float(kernel_vals.sum())
        self._zed = sum_uprob / self._kde_norm / n

    def _evaluate_kernel(self, kernel: _Kernel, x: np.ndarray) -> float:
        """Evaluate a single kernel at x; return 0 outside the cutoff.

        The ``-val_at_cutoff`` shift (OPESmetad.cpp:1671) makes the
        hard-truncated kernel go to zero continuously at the boundary.
        """
        diff = x - kernel.center
        if self._all_periodic:
            diff = _wrap_periodic(diff, self._spans, self._periodic)
        norm2 = float(np.sum((diff / kernel.sigma) ** 2))
        if norm2 >= self._cutoff2:
            return 0.0
        return kernel.height * (math.exp(-0.5 * norm2) - self._val_at_cutoff)

    def _compute_bias_grid(self) -> np.ndarray:
        """Evaluate the OPES bias V(s) on the tabulation grid.

            V(s) = kbT * bias_prefactor * log(prob(s)/Zed + epsilon)
            prob(s) = (sum_k K(s, kernel_k)) / KDEnorm

        Each kernel's contribution is added on a local index-space patch
        (a few sigmas across, much smaller than the full grid). This is
        exact — the kernel is hard-truncated at the cutoff — but cost per
        kernel scales as O(patch_size) instead of O(full_grid).

        Boundaries on periodic axes are forced equal to satisfy OpenMM's
        createPeriodicSpline (wrapped Gaussians give equality up to FP noise).
        """
        if not self._kernels:
            return np.zeros(self._grid_widths)
        prob = np.zeros(self._grid_widths, dtype=float)
        cutoff = math.sqrt(self._cutoff2)
        widths = np.array(self._grid_widths)
        dx = (self._maxs - self._mins) / np.maximum(widths - 1, 1)

        for kernel in self._kernels:
            patch_half = (cutoff * kernel.sigma / dx).astype(int) + 1
            center_idx = (kernel.center - self._mins) / dx
            local_axes: list[np.ndarray] = []
            slices: list = []
            in_box = True
            for i in range(self._ncv):
                gw = self._grid_widths[i]
                base = int(math.floor(center_idx[i]))
                lo = base - patch_half[i]
                hi = base + patch_half[i] + 1
                if self._periodic[i]:
                    raw = np.arange(lo, hi)
                    wrapped = np.mod(raw, gw)
                    grid_pts = self._mins[i] + wrapped * dx[i]
                    d = grid_pts - kernel.center[i]
                    d = d - self._spans[i] * np.round(d / self._spans[i])
                    local_axes.append(d)
                    slices.append(wrapped)
                else:
                    lo_c = max(0, lo)
                    hi_c = min(gw, hi)
                    if hi_c <= lo_c:
                        in_box = False
                        break
                    grid_pts = self._mins[i] + np.arange(lo_c, hi_c) * dx[i]
                    local_axes.append(grid_pts - kernel.center[i])
                    slices.append(slice(lo_c, hi_c))
            if not in_box:
                continue

            norm2 = np.zeros(tuple(len(a) for a in local_axes))
            for i, d in enumerate(local_axes):
                shape = [1] * self._ncv
                shape[i] = len(d)
                norm2 = norm2 + ((d / kernel.sigma[i]) ** 2).reshape(shape)
            with np.errstate(over='ignore'):
                contrib = kernel.height * (np.exp(-0.5 * norm2) - self._val_at_cutoff)
            contrib = np.where(norm2 < self._cutoff2, contrib, 0.0)
            self._add_patch_to_grid(prob, contrib, slices)

        prob = prob / self._kde_norm
        bias = self._kbt * self._bias_prefactor * np.log(prob / self._zed + self._epsilon)
        # Force periodic-axis boundary equality so OpenMM's createPeriodicSpline accepts the table.
        for i in range(self._ncv):
            if self._periodic[i]:
                idx_first = [slice(None)] * self._ncv
                idx_last = [slice(None)] * self._ncv
                idx_first[i] = 0
                idx_last[i] = -1
                # Average the two boundary slabs so the bias is symmetric across the seam.
                avg = 0.5 * (bias[tuple(idx_first)] + bias[tuple(idx_last)])
                bias[tuple(idx_first)] = avg
                bias[tuple(idx_last)] = avg
        return bias

    @staticmethod
    def _add_patch_to_grid(
            grid: np.ndarray,
            patch: np.ndarray,
            slices: list,
    ) -> None:
        """Add a per-kernel patch into the global grid.

        ``slices`` mixes ``slice()`` (non-periodic axes) with ndarrays of
        wrapped indices (periodic axes). The np.add.at + np.ix_ path handles
        the wrapped case where a kernel's patch can repeat indices.
        """
        # Fast path: all slices are slice() — single in-place add.
        if all(isinstance(s, slice) for s in slices):
            grid[tuple(slices)] += patch
            return
        # General path: build mesh of indices for fancy indexing. For 1D and 2D
        # this is a few µs; the grid is small. We use np.ix_ to make the
        # fancy indexes broadcast correctly when mixed with slice objects.
        idx_arrays = []
        for s in slices:
            if isinstance(s, slice):
                idx_arrays.append(np.arange(s.start or 0, s.stop or 0))
            else:
                idx_arrays.append(np.asarray(s, dtype=int))
        # Use np.add.at with ix_ to handle wrapped (possibly repeated) indices.
        np.add.at(grid, np.ix_(*idx_arrays), patch)

    def _refresh_bias_grid(self, context) -> None:
        """Push the freshly computed bias grid to the CustomCVForce.

        OpenMM's tabulated functions expect a flattened array in x-fastest
        C order; transposing before ravel() matches that, same idiom as
        openmm.app.metadynamics.
        """
        bias_grid = self._compute_bias_grid()
        # Flatten to OpenMM's order (last CV varies slowest).
        flat = np.ascontiguousarray(np.transpose(bias_grid)).ravel()
        if self._ncv == 1:
            self._table.setFunctionParameters(
                flat.tolist(), float(self._mins[0]), float(self._maxs[0]),
            )
        elif self._ncv == 2:
            self._table.setFunctionParameters(
                self._grid_widths[0], self._grid_widths[1],
                flat.tolist(),
                float(self._mins[0]), float(self._maxs[0]),
                float(self._mins[1]), float(self._maxs[1]),
            )
        else:
            self._table.setFunctionParameters(
                self._grid_widths[0], self._grid_widths[1], self._grid_widths[2],
                flat.tolist(),
                float(self._mins[0]), float(self._maxs[0]),
                float(self._mins[1]), float(self._maxs[1]),
                float(self._mins[2]), float(self._maxs[2]),
            )
        self._force.updateParametersInContext(context)

    def get_state(self) -> dict:
        """Plain-dict snapshot of OPES state (kernels + scalars) for restart.

        Mirrors the STATE file fields in OPESmetad.cpp:1556-1658.
        """
        return {
            'mode': self.mode,
            'pace': self.pace,
            'kbt': self._kbt,
            'biasfactor': self._biasfactor,
            'bias_prefactor': self._bias_prefactor,
            'epsilon': self._epsilon,
            'cutoff2': self._cutoff2,
            'val_at_cutoff': self._val_at_cutoff,
            'threshold2': self._threshold2,
            'sigma0': self._sigma0.tolist(),
            'av_cv': self._av_cv.tolist(),
            'av_M2': self._av_m2.tolist(),
            'adaptive_counter': self._adaptive_counter,
            'counter': self._counter,
            'sum_weights': self._sum_weights,
            'sum_weights2': self._sum_weights2,
            'zed': self._zed,
            'kernels': [
                {
                    'height': k.height,
                    'center': k.center.tolist(),
                    'sigma': k.sigma.tolist(),
                }
                for k in self._kernels
            ],
        }

    def load_state(self, state: dict, context=None) -> None:
        """Restore an OPES state previously saved with ``get_state``.

        Pass ``context`` to refresh the bias grid in the simulation
        immediately.
        """
        self._sigma0 = np.array(state['sigma0'], dtype=float)
        self._av_cv = np.array(state['av_cv'], dtype=float)
        self._av_m2 = np.array(state['av_M2'], dtype=float)
        self._adaptive_counter = int(state['adaptive_counter'])
        self._counter = int(state['counter'])
        self._sum_weights = float(state['sum_weights'])
        self._sum_weights2 = float(state['sum_weights2'])
        self._zed = float(state['zed'])
        self._kernels = [
            _Kernel(
                height=float(rec['height']),
                center=np.array(rec['center'], dtype=float),
                sigma=np.array(rec['sigma'], dtype=float),
            )
            for rec in state['kernels']
        ]
        self._kde_norm = self._counter if self.mode == EXPLORE else self._sum_weights
        self._is_first_step = False
        if context is not None:
            self._refresh_bias_grid(context)
