from __future__ import annotations

import abc
import copy
import math
from typing import Any

import numpy as np
import openmm as mm
import openmm.unit as unit
from openmmtools import cache as omm_cache
from openmmtools.integrators import ThermostatedIntegrator
from openmmtools.states import ComposableStateError, SamplerState

from sandcat.variables import KB_KJ_PER_MOL_K

# Force group reserved for the OPES_EXPANDED bias wrapper.
OPES_BIAS_FORCE_GROUP = 1

# PLUMED geometric biases sit outside the ENERGY CV. Their forces are added
# after the ENERGY derivative rescales the engine force.
PLUMED_GEOMETRIC_BIAS_FORCE_GROUP = 2

# Sentinel global parameter used to identify the OPES wrapper CustomCVForce
# (a system may host other CustomCVForces, e.g. the SWISH-X contact-map wall).
OPES_WRAP_MARKER_PARAM = 'opes_initialized'


def _temp_grid(t_min: float, t_max: float, n_temps: int, geom_spacing: bool = True) -> np.ndarray:
    """Geometric (PLUMED default) or linear-in-β temperature grid."""
    if n_temps < 2:
        raise ValueError('n_temps must be >= 2')
    if geom_spacing:
        return np.geomspace(t_min, t_max, n_temps)
    betas = np.linspace(1.0 / t_min, 1.0 / t_max, n_temps)
    return 1.0 / betas


def _build_bias_formula(n_cvs: int, n_temps: int) -> str:
    """OpenMM expression for the exact OPES_EXPANDED LSE bias contribution to PE.

    Used by ``CustomCVForce`` evaluations only — the wrapper is in
    ``OPES_BIAS_FORCE_GROUP`` and excluded from per-step force evaluation
    by ``build_opes_expanded_integrator``. Per-step dynamics get the bias
    via the integrator's ``(1 + opes_alpha)`` force scaling, where
    ``opes_alpha`` is refreshed from Python's full-LSE evaluation every PACE.

    The on-device formula is the same LSE that PLUMED's
    ``OPESexpanded.cpp`` evaluates per-step — no affine approximation. It
    is float32-safe because (a) max-shifted ``exp(arg_j - max_arg)`` is
    bounded in [0, 1] regardless of arg magnitude, and (b) the
    ``df_shift`` trick (``df_shift_j = ΔF_j − kT·Δβ_j·u_ref``) keeps
    args near zero for diagonal evals where ``U ≈ u_ref``. The 2026-04-30
    instability blocking on-device LSE was about its *gradient* through
    ``max()/exp()``; energy-only evaluation (``getState(getEnergy=True)``)
    doesn't exercise that path.
    """
    if n_temps < 2:
        raise ValueError('n_temps must be >= 2')

    log_n = math.log(float(n_temps))
    arg_terms = [
        f'arg_{j} = -opes_dbeta_{j}*(U - opes_u_ref) + opes_df_shift_{j}/opes_kt'
        for j in range(n_temps)
    ]
    # Max reduction over args: nested binary max() — OpenMM's expression
    # parser supports max(a, b) only as a binary op, so fold right-to-left.
    max_expr = f'arg_{n_temps - 1}'
    for j in range(n_temps - 2, -1, -1):
        max_expr = f'max(arg_{j}, {max_expr})'
    sum_w_expr = ' + '.join(
        f'exp(arg_{j} - max_arg)' for j in range(n_temps)
    )
    parts = [
        # V_lse = -kT * (max_arg + log(sum exp(arg_j - max_arg) / N))
        #       = -kT * (max_arg + log(sum_w) - log(N))
        f'opes_initialized * (-opes_kt * (max_arg + log({sum_w_expr}) - {log_n}))',
        f'max_arg = {max_expr}',
    ] + arg_terms + ['U = ' + ' + '.join(f'e_{k}' for k in range(n_cvs))]
    return '; '.join(parts)


def add_opes_expanded_bias(system: mm.System, n_temps: int) -> mm.CustomCVForce:
    """Add the OPES_EXPANDED ENERGY-CV wrapper to ``system``.

    Wrapper holds deep copies of every atom-affecting Force as CVs (the
    originals stay in place for native per-step eval); it lives in
    ``OPES_BIAS_FORCE_GROUP`` and is excluded from per-step force evaluation
    by ``build_opes_expanded_integrator``. PLUMED-style geometric biases are
    excluded because native PLUMED evaluates ``ENERGY`` from the engine
    potential and applies geometric bias forces separately.
    """
    skip_types = (mm.MonteCarloBarostat, mm.CMMotionRemover)
    children: list[mm.Force] = []
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if (isinstance(force, skip_types)
                or force.getForceGroup() == PLUMED_GEOMETRIC_BIAS_FORCE_GROUP):
            continue
        # addCollectiveVariable requires Python ownership; XML round-trip
        # produces a fresh, owned copy without transferring from the System.
        children.append(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))
    if not children:
        raise ValueError('add_opes_expanded_bias: system has no atom-affecting forces')

    formula = _build_bias_formula(n_cvs=len(children), n_temps=n_temps)
    cv_force = mm.CustomCVForce(formula)
    cv_force.setForceGroup(OPES_BIAS_FORCE_GROUP)
    for cv_idx, child in enumerate(children):
        cv_force.addCollectiveVariable(f'e_{cv_idx}', child)
    cv_force.addGlobalParameter(OPES_WRAP_MARKER_PARAM, 0.0)
    cv_force.addGlobalParameter('opes_u_ref', 0.0)
    # opes_alpha is read by the integrator for per-step force scaling.
    cv_force.addGlobalParameter('opes_alpha', 0.0)
    cv_force.addGlobalParameter('opes_kt', KB_KJ_PER_MOL_K * 300.0)
    for j in range(n_temps):
        cv_force.addGlobalParameter(f'opes_dbeta_{j}', 0.0)
        cv_force.addGlobalParameter(f'opes_df_shift_{j}', 0.0)
    # Re-expose child globals on the wrapper so non-recursive scans
    # (e.g. SwishLambdaState's GlobalParameterState looking for ``lam``)
    # find them. Globals are context-level, so duplicates stay coherent.
    re_exposed: dict[str, float] = {}
    for child in children:
        try:
            n_globals = child.getNumGlobalParameters()
        except AttributeError:
            continue
        for parameter_id in range(n_globals):
            name = child.getGlobalParameterName(parameter_id)
            value = child.getGlobalParameterDefaultValue(parameter_id)
            if name in re_exposed or name.startswith('opes_'):
                continue
            re_exposed[name] = value
    for name, value in re_exposed.items():
        cv_force.addGlobalParameter(name, value)

    system.addForce(cv_force)
    return cv_force


def _build_alpha_step_expression(n_temps: int) -> str:
    """Lepton expression for ``opes_alpha_step = dV/dU`` evaluated at ``energy0``.

    Uses sub-var bindings so the whole softmax-weighted average compiles to
    one CustomIntegrator kernel launch. ``energy0`` is the physical energy in
    force group 0, excluding the OPES and geometric bias energies.
    Pre-init gating is by ``opes_initialized``; ``sum_w >= 1`` always (the
    max-shifted softmax has at least one term equal to ``exp(0) = 1``) so
    division is safe.
    """
    if n_temps < 2:
        raise ValueError('n_temps must be >= 2')
    max_expr = f'arg_{n_temps - 1}'
    for j in range(n_temps - 2, -1, -1):
        max_expr = f'max(arg_{j}, {max_expr})'
    sum_w = ' + '.join(f'exp(arg_{j} - max_arg)' for j in range(n_temps))
    sum_w_db = ' + '.join(
        f'exp(arg_{j} - max_arg)*opes_dbeta_{j}' for j in range(n_temps)
    )
    arg_defs = '; '.join(
        f'arg_{j} = -opes_dbeta_{j}*(energy0 - opes_u_ref) '
        f'+ opes_df_shift_{j}/opes_kt'
        for j in range(n_temps)
    )
    return (
        f'opes_initialized * opes_kt * sum_w_db / sum_w; '
        f'sum_w_db = {sum_w_db}; '
        f'sum_w = {sum_w}; '
        f'max_arg = {max_expr}; '
        f'{arg_defs}'
    )


def build_opes_expanded_integrator(
        timestep, collision_rate, temperature, n_temps: int,
):
    """LangevinMiddle clone with on-device force scaling by ``(1 + α(U))``.

    ``opes_alpha_step = dV/dU`` is recomputed every integrator step from the
    current FF energy via the exact log-sum-exp expression. One extra GPU kernel
    launch per step. Explicit group-0 forces receive OPES scaling and group-2
    wall forces are added unscaled. All groups remain visible to the barostat
    so volume acceptance includes the complete biased energy. VRORV splitting matches
    ``openmmtools.integrators.LangevinIntegrator``. Subclasses
    ``ThermostatedIntegrator`` so ``ThermodynamicState.apply_to_context``
    drives temperature via the standard kT setter.
    """
    if not unit.is_quantity(timestep):
        timestep = timestep * unit.femtosecond
    if not unit.is_quantity(collision_rate):
        collision_rate = collision_rate / unit.picosecond
    if not unit.is_quantity(temperature):
        temperature = temperature * unit.kelvin

    alpha_step_expr = _build_alpha_step_expression(n_temps)

    class _OpesExpandedIntegrator(ThermostatedIntegrator):
        def __init__(self, timestep, temperature, collision_rate):
            super().__init__(temperature, timestep)
            gamma_per_ps = collision_rate.value_in_unit(unit.picosecond ** -1)
            dt_ps = timestep.value_in_unit(unit.picosecond)
            self.addGlobalVariable('a', math.exp(-gamma_per_ps * dt_ps))
            self.addGlobalVariable(
                'b', math.sqrt(1.0 - math.exp(-2.0 * gamma_per_ps * dt_ps)),
            )
            self.addGlobalVariable('opes_alpha_step', 0.0)
            self.addPerDofVariable('xprev', 0)
            self.setKineticEnergyExpression('0.5 * m * v * v')
            # All groups, so MonteCarloBarostat volume moves see the OPES and
            # geometric bias energies; energy0/f0 keep force scaling on group 0.
            self.setIntegrationForceGroups(0xFFFFFFFF)
            self.addUpdateContextState()
            # Refresh α from current FF energy via the exact LSE (single
            # kernel; sub-var bindings inlined).
            self.addComputeGlobal('opes_alpha_step', alpha_step_expr)
            # Scale engine forces, then add geometric bias forces unscaled.
            self.addComputePerDof('v', 'v + dt*(1 + opes_alpha_step)*f0/m')
            self.addComputePerDof('v', 'v + dt*f2/m')
            self.addConstrainVelocities()
            # R (half drift) + constraint-position correction
            self.addComputePerDof('xprev', 'x')
            self.addComputePerDof('x', 'x + 0.5*dt*v')
            self.addConstrainPositions()
            self.addComputePerDof('v', '(x - xprev)/(0.5*dt)')
            self.addConstrainVelocities()
            # O (Langevin OU)
            self.addComputePerDof('v', 'a*v + b*sqrt(kT/m)*gaussian')
            self.addConstrainVelocities()
            # R (other half drift) + constraint-position correction
            self.addComputePerDof('xprev', 'x')
            self.addComputePerDof('x', 'x + 0.5*dt*v')
            self.addConstrainPositions()
            self.addComputePerDof('v', '(x - xprev)/(0.5*dt)')
            self.addConstrainVelocities()

    integrator = _OpesExpandedIntegrator(timestep, temperature, collision_rate)
    integrator._opes_n_temps = int(n_temps)
    return integrator


def _engine_groups_mask() -> int:
    """Return all engine force groups, excluding OPES and geometric biases."""
    return (0xFFFFFFFF
            & ~(1 << OPES_BIAS_FORCE_GROUP)
            & ~(1 << PLUMED_GEOMETRIC_BIAS_FORCE_GROUP))


def _geometric_bias_groups_mask() -> int:
    """Return the force-group mask for PLUMED-style geometric biases."""
    return 1 << PLUMED_GEOMETRIC_BIAS_FORCE_GROUP


def _ff_groups_mask() -> int:
    """Return the backward-compatible engine ENERGY mask."""
    return _engine_groups_mask()


def _read_u_ff_kj(context) -> float:
    """Read the engine ENERGY scalar supplied to OPES, in kJ/mol."""
    state = context.getState(getEnergy=True, groups=_engine_groups_mask())
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _read_geometric_bias_kj(context) -> float:
    """Read PLUMED-style geometric bias energy, in kJ/mol."""
    state = context.getState(getEnergy=True, groups=_geometric_bias_groups_mask())
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _read_contact_map(context) -> float:
    """Read the live contact-map CV from its geometric upper-wall force."""
    system = context.getSystem()
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if (isinstance(force, mm.CustomCVForce)
                and force.getForceGroup() == PLUMED_GEOMETRIC_BIAS_FORCE_GROUP
                and force.getNumCollectiveVariables() == 1
                and force.getCollectiveVariableName(0) == 'cmap'):
            return float(force.getCollectiveVariableValues(context)[0])
    raise RuntimeError('Contact-map upper-wall force is missing from the Context.')


def _find_opes_wrap_force(system: mm.System) -> tuple[int, mm.CustomCVForce]:
    """Locate the wrapper CustomCVForce by its sentinel global parameter."""
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        if not isinstance(force, mm.CustomCVForce):
            continue
        params = {force.getGlobalParameterName(i)
                  for i in range(force.getNumGlobalParameters())}
        if OPES_WRAP_MARKER_PARAM in params:
            return force_idx, force
    raise ValueError(
        'No OPES_EXPANDED wrapper found in system. Call '
        '``add_opes_expanded_bias(system, n_temps)`` before constructing the '
        'OpesExpanded composable state or running production.'
    )


class OpesExpanded:
    """Composable thermodynamic-state component for OPES_EXPANDED multithermal bias.

    Holds per-state ΔF grid, rct, counter, observation buffer. Plugs into
    ``CompoundThermodynamicState`` and pushes (alpha, V_offset, U_ref,
    init) onto the context's wrapper on apply. Also drives single-replica
    runs via ``step(simulation, n_steps)``.
    """

    def __init__(
            self,
            temperature,
            t_max,
            n_temps: int,
            t_min=None,
            pace: int = 500,
            observation_steps: int = 100,
            geom_spacing: bool = True,
    ):
        if not unit.is_quantity(temperature):
            temperature = temperature * unit.kelvin
        if not unit.is_quantity(t_max):
            t_max = t_max * unit.kelvin
        if t_min is None:
            t_min = temperature
        elif not unit.is_quantity(t_min):
            t_min = t_min * unit.kelvin

        if n_temps < 2:
            raise ValueError('n_temps must be >= 2')

        self._t0_k = float(temperature.value_in_unit(unit.kelvin))
        self._t_min_k = float(t_min.value_in_unit(unit.kelvin))
        self._t_max_k = float(t_max.value_in_unit(unit.kelvin))
        self._kt0_kj = KB_KJ_PER_MOL_K * self._t0_k
        self._beta0 = 1.0 / self._kt0_kj
        self._n_temps = int(n_temps)
        self.pace = int(pace)
        self.observation_steps = int(observation_steps)
        self._geom_spacing = bool(geom_spacing)

        if not (self._t_min_k <= self._t0_k <= self._t_max_k or
                abs(self._t0_k - self._t_min_k) < 1e-3 or
                abs(self._t0_k - self._t_max_k) < 1e-3):
            print(f'+++ WARNING: TEMP={self._t0_k:.2f} K outside chosen range '
                  f'[{self._t_min_k:.2f}, {self._t_max_k:.2f}] K')

        self._temps_k = _temp_grid(
            self._t_min_k, self._t_max_k, self._n_temps, self._geom_spacing,
        )
        self._dbeta = (1.0 / (KB_KJ_PER_MOL_K * self._temps_k)) - self._beta0

        self._deltaf_kj = np.zeros(self._n_temps, dtype=float)
        self._rct_kj = 0.0
        self._counter = 0
        self._obs_energies: list[float] = []
        self._initialized = False
        # Reference U for argument shifting (single-precision CUDA precision);
        # set to <U>_obs by init_from_observations.
        self._u_ref_kj = 0.0

    @property
    def n_temps(self) -> int:
        return self._n_temps

    @property
    def temps(self) -> np.ndarray:
        return self._temps_k.copy()

    @property
    def dbeta(self) -> np.ndarray:
        return self._dbeta.copy()

    @property
    def deltaf_kj(self) -> np.ndarray:
        return self._deltaf_kj.copy()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def kt_kj(self) -> float:
        return self._kt0_kj

    def compute_bias_v(self, u_kj: float) -> float:
        """Python-side LSE evaluation of V_bias(U); returns 0 before init."""
        if not self._initialized:
            return 0.0
        delta_u = u_kj - self._u_ref_kj
        df_shift = self._deltaf_kj - self._kt0_kj * self._dbeta * self._u_ref_kj
        args = -self._dbeta * delta_u + df_shift / self._kt0_kj
        max_arg = float(args.max())
        sum_w = float(np.exp(args - max_arg).sum())
        return -self._kt0_kj * (max_arg + math.log(sum_w / self._n_temps))

    def init_from_observations(self) -> None:
        """One-shot ΔF init from unbiased samples (OPESexpanded.cpp:772-835)."""
        obs = np.array(self._obs_energies, dtype=float)
        n_obs = len(obs)
        if n_obs < 1:
            raise RuntimeError('init_from_observations called with no observations')

        self._u_ref_kj = float(obs.mean())

        self._deltaf_kj = np.zeros(self._n_temps, dtype=float)
        self._deltaf_kj += self._kt0_kj * self._dbeta * obs[0]
        for t in range(1, n_obs):
            diff = -self._dbeta * obs[t] + self._deltaf_kj / self._kt0_kj - math.log(t)
            log1p_term = math.log1p(-1.0 / (1.0 + t))
            pos = diff > 0
            update = np.where(
                pos,
                diff + np.log1p(np.exp(-diff, where=pos, out=np.zeros_like(diff))),
                np.log1p(np.exp(np.where(pos, 0.0, diff))),
            )
            self._deltaf_kj -= self._kt0_kj * (update + log1p_term)

        self._rct_kj = 0.0
        # Match PLUMED (OPESexpanded.cpp:604): restart the counter at 1 so the
        # first post-observation PACE gets log(counter-1)=0, not log(n_obs-1).
        self._counter = 1
        self._initialized = True
        self._obs_energies.clear()

    def update_deltaf(self, bias_v_kj: float, u_kj: float) -> None:
        """Recursive ΔF update — OPESexpanded.cpp:892-915."""
        self._counter += 1
        kt = self._kt0_kj
        n = self._counter

        arg = (bias_v_kj - self._rct_kj) / kt - math.log(n - 1.0)
        if arg > 0:
            increment = kt * (arg + math.log1p(math.exp(-arg)))
        else:
            increment = kt * math.log1p(math.exp(arg))

        diff = (-self._dbeta * u_kj
                + (bias_v_kj - self._rct_kj + self._deltaf_kj) / kt
                - math.log(n - 1.0))
        pos = diff > 0
        diff_pos = np.where(pos, diff, 0.0)
        diff_neg = np.where(pos, 0.0, diff)
        delta_pos = increment - kt * (diff + np.log1p(np.exp(-diff_pos)))
        delta_neg = increment - kt * np.log1p(np.exp(diff_neg))
        self._deltaf_kj += np.where(pos, delta_pos, delta_neg)

        self._rct_kj += increment + kt * math.log1p(-1.0 / n)

    def apply_to_context(self, context):
        """Push the LSE-grid + integrator-α globals to the context.

        ``opes_dbeta_j`` and ``opes_df_shift_j`` parameterise the on-device
        log-sum-exp; ``opes_alpha`` is read by the integrator for the
        per-step ``(1 + α)`` force scaling. The grid globals are pushed
        every apply rather than once because ``df_shift_j = ΔF_j −
        kT·Δβ_j·u_ref`` depends on ``u_ref``, which is refreshed every
        PACE by ``_on_pace``.
        """
        if self._initialized:
            alpha = self._compute_alpha(self._u_ref_kj)
        else:
            alpha = 0.0
        context.setParameter('opes_u_ref', float(self._u_ref_kj))
        context.setParameter('opes_alpha', float(alpha))
        context.setParameter('opes_kt', float(self._kt0_kj))
        context.setParameter('opes_initialized', 1.0 if self._initialized else 0.0)
        df_shift = self._deltaf_kj - self._kt0_kj * self._dbeta * self._u_ref_kj
        for j in range(self._n_temps):
            context.setParameter(f'opes_dbeta_{j}', float(self._dbeta[j]))
            context.setParameter(f'opes_df_shift_{j}', float(df_shift[j]))

    def _compute_alpha(self, u_kj: float) -> float:
        """alpha = dV/dU = kT * <Δβ>_w in the standard OPES_EXPANDED form."""
        delta_u = u_kj - self._u_ref_kj
        df_shift = self._deltaf_kj - self._kt0_kj * self._dbeta * self._u_ref_kj
        args = -self._dbeta * delta_u + df_shift / self._kt0_kj
        max_arg = float(args.max())
        weights = np.exp(args - max_arg)
        sum_w = float(weights.sum())
        if sum_w == 0.0:
            return 0.0
        return float(self._kt0_kj * (self._dbeta * weights).sum() / sum_w)

    def apply_to_system(self, system):
        """Verify the system has been wrapped by ``add_opes_expanded_bias``."""
        self.check_system_consistency(system)

    def check_system_consistency(self, system):
        try:
            _, force = _find_opes_wrap_force(system)
        except ValueError as ex:
            raise ComposableStateError(str(ex))
        names = {force.getGlobalParameterName(i)
                 for i in range(force.getNumGlobalParameters())}
        required = {'opes_alpha', 'opes_kt', 'opes_u_ref', OPES_WRAP_MARKER_PARAM}
        required.update(f'opes_dbeta_{j}' for j in range(self._n_temps))
        required.update(f'opes_df_shift_{j}' for j in range(self._n_temps))
        missing = required - names
        if missing:
            raise ComposableStateError(
                f'System wrapper missing required global parameters: '
                f'{sorted(missing)}'
            )

    def _standardize_system(self, system):
        """Zero wrapper defaults so ContextCache hashes match across replicas."""
        try:
            _, force = _find_opes_wrap_force(system)
        except ValueError:
            return
        n_params = force.getNumGlobalParameters()
        for i in range(n_params):
            name = force.getGlobalParameterName(i)
            if (name in ('opes_alpha', 'opes_kt', 'opes_u_ref', OPES_WRAP_MARKER_PARAM)
                    or name.startswith('opes_dbeta_')
                    or name.startswith('opes_df_shift_')):
                force.setGlobalParameterDefaultValue(i, 0.0)

    def _on_setattr(self, standard_system, attribute_name, old_composable_state):
        # ΔF / init changes only touch context globals, not standard System XML.
        return False

    def _find_force_groups_to_update(self, context, current_context_state, memo):
        for component in getattr(current_context_state, '_composable_states', []):
            if isinstance(component, OpesExpanded):
                # On-device evaluation depends on (ΔF, u_ref, init) — when
                # all three match the cached state, the wrapper's energy is
                # unchanged and we can skip the bias-group recompute.
                if (np.array_equal(component._deltaf_kj, self._deltaf_kj)
                        and component._initialized == self._initialized
                        and component._u_ref_kj == self._u_ref_kj):
                    return set()
                break
        return {OPES_BIAS_FORCE_GROUP}

    def step(self, simulation, n_steps: int) -> None:
        """Drive ``simulation`` through ``n_steps``, updating ΔF every PACE.

        ``simulation.integrator`` must be from ``build_opes_expanded_integrator``.
        """
        self.apply_to_context(simulation.context)

        steps_to_go = int(n_steps)
        while steps_to_go > 0:
            cycle_pos = simulation.currentStep % self.pace
            chunk = (min(steps_to_go, self.pace - cycle_pos) if cycle_pos
                     else min(steps_to_go, self.pace))
            simulation.step(chunk)
            steps_to_go -= chunk
            if simulation.currentStep % self.pace == 0:
                self._on_pace(simulation.context)

    def _on_pace(self, context) -> None:
        """Per-PACE: read U from the FF group, update ΔF, push new (α, V_offset)."""
        u_kj = _read_u_ff_kj(context)
        if not self._initialized:
            self._obs_energies.append(u_kj)
            if len(self._obs_energies) >= self.observation_steps:
                self.init_from_observations()
                self.apply_to_context(context)
            return
        bias_v = self.compute_bias_v(u_kj)
        self.update_deltaf(bias_v, u_kj)
        self._u_ref_kj = u_kj
        self.apply_to_context(context)


    def get_diagnostics(self) -> dict:
        return {
            'n_temps': self._n_temps,
            'temps_k': self._temps_k.copy(),
            'dbeta_per_kj_per_mol': self._dbeta.copy(),
            'deltaF_kj': self._deltaf_kj.copy(),
            'rct_kj': self._rct_kj,
            'counter': self._counter,
            'initialized': self._initialized,
            'n_obs_collected': len(self._obs_energies),
        }

    def get_state(self) -> dict:
        return {
            't0_k': self._t0_k,
            't_min_k': self._t_min_k,
            't_max_k': self._t_max_k,
            'n_temps': self._n_temps,
            'temps_k': self._temps_k.tolist(),
            'dbeta': self._dbeta.tolist(),
            'deltaF_kj': self._deltaf_kj.tolist(),
            'u_ref_kj': self._u_ref_kj,
            'rct_kj': self._rct_kj,
            'counter': self._counter,
            'pace': self.pace,
            'observation_steps': self.observation_steps,
            'initialized': self._initialized,
            'obs_energies': list(self._obs_energies),
            'geom_spacing': self._geom_spacing,
        }

    def load_state(self, state: dict) -> None:
        self._temps_k = np.array(state['temps_k'], dtype=float)
        self._dbeta = np.array(state['dbeta'], dtype=float)
        self._deltaf_kj = np.array(state['deltaF_kj'], dtype=float)
        self._u_ref_kj = float(state.get('u_ref_kj', 0.0))
        self._rct_kj = float(state['rct_kj'])
        self._counter = int(state['counter'])
        self._initialized = bool(state['initialized'])
        self._obs_energies = list(state.get('obs_energies', []))
        self._n_temps = int(state.get('n_temps', len(self._temps_k)))

    # Pickle / openmmtools serialize hooks: required so ReplicaExchangeSampler
    # can round-trip OpesExpanded through the .nc reporter on restart.
    # Schema is the raw __dict__ so it stays compatible with .nc files written
    # under Python 3.11+ default __getstate__ (which returns self.__dict__).
    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def reweighted_log_weight(self, u_kj: float, T_target_k: float) -> float:
        """Per-frame log-weight `-(β_target - β_0)·U + β_0·V` for reweighting to T_target."""
        beta_target = 1.0 / (KB_KJ_PER_MOL_K * T_target_k)
        return -(beta_target - self._beta0) * u_kj


class OpesExpandedDynamicsMove:
    """openmmtools MCMCMove that drives force-scaling Langevin dynamics and
    updates OPES_EXPANDED ΔF every PACE. One move per replica.
    """

    def __init__(self, *, timestep: unit.Quantity | float,
                 collision_rate: unit.Quantity | float,
                 n_steps_per_iteration: int,
                 n_temps: int,
                 pace: int = 500, reassign_velocities: bool = False,
                 platform_name: str | None = None,
                 seed: int | None = None) -> None:
        if not unit.is_quantity(timestep):
            timestep = timestep * unit.femtosecond
        if not unit.is_quantity(collision_rate):
            collision_rate = collision_rate / unit.picosecond
        self._timestep = timestep
        self._collision_rate = collision_rate
        self.n_steps = int(n_steps_per_iteration)
        self.reassign_velocities = bool(reassign_velocities)
        self.pace = int(pace)
        self.n_temps = int(n_temps)
        self.platform_name = platform_name
        self.seed = None if seed is None else int(seed)
        # openmmtools' canonical cross-rank sync slot; bias state lives on
        # the thermo state (already broadcast), so just keep it empty.
        self.statistics: dict = {}
        self._cumulative_steps = 0
        self._gmx_step = 0
        self.colvar_stride_steps = None
        self._colvar_rows = []
        self._private_cache = None
    def _resolve_cache(self, context_cache: Any) -> omm_cache.ContextCache:
        if self.platform_name is not None:
            if self._private_cache is None:
                platform = mm.Platform.getPlatformByName(self.platform_name)
                self._private_cache = omm_cache.ContextCache(platform=platform)
            return self._private_cache
        if isinstance(context_cache, omm_cache.ContextCache):
            return context_cache
        return omm_cache.global_context_cache

    @property
    def timestep(self) -> unit.Quantity:
        """Return the move timestep for openmmtools timing and reporting."""
        return self._timestep

    def _find_opes_state(self, thermodynamic_state: Any) -> 'OpesExpanded':
        for component in getattr(thermodynamic_state, '_composable_states', []):
            if isinstance(component, OpesExpanded):
                return component
        raise RuntimeError(
            'OpesExpandedDynamicsMove requires an OpesExpanded composable '
            'state on the thermodynamic state.'
        )

    def _build_integrator(self, thermodynamic_state: Any) -> mm.CustomIntegrator:
        return build_opes_expanded_integrator(
            timestep=self._timestep,
            collision_rate=self._collision_rate,
            temperature=thermodynamic_state.temperature,
            n_temps=self.n_temps,
        )

    def apply(
            self,
            thermodynamic_state: Any,
            sampler_state: Any,
            context_cache: Any = None,
    ) -> None:
        """Propagate one replica forward by n_steps, updating ΔF every PACE."""
        opes = self._find_opes_state(thermodynamic_state)
        cache = self._resolve_cache(context_cache)
        integrator_template = self._build_integrator(thermodynamic_state)
        if self.seed is not None:
            integrator_template.setRandomNumberSeed(self.seed)
        context, integrator = cache.get_context(thermodynamic_state, integrator_template)

        # get_context already invokes thermodynamic_state.apply_to_context,
        # which routes through OpesExpanded.apply_to_context.
        sampler_state.apply_to_context(
            context, ignore_velocities=self.reassign_velocities,
        )
        if self.reassign_velocities:
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)

        steps_to_go = self.n_steps
        while steps_to_go > 0:
            current_step = self._cumulative_steps
            target_steps = [
                current_step + steps_to_go,
                ((current_step // self.pace) + 1) * self.pace,
            ]
            if self.colvar_stride_steps is not None:
                target_steps.append(
                    ((current_step // self.colvar_stride_steps) + 1)
                    * self.colvar_stride_steps,
                )
            chunk = min(target_steps) - current_step
            integrator.step(chunk)
            self._cumulative_steps += chunk
            steps_to_go -= chunk
            if self._cumulative_steps % self.pace == 0:
                self._on_pace(context, opes)
            if (self.colvar_stride_steps is not None
                    and self._cumulative_steps % self.colvar_stride_steps == 0):
                self._record_colvar(context, opes)

        sampler_state.update_from_context(context)

    def _update_opes_python(self, opes: 'OpesExpanded', u_kj: float) -> None:
        """Advance the estimator without changing the live Context."""
        if not opes._initialized:
            opes._obs_energies.append(u_kj)
            if len(opes._obs_energies) >= opes.observation_steps:
                opes.init_from_observations()
            return
        bias_v = opes.compute_bias_v(u_kj)
        opes.update_deltaf(bias_v, u_kj)
        opes._u_ref_kj = u_kj

    def _integrate_to_step(
            self,
            integrator: mm.CustomIntegrator,
            context: mm.Context,
            opes: 'OpesExpanded',
            target_step: int,
    ) -> None:
        """Integrate until ``target_step`` is ready, staggering every PACE."""
        if target_step < self._gmx_step:
            raise RuntimeError(
                f'Cannot rewind GMX lifecycle from step {self._gmx_step} '
                f'to {target_step}.',
            )
        while self._gmx_step < target_step:
            if self._gmx_step > 0 and self._gmx_step % self.pace == 0:
                u_kj = _read_u_ff_kj(context)
                self._update_opes_python(opes, u_kj)
                integrator.step(1)
                self._gmx_step += 1
                self._cumulative_steps = self._gmx_step
                colvar_stride_steps = getattr(self, 'colvar_stride_steps', None)
                if (colvar_stride_steps is not None
                        and self._gmx_step % colvar_stride_steps == 0):
                    self._record_colvar(context, opes)
                opes.apply_to_context(context)
                continue

            next_pace = ((self._gmx_step // self.pace) + 1) * self.pace
            next_colvar = target_step
            colvar_stride_steps = getattr(self, 'colvar_stride_steps', None)
            if colvar_stride_steps is not None:
                next_colvar = (
                    (self._gmx_step // colvar_stride_steps) + 1
                ) * colvar_stride_steps
            chunk = min(target_step, next_pace, next_colvar) - self._gmx_step
            integrator.step(chunk)
            self._gmx_step += chunk
            self._cumulative_steps = self._gmx_step
            if (colvar_stride_steps is not None
                    and self._gmx_step % colvar_stride_steps == 0):
                self._record_colvar(context, opes)

    def _record_colvar(self, context: mm.Context, opes: 'OpesExpanded') -> None:
        """Capture one hard-trace row from the integrated Context."""
        u_kj = _read_u_ff_kj(context)
        self._colvar_rows.append((
            self._gmx_step * self._timestep.value_in_unit(unit.picosecond),
            _read_contact_map(context),
            _read_geometric_bias_kj(context),
            u_kj,
            opes.compute_bias_v(u_kj),
        ))

    def apply_gmx_staggered(
            self, thermodynamic_state: Any, sampler_state: Any, *,
            boundary_step: int, context_cache: Any = None,
    ) -> tuple[object, dict[str, float]]:
        """Propagate through one native GMX/PLUMED exchange boundary."""
        opes = self._find_opes_state(thermodynamic_state)
        cache = self._resolve_cache(context_cache)
        integrator_template = self._build_integrator(thermodynamic_state)
        if self.seed is not None:
            integrator_template.setRandomNumberSeed(self.seed)
        context, integrator = cache.get_context(thermodynamic_state, integrator_template)

        sampler_state.apply_to_context(
            context, ignore_velocities=self.reassign_velocities,
        )
        if self.reassign_velocities:
            context.setVelocitiesToTemperature(thermodynamic_state.temperature)
        self._integrate_to_step(integrator, context, opes, int(boundary_step))

        resident_u_kj = _read_u_ff_kj(context)
        resident_geometric_kj = _read_geometric_bias_kj(context)
        resident_pre_bias_kj = opes.compute_bias_v(resident_u_kj)
        # Only positions and box vectors cross the boundary. Constructing a
        # minimal state avoids copying unused velocities and cached energies.
        if hasattr(sampler_state, 'positions'):
            boundary_state = SamplerState(
                positions=sampler_state.positions,
                box_vectors=sampler_state.box_vectors,
            )
        else:
            # Keep lightweight test doubles and third-party state objects
            # compatible with the original generic path.
            boundary_state = copy.deepcopy(sampler_state)
        boundary_state.update_from_context(context, ignore_velocities=True)

        if self._gmx_step > 0 and self._gmx_step % self.pace == 0:
            self._update_opes_python(opes, resident_u_kj)

        integrator.step(1)
        self._gmx_step += 1
        self._cumulative_steps = self._gmx_step
        opes.apply_to_context(context)
        sampler_state.update_from_context(context)

        return boundary_state, {
            'resident_u_kj': float(resident_u_kj),
            'resident_geometric_kj': float(resident_geometric_kj),
            'resident_pre_bias_kj': float(resident_pre_bias_kj),
            'boundary_step': int(boundary_step),
            'post_step': int(self._gmx_step),
            'colvar_rows': self.pop_colvar_rows(),
        }

    def set_colvar_stride(self, stride_steps: int | None) -> None:
        """Enable exact-context COLVAR sampling at an integration-step stride."""
        if stride_steps is not None and int(stride_steps) <= 0:
            raise ValueError('COLVAR stride must be a positive number of steps.')
        self.colvar_stride_steps = None if stride_steps is None else int(stride_steps)

    def pop_colvar_rows(self) -> list[tuple[float, float, float, float, float]]:
        """Return and clear buffered COLVAR rows."""
        rows = getattr(self, '_colvar_rows', [])
        self._colvar_rows = []
        return rows

    def _on_pace(self, context: mm.Context, opes: 'OpesExpanded') -> None:
        u_kj = _read_u_ff_kj(context)
        self._update_opes_python(opes, u_kj)
        opes.apply_to_context(context)

    # Pickle support for MPI rank distribution.
    def __getstate__(self) -> dict[str, Any]:
        return {
            'timestep': self._timestep,
            'collision_rate': self._collision_rate,
            'n_steps': self.n_steps,
            'reassign_velocities': self.reassign_velocities,
            'pace': self.pace,
            'n_temps': self.n_temps,
            'platform_name': self.platform_name,
            'seed': self.seed,
            'statistics': self.statistics,
            'cumulative_steps': self._cumulative_steps,
            'gmx_step': self._gmx_step,
            'colvar_stride_steps': self.colvar_stride_steps,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._timestep = state['timestep']
        self._collision_rate = state['collision_rate']
        self.n_steps = state['n_steps']
        self.reassign_velocities = state.get('reassign_velocities', False)
        self.pace = state['pace']
        self.n_temps = int(state['n_temps'])
        self.platform_name = state.get('platform_name')
        self.seed = state.get('seed')
        self.colvar_stride_steps = state.get('colvar_stride_steps')
        self._colvar_rows = []
        self.statistics = state.get('statistics', {})
        self._cumulative_steps = state.get('cumulative_steps', 0)
        self._gmx_step = state.get('gmx_step', self._cumulative_steps)
        self._private_cache = None
