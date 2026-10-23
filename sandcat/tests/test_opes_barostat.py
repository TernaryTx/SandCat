from __future__ import annotations

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from sandcat.opes_expanded import (
    OpesExpanded,
    add_opes_expanded_bias,
    build_opes_expanded_integrator,
)


def test_explicit_engine_kick_matches_complete_biased_force() -> None:
    system = mm.System()
    system.addParticle(12)
    physical = mm.CustomExternalForce('100*(x*x+y*y+z*z)')
    physical.addParticle(0, [])
    system.addForce(physical)
    wall = mm.CustomExternalForce('25*(x-.2)^2')
    wall.addParticle(0, [])
    wall.setForceGroup(2)
    system.addForce(wall)
    add_opes_expanded_bias(system, 2)
    opes = OpesExpanded(
        temperature=300 * unit.kelvin, t_min=300 * unit.kelvin,
        t_max=600 * unit.kelvin, n_temps=2,
    )
    opes._initialized = True
    opes._u_ref_kj = 0.0
    opes._deltaf_kj = np.zeros(2)
    trajectories = []
    for explicit in (False, True):
        if explicit:
            integrator = build_opes_expanded_integrator(0.001, 0.0, 300.0, 2)
        else:
            integrator = mm.CustomIntegrator(0.000001)
            integrator.addComputePerDof('v', 'v+dt*f/m')
            integrator.addComputePerDof('x', 'x+dt*v')
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
        context.setPositions([[0.1, 0.2, 0.3]])
        context.setVelocities([[0.03, 0.04, 0.05]])
        opes.apply_to_context(context)
        integrator.step(20)
        state = context.getState(getPositions=True, getVelocities=True)
        trajectories.append((
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
        ))
        del context, integrator
    np.testing.assert_allclose(trajectories[0], trajectories[1], atol=1e-9, rtol=0)


def _volume_trace(group: int) -> list[float]:
    """Measure seeded volume moves with an energy assigned to one force group."""
    system = mm.System()
    system.addParticle(12)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.5)
    nonbonded.addParticle(0, 1, 0)
    system.addForce(nonbonded)
    add_opes_expanded_bias(system, 2)
    # Add after constructing the inactive wrapper to keep identical OPES CVs.
    bias = mm.CustomExternalForce('1000000*x*x')
    bias.addParticle(0, [])
    bias.setForceGroup(group)
    system.addForce(bias)
    barostat = mm.MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin, 1)
    barostat.setRandomNumberSeed(2718)
    system.addForce(barostat)
    # Negligible MD displacement isolates the volume acceptance calculation.
    integrator = build_opes_expanded_integrator(1e-12, 0.0, 300.0, 2)
    integrator.setRandomNumberSeed(3141)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    context.setPositions([[1, 0, 0]])
    context.setVelocities([[0, 0, 0]])
    volumes = []
    for _ in range(40):
        integrator.step(1)
        box = context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        volumes.append(float(np.linalg.det(box)))
    del context, integrator
    return volumes


@pytest.mark.parametrize('group', [1, 2])
def test_barostat_sees_bias_energy_in_each_reserved_group(group: int) -> None:
    # Contexts run serially and die before the next initializes the host RNG.
    np.testing.assert_allclose(_volume_trace(group), _volume_trace(0), rtol=0, atol=1e-10)
