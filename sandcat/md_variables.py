from __future__ import annotations

from openmm.unit import bar
from openmm.unit import femtosecond
from openmm.unit import kelvin
from openmm.unit import kilojoule
from openmm.unit import kilojoules_per_mole
from openmm.unit import mole
from openmm.unit import nanometer
from openmm.unit import picosecond
from openmm.unit.quantity import Quantity


def get_n_steps(
    time: Quantity,
    time_step: Quantity,
) -> int:
    """Returns a number of steps for a given period of time, using a defined time step.
    Args:
        time (Quantity) : time as an OpenMM Quantity
        time_step (Quantity) : time_step as an OpenMM Quantity, usually 2 or 4 femtoseconds
    Returns
        Rounded up integer of steps.
    """
    return round(time / time_step)


K_B = 0.008314462618 * kilojoule / (mole * kelvin)  # Boltzmann constant
TEMPERATURE = 300 * kelvin
BIAS_FACTOR = 4.0
FRICTION = 1.0 / picosecond
TIME_STEP_4FS = 4 * femtosecond
REPORT_INTERVAL = 100 * picosecond
PULL_INTERVAL = 2 * picosecond
SPRING_CONSTANT = 1000.0 * kilojoules_per_mole / nanometer ** 2
PRESSURE = 1 * bar
DCD_TRJ_FNAME = 'trj.dcd'  # hard-coded DCD format
SIM_LOG_FNAME = 'sim_log.csv'  # hard-coded CSV format
STATE_XML_FNAME = 'state.xml'
SYSTEM_XML_FNAME = 'system.xml'
