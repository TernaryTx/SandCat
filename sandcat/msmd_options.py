from __future__ import annotations

from collections.abc import Sequence


def normalise_msmd_probe_options(
        probe_types: str | Sequence[str],
        probe_ff_paths: str | Sequence[str] | None,
        conc_mol_per_litre: float | None,
) -> tuple[list[str], list[str | None], float | None]:
    """Validate and normalise single- and multi-ligand MSMD options.

    A single ligand retains SandCat's concentration-based mixed-solvent setup,
    including the historical 1 M default. Multi-ligand MSMD instead inserts
    exactly one copy of every ligand and therefore cannot use a concentration.

    Args:
        probe_types: One probe identifier or a sequence of ligand SDF paths.
        probe_ff_paths: Optional force-field path or one path per ligand.
        conc_mol_per_litre: Requested concentration, or None when omitted.

    Returns:
        Normalised probe paths, paired force-field paths, and resolved concentration.

    Raises:
        ValueError: If multi-ligand inputs are incomplete or incompatible.
    """
    normalised_probe_types = (
        [probe_types] if isinstance(probe_types, str) else list(probe_types)
    )
    if not normalised_probe_types:
        raise ValueError('At least one probe must be supplied.')

    if probe_ff_paths is None:
        normalised_probe_ff_paths: list[str | None] = []
    elif isinstance(probe_ff_paths, str):
        normalised_probe_ff_paths = [probe_ff_paths]
    else:
        normalised_probe_ff_paths = list(probe_ff_paths)

    if len(normalised_probe_types) == 1:
        if len(normalised_probe_ff_paths) > 1:
            raise ValueError('A single probe accepts at most one probe force field.')
        paired_probe_ff_paths = normalised_probe_ff_paths or [None]
        resolved_concentration = (
            1.0 if conc_mol_per_litre is None else conc_mol_per_litre
        )
        return normalised_probe_types, paired_probe_ff_paths, resolved_concentration

    if conc_mol_per_litre is not None:
        raise ValueError(
            '--concentration cannot be used with multiple probes; multi-ligand '
            'MSMD adds exactly one copy of each ligand.',
        )
    if len(normalised_probe_ff_paths) != len(normalised_probe_types):
        raise ValueError(
            'Multi-ligand MSMD requires one --probe_ff_path per --probe, in the '
            'same order.',
        )
    if any(probe_ff_path is None for probe_ff_path in normalised_probe_ff_paths):
        raise ValueError('Every ligand in multi-ligand MSMD requires an OFFXML path.')
    return normalised_probe_types, normalised_probe_ff_paths, None
