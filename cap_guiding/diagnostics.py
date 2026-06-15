from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


FIELD_DIAG_CANDIDATES: tuple[str, ...] = ("fields", "diag1")
ELECTRON_DENSITY_FIELD_CANDIDATES: tuple[str, ...] = (
    "rho_electrons",
    "rho_plasma_electrons",
    "rho_ionized_electrons",
)

_PARTICLE_DIAG_CANDIDATES_BY_SPECIES: dict[str, tuple[str, ...]] = {
    # Legacy campaigns sometimes used CASE/diags/electron_particles, and some
    # analysis commands point one level deeper to electron_particles/openpmd.
    "electrons": (
        "electron_particles/openpmd",
        "electron_particles",
        "electrons/openpmd",
        "electrons",
    ),
    # New ionization campaigns write one particle diagnostic per electron
    # population. Keep the direct directory first, matching the validated
    # SUNRISE ionization test layout.
    "plasma_electrons": (
        "plasma_electrons",
        "plasma_electrons/openpmd",
    ),
    "ionized_electrons": (
        "ionized_electrons",
        "ionized_electrons/openpmd",
    ),
}


def _format_candidates(paths: Sequence[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def resolve_field_diag_dir(
    case_dir: str | Path,
    *,
    require_exists: bool = True,
) -> Path:
    """Return the field diagnostic directory for a case.

    Preference order is intentionally explicit:
      1. CASE/diags/fields   new ionization-capable campaigns
      2. CASE/diags/diag1    legacy campaigns

    If require_exists=False, the preferred future path CASE/diags/fields is
    returned when neither directory exists. This is useful for campaign
    discovery/dry-runs over cases that have not produced diagnostics yet.
    """
    case_dir = Path(case_dir)
    diags_dir = case_dir / "diags"
    candidates = [diags_dir / name for name in FIELD_DIAG_CANDIDATES]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    if not require_exists:
        return candidates[0]

    raise FileNotFoundError(
        "No supported WarpX field diagnostic directory found for case "
        f"{case_dir}. Expected one of: {_format_candidates(candidates)}"
    )


def _normalize_available_fields(available_fields: Iterable[object]) -> dict[str, str]:
    normalized: dict[str, str] = {}

    for item in available_fields:
        if isinstance(item, str):
            name = item
        elif isinstance(item, (tuple, list)) and item:
            name = str(item[0])
        else:
            name = str(item)

        normalized.setdefault(name.lower(), name)

    return normalized


def resolve_electron_density_field(
    available_fields: Iterable[object],
    *,
    preferred: str | None = None,
) -> str:
    """Resolve the electron-density mesh name from openPMD available fields.

    This does not add or combine densities. In ionization campaigns the caller
    must explicitly request rho_ionized_electrons if that is the desired mesh;
    otherwise rho_plasma_electrons is the default new-campaign electron density.
    """
    normalized = _normalize_available_fields(available_fields)

    if preferred is not None:
        key = preferred.lower()
        if key in normalized:
            return normalized[key]
        raise ValueError(
            f"Requested electron density field {preferred!r} is not available. "
            f"Available fields: {sorted(normalized.values())}"
        )

    for candidate in ELECTRON_DENSITY_FIELD_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    raise ValueError(
        "No supported electron density field found. Expected one of "
        f"{list(ELECTRON_DENSITY_FIELD_CANDIDATES)}; "
        f"available fields: {sorted(normalized.values())}"
    )


def resolve_particle_diag_dir(
    case_dir: str | Path,
    species_name: str,
    *,
    particle_diag_name: str | None = None,
) -> Path:
    """Return the particle diagnostic directory for a species.

    particle_diag_name is an explicit override relative to CASE/diags, and can
    include subdirectories such as electron_particles/openpmd. Without an
    override, the species name selects the legacy/new compatible candidates.
    """
    case_dir = Path(case_dir)
    species_name = str(species_name)
    diags_dir = case_dir / "diags"

    if particle_diag_name not in (None, "", "auto"):
        candidate = diags_dir / str(particle_diag_name)
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"Requested particle diagnostic directory does not exist: {candidate}"
        )

    candidate_names = _PARTICLE_DIAG_CANDIDATES_BY_SPECIES.get(
        species_name,
        (species_name, f"{species_name}/openpmd"),
    )
    candidates = [diags_dir / name for name in candidate_names]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"No supported particle diagnostic directory found for species "
        f"{species_name!r} in case {case_dir}. Expected one of: "
        f"{_format_candidates(candidates)}"
    )
