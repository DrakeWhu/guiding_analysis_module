from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .openpmd_io import get_iterations, open_series


EXIT_TARGET_KEYS = {
    "plateau": "plateau_exit",
    "capillary": "capillary_exit",
}


def read_resolved_particle_exit_target(
    resolved_parameters: str | Path,
    *,
    exit_kind: str,
) -> dict[str, Any]:
    """Read the exact particle-diagnostic exit target persisted by the input.

    For CLPU production cases, ``particle_diagnostic_targets`` is the
    authoritative contract for particle snapshots.  In particular, the
    recorded iteration must not be replaced by a nearby regular field frame.
    """

    path = Path(resolved_parameters)
    if not path.is_file():
        raise FileNotFoundError(f"Missing resolved simulation parameters: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read resolved simulation parameters: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Resolved simulation parameters must be a JSON object: {path}")

    try:
        target_key = EXIT_TARGET_KEYS[exit_kind]
    except KeyError as exc:
        raise ValueError(f"Unknown exit_kind: {exit_kind}") from exc

    targets = payload.get("particle_diagnostic_targets")
    if not isinstance(targets, dict):
        raise ValueError(
            f"Resolved simulation parameters lack particle_diagnostic_targets in {path}"
        )

    target = targets.get(target_key)
    if not isinstance(target, dict):
        raise ValueError(
            f"Resolved simulation parameters lack particle target {target_key!r} in {path}"
        )

    if "iteration" not in target:
        raise ValueError(f"Particle target {target_key!r} lacks iteration in {path}")

    raw_iteration = target["iteration"]
    if isinstance(raw_iteration, bool):
        raise ValueError(f"Particle target {target_key!r} iteration must be an integer")
    try:
        iteration = int(raw_iteration)
        numeric_iteration = float(raw_iteration)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Particle target {target_key!r} iteration is not numeric in {path}"
        ) from exc
    if not math.isfinite(numeric_iteration) or numeric_iteration != iteration or iteration < 0:
        raise ValueError(
            f"Particle target {target_key!r} iteration must be a non-negative integer"
        )

    result: dict[str, Any] = {
        "selection_mode": "exit",
        "particle_exit_selection_policy": "exact_resolved_v1",
        "resolved_parameters_path": str(path.resolve()),
        "resolved_particle_target_key": target_key,
        "target_particle_iteration": iteration,
    }

    for key in ("target_distance_m", "dump_distance_m", "distance_error_m"):
        if key not in target:
            continue
        try:
            value = float(target[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Particle target {target_key!r} field {key!r} is not numeric in {path}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"Particle target {target_key!r} field {key!r} is not finite in {path}"
            )
        result[key] = value

    if "target_distance_m" in result:
        result["target_propagation_mm"] = float(result["target_distance_m"]) * 1.0e3

    return result


def require_exact_particle_iteration(
    diag: str | Path,
    *,
    target_iteration: int,
) -> dict[str, Any]:
    """Require the exact target iteration to exist in the particle diagnostic.

    This intentionally has no nearest-snapshot or last-snapshot fallback.
    A missing target is an invalid exit measurement, even if another dump is
    only one timestep away.
    """

    ts = open_series(diag)
    available = sorted({int(value) for value in get_iterations(ts, stride=1)})
    if not available:
        raise RuntimeError(f"No particle iterations found in {diag}")

    target = int(target_iteration)
    if target not in available:
        raise RuntimeError(
            "Exact particle exit dump is missing: "
            f"target iteration={target}, available min={available[0]}, "
            f"available max={available[-1]}, count={len(available)}. "
            "Refusing nearest/last fallback."
        )

    return {
        "selected_particle_iteration": target,
        "target_iteration_delta": 0,
        "available_particle_iterations_min": available[0],
        "available_particle_iterations_max": available[-1],
        "n_available_particle_iterations": len(available),
    }
