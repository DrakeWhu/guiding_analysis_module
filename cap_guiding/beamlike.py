from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class BeamlikeConfig:
    """Configuration for reduced beamlike scoring from particle summaries.

    The score is intentionally derived only from already-reduced particle metrics.
    Raw physical metrics remain separate from normalized score components.

    Hard eligibility uses statistics, hot charge, and robust hot-electron energy.
    The mono proxy is kept as a score component / quality flag, not as a hard
    eligibility cut, to preserve the behavior of the SUNRISE prototype script.
    """

    min_hot_macroparticles: float = 200.0
    min_hot_charge_pC: float = 100.0
    min_hot_E95_MeV: float = 50.0
    min_mono_proxy: float = 0.30

    charge_ref_pC: float = 1200.0
    n_hot_ref: float = 1000.0
    energy_ref_MeV: float = 220.0
    mono_ref: float = 0.65
    z_span_ref_mm: float = 0.50

    score_scale: float = 1000.0

    charge_exponent: float = 0.90
    statistics_exponent: float = 1.00
    energy_exponent: float = 1.20
    mono_exponent: float = 0.75

    z_compact_weight: float = 0.15
    divergence_weight: float = 0.20


BEAMLIKE_OUTPUT_COLUMNS = [
    "eligible_beamlike",
    "beamlike_status",
    "beamlike_score",
    "beam_yield_score",
    "beamlike_rejection_reasons",
    "beamlike_tags",
    "mono_proxy_E95_over_Emax",
    "z_span_hot_mm",
    "statistics_component",
    "charge_component",
    "energy_component",
    "mono_component",
    "z_compact_component",
    "divergence_component",
    "divergence_source",
    "theta_x_rms_mrad",
    "theta_y_rms_mrad",
    "theta_rms_mrad",
    "theta_x_p95_mrad",
    "theta_y_p95_mrad",
    "emit_x_norm_mm_mrad",
    "emit_y_norm_mm_mrad",
]


def _finite_float(
    row: dict[str, Any], key: str, default: float = float("nan")
) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _format_threshold(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def log_component(value: float, minimum: float, reference: float) -> float:
    """Return a saturating 0..1 logarithmic component.

    Useful for quantities where an order-of-magnitude improvement matters less
    once the case is already good enough, e.g. hot charge and macroparticle count.
    """
    if not math.isfinite(value) or value <= minimum:
        return 0.0
    if reference <= minimum:
        return 1.0
    return clamp(math.log10(value / minimum) / math.log10(reference / minimum))


def linear_component(value: float, minimum: float, reference: float) -> float:
    """Return a saturating 0..1 linear component."""
    if not math.isfinite(value) or value <= minimum:
        return 0.0
    if reference <= minimum:
        return 1.0
    return clamp((value - minimum) / (reference - minimum))


def mono_proxy_from_energies(e95_mev: float, emax_mev: float) -> float:
    """Proxy for spectral compactness: high E95/Emax avoids one-particle tails."""
    if not math.isfinite(e95_mev) or not math.isfinite(emax_mev) or emax_mev <= 0.0:
        return float("nan")
    return e95_mev / emax_mev


def divergence_component_from_row(row: dict[str, Any]) -> tuple[float, str]:
    """Return a soft divergence score component and the column used.

    Fase 1 normally has no divergence columns yet. In that case the component is
    neutral, so missing divergence does not penalize the score. Fase 2 can add
    theta/divergence columns without changing the beamlike scorer.
    """
    divergence = _finite_float(row, "divergence_rms_mrad")
    source = "divergence_rms_mrad"

    if not math.isfinite(divergence):
        divergence = _finite_float(row, "theta_rms_mrad")
        source = "theta_rms_mrad"

    if not math.isfinite(divergence):
        theta_x = _finite_float(row, "theta_x_rms_mrad")
        theta_y = _finite_float(row, "theta_y_rms_mrad")
        if math.isfinite(theta_x) and math.isfinite(theta_y):
            divergence = math.sqrt(theta_x * theta_x + theta_y * theta_y)
            source = "theta_x/y_rms_mrad"

    if not math.isfinite(divergence):
        return 1.0, "not_available"

    # Soft placeholder: 0-5 mrad is good, ~20 mrad is strongly penalized.
    # This remains deliberately conservative until the divergence definition is
    # validated against the actual WarpX/openPMD particle output for each geometry.
    return clamp(1.0 / (1.0 + divergence / 8.0)), source


def _empty_transverse_columns() -> dict[str, str]:
    return {
        "theta_x_rms_mrad": "",
        "theta_y_rms_mrad": "",
        "theta_rms_mrad": "",
        "theta_x_p95_mrad": "",
        "theta_y_p95_mrad": "",
        "emit_x_norm_mm_mrad": "",
        "emit_y_norm_mm_mrad": "",
    }


def score_particle_summary_row(
    row: dict[str, Any],
    *,
    config: BeamlikeConfig | None = None,
) -> dict[str, Any]:
    """Compute beamlike eligibility, score components, and rejection reasons.

    Input is one already-reduced particle summary row, typically produced by
    cap_guiding.particles.summarize_dump(). The returned dictionary is intended
    to be merged into particle_summary.csv.
    """
    cfg = config or BeamlikeConfig()

    charge = _finite_float(row, "charge_hot_pC")
    n_hot = _finite_float(row, "n_macroparticles_hot")
    e95 = _finite_float(row, "E95_hot_MeV")
    emean = _finite_float(row, "Emean_hot_MeV")
    emax = _finite_float(row, "Emax_hot_MeV")
    q_long_min = _finite_float(row, "q_long_min_hot_mm")
    q_long_max = _finite_float(row, "q_long_max_hot_mm")

    hard_rejection_reasons: list[str] = []
    quality_reasons: list[str] = []

    if not math.isfinite(n_hot) or n_hot < cfg.min_hot_macroparticles:
        hard_rejection_reasons.append(
            f"low_n_hot<{_format_threshold(cfg.min_hot_macroparticles)}"
        )
    if not math.isfinite(charge) or charge < cfg.min_hot_charge_pC:
        hard_rejection_reasons.append(
            f"low_charge<{_format_threshold(cfg.min_hot_charge_pC)}pC"
        )
    if not math.isfinite(e95) or e95 < cfg.min_hot_E95_MeV:
        hard_rejection_reasons.append(
            f"low_E95<{_format_threshold(cfg.min_hot_E95_MeV)}MeV"
        )

    eligible = len(hard_rejection_reasons) == 0

    charge_component = log_component(
        charge,
        cfg.min_hot_charge_pC,
        cfg.charge_ref_pC,
    )
    statistics_component = log_component(
        n_hot,
        cfg.min_hot_macroparticles,
        cfg.n_hot_ref,
    )

    robust_energy = 0.65 * e95 + 0.30 * emean + 0.05 * emax
    energy_component = linear_component(
        robust_energy,
        cfg.min_hot_E95_MeV,
        cfg.energy_ref_MeV,
    )

    mono_proxy = mono_proxy_from_energies(e95, emax)
    mono_component = linear_component(mono_proxy, cfg.min_mono_proxy, cfg.mono_ref)
    if math.isfinite(mono_proxy) and mono_proxy < cfg.min_mono_proxy:
        quality_reasons.append(
            f"broad_proxy_E95_over_Emax<{_format_threshold(cfg.min_mono_proxy)}"
        )

    if math.isfinite(q_long_min) and math.isfinite(q_long_max):
        z_span_hot_mm = max(q_long_max - q_long_min, 0.0)
        z_compact_component = 1.0 / (1.0 + z_span_hot_mm / cfg.z_span_ref_mm)
    else:
        z_span_hot_mm = float("nan")
        z_compact_component = 1.0

    divergence_component, divergence_source = divergence_component_from_row(row)

    if eligible:
        z_factor = (
            1.0 - cfg.z_compact_weight + cfg.z_compact_weight * z_compact_component
        )
        divergence_factor = (
            1.0 - cfg.divergence_weight + cfg.divergence_weight * divergence_component
        )
        beamlike_score = (
            cfg.score_scale
            * (charge_component**cfg.charge_exponent)
            * (statistics_component**cfg.statistics_exponent)
            * (energy_component**cfg.energy_exponent)
            * (mono_component**cfg.mono_exponent)
            * z_factor
            * divergence_factor
        )
        beam_yield_score = (
            math.log10(1.0 + max(charge, 0.0))
            * math.log10(1.0 + max(n_hot, 0.0))
            * max(e95, 0.0)
        )
    else:
        beamlike_score = 0.0
        beam_yield_score = 0.0

    tags: list[str] = []
    if eligible:
        tags.append("beamlike_candidate")
    if charge >= 1000.0:
        tags.append("nC_class")
    elif charge >= 300.0:
        tags.append("high_charge")
    elif charge >= cfg.min_hot_charge_pC:
        tags.append("usable_charge")

    if n_hot >= 1000.0:
        tags.append("good_statistics")
    elif n_hot >= cfg.min_hot_macroparticles:
        tags.append("usable_statistics")

    if e95 >= 180.0:
        tags.append("high_E95")
    elif e95 >= 100.0:
        tags.append("medium_E95")

    if emax >= 300.0:
        tags.append("high_Emax")

    if math.isfinite(mono_proxy):
        if mono_proxy >= 0.60:
            tags.append("compact_spectrum_proxy")
        elif mono_proxy < 0.35:
            tags.append("broad_spectrum_proxy")

    if divergence_source == "not_available":
        tags.append("no_divergence_metric")

    if not math.isfinite(n_hot) or n_hot < cfg.min_hot_macroparticles:
        status = "insufficient_hot_electron_statistics"
    elif not math.isfinite(charge) or charge < cfg.min_hot_charge_pC:
        status = "insufficient_hot_charge"
    elif not math.isfinite(e95) or e95 < cfg.min_hot_E95_MeV:
        status = "insufficient_hot_energy"
    elif quality_reasons:
        status = "eligible_with_quality_flags"
    else:
        status = "eligible_beamlike"

    reasons = [*hard_rejection_reasons, *quality_reasons]

    return {
        "eligible_beamlike": bool(eligible),
        "beamlike_status": status,
        "beamlike_score": float(beamlike_score),
        "beam_yield_score": float(beam_yield_score),
        "beamlike_rejection_reasons": ";".join(reasons),
        "beamlike_tags": ";".join(tags),
        "mono_proxy_E95_over_Emax": mono_proxy if math.isfinite(mono_proxy) else "",
        "z_span_hot_mm": z_span_hot_mm if math.isfinite(z_span_hot_mm) else "",
        "statistics_component": float(statistics_component),
        "charge_component": float(charge_component),
        "energy_component": float(energy_component),
        "mono_component": float(mono_component),
        "z_compact_component": float(z_compact_component),
        "divergence_component": float(divergence_component),
        "divergence_source": divergence_source,
        **_empty_transverse_columns(),
    }


def add_beamlike_metrics(
    row: dict[str, Any],
    *,
    config: BeamlikeConfig | None = None,
) -> dict[str, Any]:
    """Return a copy of a particle summary row with beamlike metrics added."""
    out = dict(row)
    metrics = score_particle_summary_row(out, config=config)
    out.update(metrics)
    return out
