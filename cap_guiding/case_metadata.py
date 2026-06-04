from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_RAMP_UP_MM = 5.0

_PLATEAU_TOKEN_RE = re.compile(
    r"(?:^|[\\/\\s_-])L(?P<length>\d+(?:[p.]\d+)?)mm(?:$|[\\/\\s_-])",
    re.IGNORECASE,
)


def _parse_mm_number(token: str) -> float:
    """Parse campaign numeric tokens such as '5', '25', '2.5' or '2p5'."""
    return float(token.replace("p", "."))


def infer_plateau_length_mm_from_text(text: str) -> float | None:
    """Return plateau length encoded as L<length>mm in a case/path string.

    Examples accepted:
    - 003_f20_chan_n4e18cm3_L5mm_foc0um_rz
    - 078_f32_chan_n6e18cm3_L25mm_focp500um_rz/guiding_metrics.csv
    - C:\\runs\\case_L2p5mm_test
    """
    match = _PLATEAU_TOKEN_RE.search(str(text))
    if match is None:
        return None
    return _parse_mm_number(match.group("length"))


def plateau_window_from_length_mm(
    plateau_length_mm: float,
    ramp_up_mm: float = DEFAULT_RAMP_UP_MM,
) -> tuple[float, float]:
    """Return (plateau_start_mm, plateau_end_mm) from plateau length."""
    if plateau_length_mm <= 0.0:
        raise ValueError(f"plateau_length_mm must be > 0, got {plateau_length_mm}")
    return float(ramp_up_mm), float(ramp_up_mm + plateau_length_mm)


def infer_plateau_window_mm_from_text(
    text: str,
    ramp_up_mm: float = DEFAULT_RAMP_UP_MM,
) -> tuple[float, float] | None:
    """Infer plateau start/end positions from any case/path string."""
    plateau_length_mm = infer_plateau_length_mm_from_text(text)
    if plateau_length_mm is None:
        return None
    return plateau_window_from_length_mm(plateau_length_mm, ramp_up_mm=ramp_up_mm)


def infer_plateau_window_mm_from_sources(
    sources: Iterable[str],
    ramp_up_mm: float = DEFAULT_RAMP_UP_MM,
) -> tuple[float, float] | None:
    """Infer one consistent plateau window from source CSV paths/case names.

    Returns None if no source contains an L<length>mm token. Raises if several
    inconsistent plateau lengths are detected inside the same triplet.
    """
    windows: list[tuple[float, float]] = []

    for source in sources:
        window = infer_plateau_window_mm_from_text(source, ramp_up_mm=ramp_up_mm)
        if window is not None:
            windows.append(window)

    if not windows:
        return None

    unique = {(round(start, 9), round(end, 9)) for start, end in windows}
    if len(unique) != 1:
        raise ValueError(
            "Inconsistent plateau windows inferred from triplet sources: "
            f"{sorted(unique)}"
        )

    return windows[0]
