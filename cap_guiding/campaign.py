from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CASE_TYPES = ("channel", "uniform", "vacuum")

_LASER_RE = re.compile(r"(?:^|_)f(?P<fnum>\d+)(?:_|$)", re.IGNORECASE)
_DENSITY_RE = re.compile(r"(?:^|_)n(?P<density>[^_]+)(?:_|$)", re.IGNORECASE)
_REF_DENSITY_RE = re.compile(r"(?:^|_)refn(?P<density>[^_]+)(?:_|$)", re.IGNORECASE)
_PLATEAU_RE = re.compile(
    r"(?:^|_)L(?P<plateau>\d+(?:[p.]\d+)?)mm(?:_|$)",
    re.IGNORECASE,
)
_FOCUS_RE = re.compile(
    r"(?:^|_)foc(?P<focus>[mp]?\d+(?:[p.]\d+)?)um(?:_|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    case_type: str
    case_dir: Path
    diag_dir: Path
    h5_count: int
    laser_case: str | None
    density: str | None
    ref_density: str | None
    plateau: str | None
    focus: str | None
    source: str

    @property
    def density_key(self) -> str | None:
        if self.case_type == "vacuum":
            return self.ref_density
        return self.density

    @property
    def base_key(self) -> tuple[str | None, str | None, str | None]:
        return (self.laser_case, self.plateau, self.focus)

    @property
    def full_key(self) -> tuple[str | None, str | None, str | None, str | None]:
        return (self.laser_case, self.plateau, self.focus, self.density_key)


@dataclass(frozen=True)
class TripletInfo:
    key: tuple[str | None, str | None, str | None, str | None]
    channel: CaseInfo | None
    uniform: CaseInfo | None
    vacuum: CaseInfo | None

    @property
    def complete(self) -> bool:
        return (
            self.channel is not None
            and self.uniform is not None
            and self.vacuum is not None
        )

    @property
    def label(self) -> str:
        fnum, plateau, focus, density = self.key
        parts = [
            f"f{fnum}" if fnum is not None else None,
            f"n{density}" if density is not None else None,
            f"L{plateau}mm" if plateau is not None else None,
            f"foc{focus}um" if focus is not None else None,
        ]
        return "_".join(p for p in parts if p is not None)


def _norm_token(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace(".", "p").lower()


def _first_match(regex: re.Pattern[str], text: str, group: str) -> str | None:
    match = regex.search(text)
    if match is None:
        return None
    return _norm_token(match.group(group))


def infer_case_type(case_id: str) -> str | None:
    name = case_id.lower()
    tokens = re.split(r"[_\-/\\]+", name)

    if "chan" in tokens or "channel" in tokens:
        return "channel"
    if "uni" in tokens or "uniform" in tokens:
        return "uniform"
    if "vac" in tokens or "vacuum" in tokens:
        return "vacuum"

    return None


def iter_h5_files(diag_dir: str | Path) -> list[Path]:
    diag_dir = Path(diag_dir)
    if not diag_dir.is_dir():
        return []
    return sorted(path for path in diag_dir.rglob("*.h5") if path.is_file())


def count_h5(diag_dir: str | Path) -> int:
    return len(iter_h5_files(diag_dir))


def newest_h5_mtime(diag_dir: str | Path) -> float | None:
    h5_files = iter_h5_files(diag_dir)
    if not h5_files:
        return None
    return max(path.stat().st_mtime for path in h5_files)


def newest_h5_age_min(diag_dir: str | Path, now: float | None = None) -> float | None:
    mtime = newest_h5_mtime(diag_dir)
    if mtime is None:
        return None

    if now is None:
        now = time.time()

    return max(0.0, (now - mtime) / 60.0)


def case_has_min_h5(case: CaseInfo, min_h5: int) -> bool:
    return case.h5_count >= min_h5


def case_has_stable_h5(
    case: CaseInfo,
    min_last_h5_age_min: float = 0.0,
    now: float | None = None,
) -> bool:
    """Return True if the newest HDF5 file is old enough.

    min_last_h5_age_min=0 disables this stability gate.
    """
    if min_last_h5_age_min <= 0.0:
        return True

    age = newest_h5_age_min(case.diag_dir, now=now)
    return age is not None and age >= min_last_h5_age_min


def case_is_ready(
    case: CaseInfo,
    min_h5: int,
    min_last_h5_age_min: float = 0.0,
    now: float | None = None,
) -> bool:
    return case_has_min_h5(case, min_h5=min_h5) and case_has_stable_h5(
        case,
        min_last_h5_age_min=min_last_h5_age_min,
        now=now,
    )


def triplet_ready_min_h5(triplet: TripletInfo, min_h5: int) -> bool:
    if not triplet.complete:
        return False

    return all(
        case is not None and case_has_min_h5(case, min_h5=min_h5)
        for case in [triplet.channel, triplet.uniform, triplet.vacuum]
    )


def triplet_is_ready(
    triplet: TripletInfo,
    min_h5: int,
    min_last_h5_age_min: float = 0.0,
    now: float | None = None,
) -> bool:
    if not triplet.complete:
        return False

    return all(
        case is not None
        and case_is_ready(
            case,
            min_h5=min_h5,
            min_last_h5_age_min=min_last_h5_age_min,
            now=now,
        )
        for case in [triplet.channel, triplet.uniform, triplet.vacuum]
    )


def infer_case_info_from_dir(
    case_dir: str | Path, source: str = "name"
) -> CaseInfo | None:
    case_dir = Path(case_dir)
    case_id = case_dir.name
    case_type = infer_case_type(case_id)

    if case_type is None:
        return None

    diag_dir = case_dir / "diags" / "diag1"

    return CaseInfo(
        case_id=case_id,
        case_type=case_type,
        case_dir=case_dir,
        diag_dir=diag_dir,
        h5_count=count_h5(diag_dir),
        laser_case=_first_match(_LASER_RE, case_id, "fnum"),
        density=_first_match(_DENSITY_RE, case_id, "density"),
        ref_density=_first_match(_REF_DENSITY_RE, case_id, "density"),
        plateau=_first_match(_PLATEAU_RE, case_id, "plateau"),
        focus=_first_match(_FOCUS_RE, case_id, "focus"),
        source=source,
    )


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _read_cases_full_dataframe(cases_path: Path) -> pd.DataFrame:
    """Read cases_full.tsv robustly.

    Preferred format is true TSV. For local/manual probes, also tolerate
    whitespace-separated tables. If the file exists but cannot be parsed as a
    table with multiple columns, fail loudly instead of silently detecting zero
    cases.
    """
    attempts = [
        ("tab", {"sep": "\t"}),
        ("whitespace", {"sep": r"\s+", "engine": "python"}),
        ("comma", {"sep": ","}),
    ]

    errors: list[str] = []

    for name, kwargs in attempts:
        try:
            df = pd.read_csv(
                cases_path,
                dtype=str,
                comment="#",
                **kwargs,
            ).fillna("")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue

        df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]

        # Strip string cells as well.
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        if len(df.columns) >= 2:
            return df

        errors.append(f"{name}: parsed only one column: {list(df.columns)}")

    raise ValueError(
        f"Could not parse {cases_path} as a useful cases table. " + " | ".join(errors)
    )


def load_cases_from_cases_full(campaign_root: str | Path) -> list[CaseInfo]:
    campaign_root = Path(campaign_root)
    cases_path = campaign_root / "cases_full.tsv"

    if not cases_path.is_file():
        return []

    df = _read_cases_full_dataframe(cases_path)
    columns = list(df.columns)

    # Prefer columns that contain the actual case folder/name over numeric IDs.
    # In the F32 campaign, CASE_ID is "000" but CASE_NAME is the real directory:
    # 000_f32_chan_n7e17cm3_L5mm_focm500um_rz
    case_col = _pick_column(
        columns,
        [
            "case_name",
            "case_dir_name",
            "case_folder",
            "case",
            "name",
            "run_name",
            "run",
            "run_id",
            "case_id",
        ],
    )
    path_col = _pick_column(
        columns,
        [
            "case_dir",
            "dir",
            "directory",
            "path",
            "run_dir",
            "case_path",
        ],
    )
    diag_col = _pick_column(
        columns,
        [
            "diag",
            "diag_dir",
            "diag_path",
        ],
    )
    type_col = _pick_column(
        columns,
        [
            "case_type",
            "type",
            "profile",
            "plasma",
            "plasma_kind",
            "kind",
        ],
    )

    if case_col is None and path_col is None:
        raise ValueError(
            f"{cases_path} exists but does not contain a recognizable case/path "
            f"column. Parsed columns: {columns}"
        )

    cases: list[CaseInfo] = []

    for _, row in df.iterrows():
        raw_case = str(row[case_col]).strip() if case_col else ""
        raw_path = str(row[path_col]).strip() if path_col else ""
        raw_diag = str(row[diag_col]).strip() if diag_col else ""

        if raw_path:
            case_dir = Path(raw_path)
            if not case_dir.is_absolute():
                case_dir = campaign_root / case_dir
        elif raw_case:
            case_dir = campaign_root / raw_case
        else:
            continue

        inferred = infer_case_info_from_dir(case_dir, source="cases_full.tsv")
        if inferred is None:
            continue

        case_type = inferred.case_type
        if type_col:
            explicit = infer_case_type(str(row[type_col]).strip())
            if explicit is not None:
                case_type = explicit

        diag_dir = inferred.diag_dir
        if raw_diag:
            diag_dir = Path(raw_diag)
            if not diag_dir.is_absolute():
                diag_dir = campaign_root / diag_dir

        cases.append(
            CaseInfo(
                case_id=inferred.case_id,
                case_type=case_type,
                case_dir=case_dir,
                diag_dir=diag_dir,
                h5_count=count_h5(diag_dir),
                laser_case=inferred.laser_case,
                density=inferred.density,
                ref_density=inferred.ref_density,
                plateau=inferred.plateau,
                focus=inferred.focus,
                source="cases_full.tsv",
            )
        )

    if not cases:
        raise ValueError(
            f"{cases_path} was parsed, but no recognizable campaign cases were "
            "found. Check case names, case_dir paths, and type/profile columns."
        )

    return cases


def discover_cases_by_name(campaign_root: str | Path) -> list[CaseInfo]:
    campaign_root = Path(campaign_root)
    cases: list[CaseInfo] = []

    for child in sorted(campaign_root.iterdir()):
        if not child.is_dir():
            continue

        info = infer_case_info_from_dir(child, source="name")
        if info is not None:
            cases.append(info)

    return cases


def discover_cases(campaign_root: str | Path) -> list[CaseInfo]:
    cases = load_cases_from_cases_full(campaign_root)
    if cases:
        return cases

    return discover_cases_by_name(campaign_root)


def build_triplets(cases: list[CaseInfo]) -> list[TripletInfo]:
    channels = [c for c in cases if c.case_type == "channel"]
    uniforms = [c for c in cases if c.case_type == "uniform"]
    vacuums = [c for c in cases if c.case_type == "vacuum"]

    uniform_by_key: dict[
        tuple[str | None, str | None, str | None, str | None], CaseInfo
    ] = {}
    for case in uniforms:
        uniform_by_key[case.full_key] = case

    vacuum_by_full_key: dict[
        tuple[str | None, str | None, str | None, str | None], CaseInfo
    ] = {}
    vacuum_by_base_key: dict[tuple[str | None, str | None, str | None], CaseInfo] = {}

    for case in vacuums:
        if case.ref_density is not None:
            vacuum_by_full_key[case.full_key] = case

        vacuum_by_base_key.setdefault(case.base_key, case)

    triplets: list[TripletInfo] = []
    used_keys: set[tuple[str | None, str | None, str | None, str | None]] = set()

    for channel in channels:
        key = channel.full_key
        uniform = uniform_by_key.get(key)
        vacuum = vacuum_by_full_key.get(key) or vacuum_by_base_key.get(channel.base_key)

        triplets.append(
            TripletInfo(
                key=key,
                channel=channel,
                uniform=uniform,
                vacuum=vacuum,
            )
        )
        used_keys.add(key)

    for uniform in uniforms:
        key = uniform.full_key
        if key in used_keys:
            continue

        vacuum = vacuum_by_full_key.get(key) or vacuum_by_base_key.get(uniform.base_key)

        triplets.append(
            TripletInfo(
                key=key,
                channel=None,
                uniform=uniform,
                vacuum=vacuum,
            )
        )
        used_keys.add(key)

    return sorted(triplets, key=lambda t: t.label)


def cases_with_insufficient_h5(cases: list[CaseInfo], min_h5: int) -> list[CaseInfo]:
    return [case for case in cases if case.h5_count < min_h5]


def write_campaign_report(
    cases: list[CaseInfo],
    triplets: list[TripletInfo],
    outdir: str | Path,
    min_h5: int,
    min_last_h5_age_min: float = 0.0,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    now = time.time()

    paths = {
        "cases": outdir / "campaign_cases.csv",
        "triplets": outdir / "campaign_triplets.csv",
        "insufficient_h5": outdir / "campaign_insufficient_h5.csv",
        "unstable_h5": outdir / "campaign_unstable_h5.csv",
    }

    with paths["cases"].open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "case_type",
            "case_dir",
            "diag_dir",
            "h5_count",
            "newest_h5_age_min",
            "ready_min_h5",
            "stable_min_age",
            "ready_for_analysis",
            "laser_case",
            "density",
            "ref_density",
            "plateau",
            "focus",
            "source",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            age = newest_h5_age_min(case.diag_dir, now=now)
            ready_min = case_has_min_h5(case, min_h5=min_h5)
            stable = case_has_stable_h5(
                case,
                min_last_h5_age_min=min_last_h5_age_min,
                now=now,
            )

            writer.writerow(
                {
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "case_dir": case.case_dir,
                    "diag_dir": case.diag_dir,
                    "h5_count": case.h5_count,
                    "newest_h5_age_min": "" if age is None else f"{age:.3f}",
                    "ready_min_h5": ready_min,
                    "stable_min_age": stable,
                    "ready_for_analysis": ready_min and stable,
                    "laser_case": case.laser_case,
                    "density": case.density,
                    "ref_density": case.ref_density,
                    "plateau": case.plateau,
                    "focus": case.focus,
                    "source": case.source,
                }
            )

    with paths["triplets"].open("w", newline="") as f:
        fieldnames = [
            "label",
            "complete",
            "ready_min_h5",
            "stable_min_age",
            "ready_for_analysis",
            "channel",
            "uniform",
            "vacuum",
            "channel_h5",
            "uniform_h5",
            "vacuum_h5",
            "channel_newest_h5_age_min",
            "uniform_newest_h5_age_min",
            "vacuum_newest_h5_age_min",
            "min_last_h5_age_min",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for triplet in triplets:
            channel_age = (
                newest_h5_age_min(triplet.channel.diag_dir, now=now)
                if triplet.channel is not None
                else None
            )
            uniform_age = (
                newest_h5_age_min(triplet.uniform.diag_dir, now=now)
                if triplet.uniform is not None
                else None
            )
            vacuum_age = (
                newest_h5_age_min(triplet.vacuum.diag_dir, now=now)
                if triplet.vacuum is not None
                else None
            )

            ready_min = triplet_ready_min_h5(triplet, min_h5=min_h5)
            stable = triplet.complete and all(
                case is not None
                and case_has_stable_h5(
                    case,
                    min_last_h5_age_min=min_last_h5_age_min,
                    now=now,
                )
                for case in [triplet.channel, triplet.uniform, triplet.vacuum]
            )

            writer.writerow(
                {
                    "label": triplet.label,
                    "complete": triplet.complete,
                    "ready_min_h5": ready_min,
                    "stable_min_age": stable,
                    "ready_for_analysis": ready_min and stable,
                    "channel": triplet.channel.case_id if triplet.channel else "",
                    "uniform": triplet.uniform.case_id if triplet.uniform else "",
                    "vacuum": triplet.vacuum.case_id if triplet.vacuum else "",
                    "channel_h5": triplet.channel.h5_count if triplet.channel else "",
                    "uniform_h5": triplet.uniform.h5_count if triplet.uniform else "",
                    "vacuum_h5": triplet.vacuum.h5_count if triplet.vacuum else "",
                    "channel_newest_h5_age_min": ""
                    if channel_age is None
                    else f"{channel_age:.3f}",
                    "uniform_newest_h5_age_min": ""
                    if uniform_age is None
                    else f"{uniform_age:.3f}",
                    "vacuum_newest_h5_age_min": ""
                    if vacuum_age is None
                    else f"{vacuum_age:.3f}",
                    "min_last_h5_age_min": min_last_h5_age_min,
                }
            )

    bad = cases_with_insufficient_h5(cases, min_h5=min_h5)

    with paths["insufficient_h5"].open("w", newline="") as f:
        fieldnames = ["case_id", "case_type", "diag_dir", "h5_count", "min_h5"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in bad:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "diag_dir": case.diag_dir,
                    "h5_count": case.h5_count,
                    "min_h5": min_h5,
                }
            )

    unstable = [
        case
        for case in cases
        if case_has_min_h5(case, min_h5=min_h5)
        and not case_has_stable_h5(
            case,
            min_last_h5_age_min=min_last_h5_age_min,
            now=now,
        )
    ]

    with paths["unstable_h5"].open("w", newline="") as f:
        fieldnames = [
            "case_id",
            "case_type",
            "diag_dir",
            "h5_count",
            "newest_h5_age_min",
            "min_last_h5_age_min",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case in unstable:
            age = newest_h5_age_min(case.diag_dir, now=now)
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "diag_dir": case.diag_dir,
                    "h5_count": case.h5_count,
                    "newest_h5_age_min": "" if age is None else f"{age:.3f}",
                    "min_last_h5_age_min": min_last_h5_age_min,
                }
            )

    return paths
