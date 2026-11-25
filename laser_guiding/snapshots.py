"""Select immutable, exact HDF5 snapshots without scanning a growing viewer series."""
from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py

from .openpmd_io import open_series


@dataclass(frozen=True)
class Source:
    path: Path
    size_bytes: int
    mtime_ns: int

    def check(self) -> None:
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.size_bytes, self.mtime_ns):
            raise RuntimeError(f"Source changed during analysis: {self.path}")

    def metadata(self) -> dict:
        return {"path": str(self.path), "size_bytes": self.size_bytes,
                "mtime_ns": self.mtime_ns}


def completed_step(log: str | Path) -> int:
    with Path(log).open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 1048576))
        tail = stream.read().decode("utf-8", errors="replace")
    steps = re.findall(r"^STEP\s+(\d+) ends\.", tail, flags=re.MULTILINE)
    if not steps:
        raise ValueError(f"No completed STEP record in {log}")
    return max(map(int, steps))


def inventory(directory: str | Path, *, min_age_seconds: float = 600.0,
              max_iteration: int | None = None) -> dict[int, Source]:
    """Age is an extra guard, not evidence that a simulation completed.

    Use a completed-step log for a running case. HDF5 locking stays enabled.
    """
    if min_age_seconds < 0:
        raise ValueError("min_age_seconds must be nonnegative")
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.glob("*.h5")):
        stat = path.stat()
        source = Source(path, stat.st_size, stat.st_mtime_ns)
        if time.time() - source.mtime_ns / 1e9 < min_age_seconds:
            continue
        try:
            with h5py.File(path, "r") as handle:
                iterations = list(map(int, handle["data"].keys()))
            source.check()
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            print(f"[SKIP] {path.name}: {exc}", flush=True)
            continue
        for it in iterations:
            if max_iteration is not None and it > max_iteration:
                continue
            if it in result:
                raise ValueError(f"Duplicate diagnostic iteration {it}: {directory}")
            result[it] = source
    return result


def select_iterations(fields: dict[int, Source], particles: dict[int, Source] | None,
                      requested: list[int] | None) -> list[int]:
    if requested is None:
        if particles is not None and set(fields) != set(particles):
            raise ValueError("Eligible field/particle iterations differ; request exact common iterations")
        selected = sorted(fields)
    else:
        if len(set(requested)) != len(requested) or any(it < 0 for it in requested):
            raise ValueError("Requested iterations must be unique and nonnegative")
        selected = sorted(requested)
        missing = set(selected) - set(fields)
        if particles is not None:
            missing |= set(selected) - set(particles)
        if missing:
            raise ValueError(f"Exact synchronized iterations unavailable: {sorted(missing)}")
    if not selected:
        raise ValueError("No eligible snapshots")
    return selected


@contextmanager
def snapshot_series(source: Source, iteration: int):
    """Only expose the selected file to openPMD-viewer, including its own species."""
    source.check()
    with TemporaryDirectory(prefix="laser_guiding_reader_") as tmp:
        (Path(tmp) / source.path.name).symlink_to(source.path)
        series = open_series(tmp, backend="h5py", check_all_files=False)
        if int(iteration) not in set(map(int, series.iterations)):
            raise ValueError(f"Exact iteration {iteration} absent from {source.path}")
        yield series
    source.check()
