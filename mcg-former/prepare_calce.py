"""Read the six CALCE CS2 Arbin exports and write cycle arrays."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from lxml import etree as ET
import numpy as np

CELLS = {
    "CS2_33": 0.55, "CS2_34": 0.55, "CS2_35": 1.10,
    "CS2_36": 1.10, "CS2_37": 1.10, "CS2_38": 1.10,
}
Q_RATED = 1.10
EOL_AH = 0.880
CYCLE_LEN = 100
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REQUIRED = (
    "Date_Time", "Step_Time(s)", "Step_Index", "Cycle_Index",
    "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)",
)


@dataclass
class Cycle:
    start: int
    end: int
    source: str
    local_index: int
    capacity: float
    features: np.ndarray


def channel_sheets(book):
    workbook = ET.fromstring(book.read("xl/workbook.xml"))
    links = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
    targets = {link.get("Id"): link.get("Target") for link in links}
    paths = []
    for sheet in workbook.findall(f".//{NS}sheet"):
        if sheet.get("name", "").lower().startswith("channel"):
            target = targets[sheet.get(REL + "id")]
            paths.append(target.lstrip("/") if target.startswith("/") else "xl/" + target)
    if not paths:
        raise ValueError("No Arbin Channel worksheet found.")
    return paths


def sheet_rows(book, sheet, headers):
    with book.open(sheet) as stream:
        wanted = None
        for _, row in ET.iterparse(stream, events=("end",), tag=NS + "row"):
            if wanted is None:
                names = {}
                for cell in row:
                    raw = cell.findtext(NS + "v")
                    if cell.get("t") == "s":
                        raw = headers[int(raw)]
                    elif cell.get("t") == "inlineStr":
                        raw = "".join(cell.itertext())
                    if raw:
                        ref = cell.get("r")
                        names[raw.strip()] = ref.rstrip("0123456789")
                missing = sorted(set(REQUIRED) - names.keys())
                if missing:
                    raise ValueError(f"{sheet}: columns missing: {missing}")
                wanted = {names[name]: i for i, name in enumerate(REQUIRED)}
                yield None, set(names)
            else:
                values = [None] * len(REQUIRED)
                for cell in row:
                    idx = wanted.get(cell.get("r", "").rstrip("0123456789"))
                    if idx is not None:
                        raw = cell.findtext(NS + "v")
                        if raw is not None:
                            values[idx] = float(raw)
                if all(value is not None for value in values):
                    yield values, None
            row.clear()
            while row.getprevious() is not None:
                del row.getparent()[0]


def features(voltage, current):
    # Preserve the NASA implementation's sample-position interpolation and IC proxy.
    source_x = np.linspace(0, 1, len(voltage))
    target_x = np.linspace(0, 1, CYCLE_LEN)
    volts = np.interp(target_x, source_x, voltage)
    amps = np.interp(target_x, source_x, current)
    dq = np.gradient(np.cumsum(np.abs(amps)))
    dv = np.gradient(volts) + 1e-6
    with np.errstate(divide="ignore"):
        ic = np.clip(dq / dv, -50, 50)
    return np.column_stack((volts / 4.2, amps / 2.0, ic / 50.0)).astype(np.float32)


def add_discharge(records, previous, source, nominal_current, found, rejected):
    if not records:
        return
    data = np.asarray(records)
    stamp, step_time, _, local_index, current, voltage, cum_charge = data.T
    if abs(float(np.median(current)) + nominal_current) > 0.05 * nominal_current:
        rejected["current"] += 1
        return
    # Complete discharge with a recorded start and cutoff.
    if len(records) < 3 or not 2.68 <= voltage[-1] <= 2.72:
        rejected["incomplete"] += 1
        return
    if previous is None or previous[3] != local_index[0] or previous[4] < -0.15:
        rejected["missing_charge_baseline"] += 1
        return
    capacity = float(cum_charge[-1] - previous[-1])
    integrated = float(np.dot(-current, np.diff(step_time, prepend=0)) / 3600)
    if capacity <= 0 or abs(integrated - capacity) > max(0.015, 0.04 * capacity):
        rejected["capacity_counter"] += 1
        return
    found.append(Cycle(round(stamp[0] * 86400), round(stamp[-1] * 86400),
                       source, int(local_index[0]), capacity, features(voltage, current)))


def read_book(raw, source, nominal_current, found, rejected):
    all_headers = set()
    with ZipFile(BytesIO(raw)) as book:
        header_path = "xl/sharedStrings.xml"
        headers = []
        if header_path in book.namelist():
            root = ET.fromstring(book.read(header_path))
            headers = ["".join(node.itertext()) for node in root]
        for sheet in channel_sheets(book):
            stream = sheet_rows(book, sheet, headers)
            _, columns = next(stream)
            all_headers.update(columns)
            last_row = previous = None
            segment = []
            for row, _ in stream:
                discharge = row[4] < -0.15
                if segment and (not discharge or row[2:4] != segment[-1][2:4]):
                    add_discharge(segment, previous, source, nominal_current, found, rejected)
                    segment = []
                if discharge:
                    if not segment:
                        previous = last_row
                    segment.append(row)
                last_row = row
            add_discharge(segment, previous, source, nominal_current, found, rejected)
    return all_headers


def source_books(raw_dir, cell):
    archive = raw_dir / f"{cell}.zip"
    if archive.is_file():
        with ZipFile(archive) as bundle:
            for entry in sorted(bundle.infolist(), key=lambda item: item.filename):
                if entry.filename.lower().endswith(".xlsx"):
                    yield PurePosixPath(entry.filename).name, bundle.read(entry)
        return
    folder = raw_dir / cell
    for entry in sorted(folder.rglob("*.xlsx")):
        if not entry.name.startswith("~$"):
            yield entry.name, entry.read_bytes()


def eol_cycle(capacity, threshold=EOL_AH, stable=2):
    """Find the first of consecutive observed discharges at or below EOL."""
    capacity = np.asarray(capacity, dtype=np.float64)
    if len(capacity) < stable:
        return None
    runs = np.lib.stride_tricks.sliding_window_view(capacity <= threshold, stable)
    matches = np.flatnonzero(np.all(runs, axis=1))
    return int(matches[0] + 1) if matches.size else None


def prepare_cell(cell, raw_dir, output_dir):
    found = []
    headers = set()
    rejected = dict.fromkeys(("current", "incomplete", "missing_charge_baseline",
                              "capacity_counter", "duplicate_timestamp"), 0)
    count = 0
    for source, raw in source_books(raw_dir, cell):
        count += 1
        try:
            headers.update(read_book(raw, source, CELLS[cell], found, rejected))
        except (ValueError, KeyError, IndexError, StopIteration) as error:
            raise ValueError(f"{cell}/{source}: {error}") from error
    if not count or not found:
        raise ValueError(f"{cell}: no usable cycles in {raw_dir}")
    unique = {}
    for cycle in sorted(found, key=lambda item: (item.start, item.end, item.source)):
        key = cycle.start, cycle.end
        if key in unique:
            prior = unique[key]
            if not np.isclose(prior.capacity, cycle.capacity, atol=0.005):
                raise ValueError(f"{cell}: conflicting records for timestamp {key}.")
            rejected["duplicate_timestamp"] += 1
        else:
            unique[key] = cycle

    cycles = list(unique.values())
    raw_capacity = np.array([cycle.capacity for cycle in cycles], dtype=np.float64)
    eol = eol_cycle(raw_capacity)
    if eol is None:
        raise ValueError(f"{cell}: no observed EOL at {EOL_AH:.3f} Ah.")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / f"{cell}.npz"
    tmp_path = output_dir / f"{cell}.tmp.npz"
    np.savez_compressed(tmp_path, features=np.stack([cycle.features for cycle in cycles]),
                        capacity_ah=raw_capacity, eol_cycle=eol)
    tmp_path.replace(arrays_path)
    with (output_dir / f"{cell}_cycles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("cycle", "capacity_ah", "source", "local_cycle",
                         "start_excel_epoch_seconds", "end_excel_epoch_seconds"))
        writer.writerows((i, c.capacity, c.source, c.local_index, c.start, c.end)
                         for i, c in enumerate(cycles, start=1))
    return {
        "battery": cell, "workbooks": count, "valid_cycles": len(cycles),
        "first_capacity_ah": round(float(raw_capacity[0]), 6),
        "capacity_at_eol_ah": round(float(raw_capacity[eol - 1]), 6),
        "last_capacity_ah": round(float(raw_capacity[-1]), 6),
        "eol_cycle": eol, "rul_at_300": eol - 300, "rul_at_400": eol - 400,
        "rul_at_500": eol - 500, "has_temperature_header": any("temp" in c.lower() for c in headers),
        **rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/CALCE", type=Path)
    parser.add_argument("--output-dir", default="data/processed_calce", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(prepare_cell, cell, args.raw_dir, args.output_dir) for cell in CELLS]
        diagnostics = []
        for job in jobs:
            result = job.result()
            print(result, flush=True)
            diagnostics.append(result)
    with (args.output_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)


if __name__ == "__main__":
    main()
