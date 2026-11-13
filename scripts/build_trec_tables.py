"""Build patient, label, and judged-trial tables from the TREC Clinical Trials data.

Read <data_dir>/raw/trec_ct/ and write deterministic JSONL tables plus stats.json
to <data_dir>/processed/.

Usage:
    python scripts/build_trec_tables.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clinreason.config import load_config
from clinreason.data.trec import iter_trials, load_qrels, load_topics

YEARS = (2021, 2022)
SPLIT_STATUSES = (
    "both",
    "inclusion_only",
    "exclusion_only",
    "unsectioned",
    "no_criteria",
)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSON records, each followed by a newline."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="override config data_dir")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else Path(load_config()["data_dir"])
    raw_dir = data_dir / "raw" / "trec_ct"
    out_dir = data_dir / "processed"

    patients = []
    labels = []
    for year in YEARS:
        patients.extend(load_topics(raw_dir / f"topics{year}.xml", year))
        labels.extend(load_qrels(raw_dir / f"qrels{year}.txt", year))
    patients.sort(key=lambda row: (row["year"], row["topic"]))
    labels.sort(
        key=lambda row: (
            row["year"],
            int(row["patient_id"].rsplit("-", 1)[1]),
            row["nct_id"],
        )
    )

    judged = {row["nct_id"] for row in labels}
    archives = sorted(raw_dir.glob("ClinicalTrials.2021-04-27.part*.zip"))
    trials_by_id = {}
    for trial in iter_trials(archives, keep=judged):
        trials_by_id.setdefault(trial["nct_id"], trial)
    trials = [trials_by_id[nct_id] for nct_id in sorted(trials_by_id)]

    patients_per_year = {str(year): 0 for year in YEARS}
    label_counts_per_year = {str(year): dict.fromkeys(("0", "1", "2"), 0) for year in YEARS}
    split_status_counts = dict.fromkeys(SPLIT_STATUSES, 0)
    for patient in patients:
        patients_per_year[str(patient["year"])] += 1
    for label in labels:
        label_counts_per_year[str(label["year"])][str(label["label"])] += 1
    for trial in trials:
        split_status_counts[trial["split_status"]] += 1

    missing = sorted(judged - trials_by_id.keys())
    sectioned = sum(split_status_counts[status] for status in SPLIT_STATUSES[:3])
    sectioned_share = sectioned / len(judged) if judged else 0.0
    stats = {
        "patients_per_year": patients_per_year,
        "label_counts_per_year": label_counts_per_year,
        "judged_unique_trials": len(judged),
        "trials_found": len(trials),
        "missing_nct_ids": missing,
        "split_status_counts": split_status_counts,
        "sectioned_share": sectioned_share,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "patients.jsonl", patients)
    write_jsonl(out_dir / "labels.jsonl", labels)
    write_jsonl(out_dir / "trials.jsonl", trials)
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"done: {len(patients)} patients, {len(labels)} labels, {len(trials)} trials, "
        f"{len(missing)} missing; sectioned share {sectioned_share:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
