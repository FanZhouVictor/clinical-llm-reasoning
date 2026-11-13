"""Offline regression tests for TREC topics, judgments, trials, and table building."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from clinreason.data.trec import (
    iter_trials,
    load_qrels,
    load_topics,
    parse_age_years,
    parse_trial_xml,
    split_criteria,
)

FIXTURES = Path(__file__).parent / "fixtures"
TRIALS = FIXTURES / "trials"
REPO = Path(__file__).resolve().parents[1]
SPLIT_STATUSES = {
    "both",
    "inclusion_only",
    "exclusion_only",
    "unsectioned",
    "no_criteria",
}
TRIAL_KEYS = {
    "nct_id",
    "brief_title",
    "official_title",
    "conditions",
    "brief_summary",
    "overall_status",
    "phase",
    "study_type",
    "gender",
    "minimum_age_years",
    "maximum_age_years",
    "healthy_volunteers",
    "criteria_text",
    "inclusion_text",
    "exclusion_text",
    "split_status",
}


def _trial(nct_id):
    return parse_trial_xml((TRIALS / f"{nct_id}.xml").read_bytes())


def test_load_topics_uses_supplied_year_and_unescapes_text():
    topics = list(load_topics(FIXTURES / "topics_sample.xml", year=2022))
    assert len(topics) == 2
    assert [row["patient_id"] for row in topics] == ["2022-1", "2022-2"]
    assert [row["topic"] for row in topics] == [1, 2]
    assert all(type(row["topic"]) is int and row["year"] == 2022 for row in topics)
    assert all(set(row) == {"patient_id", "year", "topic", "text"} for row in topics)
    assert '"baby face"' in topics[0]["text"]
    assert "&quot;" not in topics[0]["text"]
    assert "β-hCG" in topics[1]["text"]
    assert all(row["text"] == " ".join(row["text"].split()) for row in topics)


def test_load_qrels_returns_only_label_record_fields():
    rows = list(load_qrels(FIXTURES / "qrels_sample.txt", year=2022))
    assert len(rows) == 6
    assert rows[0] == {
        "patient_id": "2022-1",
        "year": 2022,
        "nct_id": "NCT00000102",
        "label": 2,
    }
    assert {row["label"] for row in rows} == {0, 1, 2}
    assert all(type(row["label"]) is int for row in rows)
    assert all(set(row) == {"patient_id", "year", "nct_id", "label"} for row in rows)


@pytest.mark.parametrize(
    "line",
    [
        "1 0 NCT00000102 3",
        "1 0 NCT00000102 -1",
        "1 0 NCT00000102 eligible",
        "1 0 NCT00000102",
        "1 0 NCT00000102 2 extra",
        "one 0 NCT00000102 2",
    ],
)
def test_load_qrels_rejects_invalid_rows(tmp_path, line):
    path = tmp_path / "qrels.txt"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(load_qrels(path, year=2021))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("18 Years", 18.0),
        ("6 Months", 0.5),
        ("2 Weeks", 2 / 52),
        ("10 Days", 10 / 365),
        ("N/A", None),
        (None, None),
        ("", None),
        ("unknown", None),
        ("18 Fortnights", None),
        ("-1 Years", None),
    ],
)
def test_parse_age_years(text, expected):
    value = parse_age_years(text)
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)
        assert isinstance(value, float)


@pytest.mark.parametrize(
    "header",
    [
        "Inclusion Criteria:",
        "Key inclusion criteria:",
        "Main Inclusion Criteria:",
        "General inclusion criteria:",
        "Patient inclusion criteria:",
        "Subject inclusion criteria:",
        "Patient/Subject inclusion criteria:",
        "Inclusion criteria for patients:",
    ],
)
def test_split_criteria_recognizes_inclusion_header_variants(header):
    assert split_criteria(f"{header}\n- a") == ("- a", None, "inclusion_only")


@pytest.mark.parametrize(
    "header",
    [
        "Exclusion Criteria:",
        "Key exclusion criteria:",
        "Main exclusion criteria:",
        "General exclusion criteria:",
        "Patient exclusion criteria:",
        "Subject exclusion criteria:",
        "Non-inclusion criteria:",
        "Non inclusion criteria:",
    ],
)
def test_split_criteria_recognizes_exclusion_header_variants(header):
    assert split_criteria(f"{header}\n- b") == (None, "- b", "exclusion_only")


def test_split_criteria_distinguishes_non_inclusion_from_inclusion():
    text = "Inclusion Criteria:\n- a\nNon-inclusion criteria:\n- b"
    assert split_criteria(text) == ("- a", "- b", "both")


def test_split_criteria_preserves_content_after_header_colon():
    text = "Inclusion Criteria: - a\nExclusion Criteria: - b"
    assert split_criteria(text) == ("- a", "- b", "both")


@pytest.mark.parametrize("verb", ["include", "includes"])
def test_split_criteria_handles_inline_exclusion_sentence(verb):
    text = f"Patients with myopia are eligible. Exclusion criteria {verb} strabismus."
    inclusion, exclusion, status = split_criteria(text)
    assert status == "both"
    assert inclusion == "Patients with myopia are eligible."
    assert exclusion == "strabismus."


def test_split_criteria_ignores_preamble_when_inclusion_header_exists():
    text = "Administrative preamble.\nInclusion Criteria:\n- a\nExclusion Criteria:\n- b"
    assert split_criteria(text) == ("- a", "- b", "both")


def test_split_criteria_uses_preamble_as_inclusion_without_inclusion_header():
    text = "Patients with myopia are eligible.\nExclusion Criteria:\n- strabismus"
    assert split_criteria(text) == (
        "Patients with myopia are eligible.",
        "- strabismus",
        "both",
    )


def test_split_criteria_concatenates_repeated_blocks_in_document_order():
    text = (
        "Inclusion Criteria:\n- a\nExclusion Criteria:\n- b\n"
        "Inclusion Criteria:\n- c\nExclusion Criteria:\n- d"
    )
    assert split_criteria(text) == ("- a\n\n- c", "- b\n\n- d", "both")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, (None, None, "no_criteria")),
        (" \n\t", (None, None, "no_criteria")),
        ("Ordinary eligibility prose.", (None, None, "unsectioned")),
        ("Inclusion Criteria:\nExclusion Criteria:\n- b", (None, "- b", "exclusion_only")),
        ("Inclusion Criteria:\n- a\nExclusion Criteria:", ("- a", None, "inclusion_only")),
        ("Inclusion Criteria:\nExclusion Criteria:", (None, None, "unsectioned")),
    ],
)
def test_split_criteria_status_depends_on_nonempty_sections(text, expected):
    assert split_criteria(text) == expected


def test_parse_trial_with_both_sections_and_metadata():
    trial = _trial("NCT00000102")
    assert set(trial) == TRIAL_KEYS
    assert trial["nct_id"] == "NCT00000102"
    assert trial["brief_title"].startswith("Congenital Adrenal Hyperplasia:")
    assert trial["official_title"] is None
    assert trial["conditions"] == ["Congenital Adrenal Hyperplasia"]
    assert "extended release nifedipine" in trial["brief_summary"]
    assert trial["overall_status"] == "Completed"
    assert trial["phase"] == "Phase 1/Phase 2"
    assert trial["study_type"] == "Interventional"
    assert trial["gender"] == "All"
    assert trial["minimum_age_years"] == 14.0
    assert trial["maximum_age_years"] == 35.0
    assert trial["healthy_volunteers"] is False
    assert trial["split_status"] == "both"
    assert "Congenital Adrenal Hyperplasia" in trial["inclusion_text"]
    assert "history of liver disease" in trial["exclusion_text"]
    assert "liver disease" not in trial["inclusion_text"]


def test_parse_trial_with_only_inclusion_and_healthy_volunteers():
    trial = _trial("NCT00000108")
    assert trial["split_status"] == "inclusion_only"
    assert "Postmenopausal" in trial["inclusion_text"]
    assert trial["exclusion_text"] is None
    assert trial["gender"] == "Female"
    assert trial["minimum_age_years"] == 50.0
    assert trial["maximum_age_years"] == 65.0
    assert trial["healthy_volunteers"] is True
    assert trial["conditions"] == ["Cardiovascular Diseases", "Coronary Disease"]


def test_parse_trial_with_inline_exclusion():
    trial = _trial("NCT00000113")
    assert trial["split_status"] == "both"
    assert "myopia" in trial["inclusion_text"]
    assert "strabismus" not in trial["inclusion_text"]
    assert "strabismus" in trial["exclusion_text"]
    assert trial["official_title"] == "Correction of Myopia Evaluation Trial (COMET)"


def test_parse_trial_preserves_unsectioned_criteria():
    trial = _trial("NCT00000114")
    assert trial["split_status"] == "unsectioned"
    assert "Men and nonpregnant women" in trial["criteria_text"]
    assert trial["inclusion_text"] is None
    assert trial["exclusion_text"] is None
    assert trial["healthy_volunteers"] is None


def test_parse_trial_with_missing_criteria_and_unavailable_ages():
    trial = _trial("NCT00005736")
    assert trial["split_status"] == "no_criteria"
    assert trial["criteria_text"] is None
    assert trial["inclusion_text"] is None
    assert trial["exclusion_text"] is None
    assert trial["minimum_age_years"] is None
    assert trial["maximum_age_years"] is None
    assert trial["phase"] is None
    assert trial["study_type"] == "Observational"


def test_parse_trial_recognizes_key_inclusion_header():
    trial = _trial("NCT00043316")
    assert trial["split_status"] == "inclusion_only"
    assert "cystic fibrosis" in trial["inclusion_text"]
    assert "Key inclusion criteria" not in trial["inclusion_text"]
    assert trial["exclusion_text"] is None


def test_parse_trial_keeps_nested_scoring_headings_in_inclusion():
    trial = _trial("NCT02006511")
    assert trial["split_status"] == "both"
    assert "MAJOR 4 points" in trial["inclusion_text"]
    assert "Female gender" in trial["inclusion_text"]
    assert "MAJOR" not in trial["exclusion_text"]
    assert "Is pregnant" in trial["exclusion_text"]


@pytest.mark.parametrize("path", sorted(TRIALS.glob("*.xml")), ids=lambda p: p.stem)
def test_parse_trial_normalizes_newlines_and_trailing_whitespace(path):
    trial = parse_trial_xml(path.read_bytes())
    for value in trial.values():
        if isinstance(value, str):
            assert "\r" not in value
        elif isinstance(value, list):
            assert all("\r" not in item for item in value if isinstance(item, str))
    criteria = trial["criteria_text"]
    if criteria is not None:
        assert "\n" in criteria
        assert criteria == criteria.strip()
        assert all(line == line.rstrip() for line in criteria.splitlines())


def test_iter_trials_filters_archive_names_before_parsing(tmp_path):
    path = tmp_path / "trials.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trials/", "")
        archive.writestr("Contents.txt", "not XML")
        archive.writestr("trials/NCT09999999.xml", "<invalid excluded XML")
        for nct_id in ("NCT00000108", "NCT00000102"):
            archive.write(TRIALS / f"{nct_id}.xml", f"trials/{nct_id}.xml")
    rows = list(iter_trials([path], keep={"NCT00000102"}))
    assert [row["nct_id"] for row in rows] == ["NCT00000102"]


def test_iter_trials_without_filter_returns_both_fixture_trials(tmp_path):
    path = tmp_path / "trials.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trials/", "")
        archive.writestr("Contents.txt", "not XML")
        for nct_id in ("NCT00000108", "NCT00000102"):
            archive.write(TRIALS / f"{nct_id}.xml", f"trials/{nct_id}.xml")
    rows = list(iter_trials([path], keep=None))
    assert {row["nct_id"] for row in rows} == {"NCT00000102", "NCT00000108"}
    assert len(rows) == 2


def _load_builder():
    path = REPO / "scripts" / "build_trec_tables.py"
    spec = importlib.util.spec_from_file_location("build_trec_tables_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _make_raw_data(data_dir, *, empty_judgments=False):
    raw = data_dir / "raw" / "trec_ct"
    raw.mkdir(parents=True)
    topics = (FIXTURES / "topics_sample.xml").read_text(encoding="utf-8")
    topics = topics.replace('number="1"', 'number="10"')
    for year in (2021, 2022):
        (raw / f"topics{year}.xml").write_text(topics, encoding="utf-8")
    qrels = {
        2021: (
            "10 0 NCT00000108 1\n2 0 NCT00000114 2\n2 0 NCT00000102 0\n"
            "2 0 NCT09999999 1\n10 0 NCT09999998 0\n"
        ),
        2022: ("10 0 NCT00043316 2\n2 0 NCT00000102 2\n2 0 NCT00005736 0\n10 0 NCT00000113 1\n"),
    }
    for year, text in qrels.items():
        (raw / f"qrels{year}.txt").write_text("" if empty_judgments else text, encoding="utf-8")
    for part, ids in enumerate(
        [
            ["NCT00000114", "NCT00000108", "NCT00000102"],
            ["NCT00043316", "NCT00005736", "NCT00000113", "NCT00000102"],
        ],
        start=1,
    ):
        with zipfile.ZipFile(raw / f"ClinicalTrials.2021-04-27.part{part}.zip", "w") as archive:
            for nct_id in ids:
                archive.write(TRIALS / f"{nct_id}.xml", f"trials/{nct_id}.xml")
            archive.writestr("Contents.txt", "archive index")
            archive.writestr("trials/NCT08888888.xml", "<invalid unjudged XML")


def test_build_tables_writes_deterministic_records_and_complete_statistics(tmp_path):
    _make_raw_data(tmp_path)
    builder = _load_builder()
    assert builder.main(["--data-dir", str(tmp_path)]) == 0
    processed = tmp_path / "processed"
    patients = _read_jsonl(processed / "patients.jsonl")
    labels = _read_jsonl(processed / "labels.jsonl")
    trials = _read_jsonl(processed / "trials.jsonl")
    stats = json.loads((processed / "stats.json").read_text(encoding="utf-8"))

    assert [row["patient_id"] for row in patients] == ["2021-2", "2021-10", "2022-2", "2022-10"]
    label_order = [
        (row["year"], int(row["patient_id"].split("-")[1]), row["nct_id"]) for row in labels
    ]
    assert label_order == sorted(label_order)
    assert len(labels) == 9
    assert all(set(row) == {"patient_id", "year", "nct_id", "label"} for row in labels)
    nct_ids = [row["nct_id"] for row in trials]
    assert nct_ids == sorted(set(nct_ids))
    assert len(nct_ids) == 6
    assert "β-hCG" in (processed / "patients.jsonl").read_text(encoding="utf-8")
    assert "\\u03b2" not in (processed / "patients.jsonl").read_text(encoding="utf-8")
    assert stats["patients_per_year"] == {"2021": 2, "2022": 2}
    assert stats["label_counts_per_year"] == {
        "2021": {"0": 2, "1": 2, "2": 1},
        "2022": {"0": 1, "1": 1, "2": 2},
    }
    assert stats["judged_unique_trials"] == 8
    assert stats["trials_found"] == 6
    assert stats["missing_nct_ids"] == ["NCT09999998", "NCT09999999"]
    assert stats["split_status_counts"] == {
        "both": 2,
        "inclusion_only": 2,
        "exclusion_only": 0,
        "unsectioned": 1,
        "no_criteria": 1,
    }
    assert stats["sectioned_share"] == pytest.approx(4 / 8)

    first = {path.name: path.read_bytes() for path in processed.iterdir() if path.is_file()}
    assert builder.main(["--data-dir", str(tmp_path)]) == 0
    assert {path.name: path.read_bytes() for path in processed.iterdir() if path.is_file()} == first


def test_build_tables_uses_configured_data_dir_and_handles_no_judgments(tmp_path, monkeypatch):
    _make_raw_data(tmp_path, empty_judgments=True)
    monkeypatch.setenv("CLINREASON_DATA_DIR", str(tmp_path))
    builder = _load_builder()
    assert builder.main([]) == 0
    processed = tmp_path / "processed"
    stats = json.loads((processed / "stats.json").read_text(encoding="utf-8"))
    assert len(_read_jsonl(processed / "patients.jsonl")) == 4
    assert _read_jsonl(processed / "labels.jsonl") == []
    assert _read_jsonl(processed / "trials.jsonl") == []
    assert stats["label_counts_per_year"] == {
        "2021": {"0": 0, "1": 0, "2": 0},
        "2022": {"0": 0, "1": 0, "2": 0},
    }
    assert stats["judged_unique_trials"] == 0
    assert stats["trials_found"] == 0
    assert stats["missing_nct_ids"] == []
    assert stats["split_status_counts"] == dict.fromkeys(SPLIT_STATUSES, 0)
    assert stats["sectioned_share"] == 0.0


@pytest.mark.parametrize(
    ("inc_header", "exc_header"),
    [
        ("Inclusion:", "Exclusion:"),
        ("- Inclusion Criteria:", "- Exclusion Criteria:"),
        ("1. Inclusion criteria", "2. Exclusion criteria"),
        ("a. Inclusion", "b. Exclusion"),
        ("Major inclusion criteria:", "Major exclusion criteria:"),
        ("Infant Inclusion Criteria:", "Recipient Exclusion Criteria:"),
        ("Inclusion criteria - donor:", "Criteria for exclusion:"),
        ("Inclusion criteria.", "Exclusion criteria."),
    ],
)
def test_split_criteria_header_variants(inc_header, exc_header):
    text = f"{inc_header}\n- adult\n{exc_header}\n- pregnant"
    inclusion, exclusion, status = split_criteria(text)
    assert status == "both"
    assert inclusion == "- adult"
    assert exclusion == "- pregnant"


def test_split_criteria_non_inclusion_with_marker_is_exclusion():
    inclusion, exclusion, status = split_criteria(
        "- Inclusion criteria:\n- adult\n- Non inclusion criteria:\n- pregnant"
    )
    assert (inclusion, exclusion, status) == ("- adult", "- pregnant", "both")


def test_split_criteria_inline_past_tense_exclusion():
    inclusion, exclusion, status = split_criteria(
        "Adults with uveitis were eligible. Exclusion criteria included prior surgery."
    )
    assert status == "both"
    assert "uveitis" in inclusion
    assert exclusion.startswith("prior surgery")


def test_split_criteria_combined_header_is_not_a_section():
    _, _, status = split_criteria("Inclusion/exclusion criteria:\nAdults with asthma.")
    assert status == "unsectioned"


def test_split_criteria_keeps_text_of_long_header_without_colon():
    inclusion, exclusion, status = split_criteria(
        "Inclusion criteria for controls (all three) males and females"
        " of 18 years of age and older.\n- healthy\nExclusion criteria:\n- smoker"
    )
    assert status == "both"
    assert "18 years of age" in inclusion
    assert exclusion == "- smoker"
