"""Parse TREC Clinical Trials topics, judgments, and archived trial records."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator
from html import unescape
from pathlib import Path
from zipfile import ZipFile

_HEADER = re.compile(
    r"^\s*(?:(?:key|main|general|patient(?:\s*/\s*subject)?|subject)\s+)?"
    r"(?P<kind>non[-\s]+inclusion|inclusion|exclusion)\s+criteria\b"
    r"(?:\s+for\s+[^:\n]+)?\s*(?::\s*(?P<body>.*))?$",
    re.IGNORECASE,
)
_INLINE_EXCLUSION = re.compile(
    r"\b(?:exclusion|non[-\s]+inclusion)\s+criteria\s+includes?\b", re.IGNORECASE
)
_AGE = re.compile(r"(\d+(?:\.\d+)?)\s+(years?|months?|weeks?|days?)", re.IGNORECASE)
_NCT_ID = re.compile(r"NCT\d{8}")


def _normalize_text(text: str | None) -> str | None:
    if text is None:
        return None
    return " ".join(unescape(text).split()) or None


def _clean_criteria(text: str | None) -> str | None:
    if text is None:
        return None
    return "\n".join(line.rstrip() for line in text.replace("\r", "").split("\n")).strip() or None


def _element_text(root: ET.Element, path: str) -> str | None:
    element = root.find(path)
    return "".join(element.itertext()) if element is not None else None


def load_topics(path: str | Path, year: int) -> list[dict]:
    """Load patient narratives, using the caller's year rather than the XML task."""
    root = ET.parse(path).getroot()
    patients = []
    for element in root.findall("topic"):
        number = int(element.attrib["number"])
        patients.append(
            {
                "patient_id": f"{year}-{number}",
                "year": year,
                "topic": number,
                "text": _normalize_text("".join(element.itertext())) or "",
            }
        )
    return patients


def load_qrels(path: str | Path, year: int) -> list[dict]:
    """Load judgments; reject invalid topics, NCT identifiers, or relevance labels."""
    labels = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            error = f"{path}:{line_number}: malformed qrels line: {line.strip()!r}"
            if len(fields) != 4:
                raise ValueError(error)
            topic, placeholder, nct_id, label_text = fields
            try:
                number = int(topic)
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(error) from exc
            if number < 1 or placeholder != "0" or _NCT_ID.fullmatch(nct_id) is None:
                raise ValueError(error)
            if label not in {0, 1, 2}:
                raise ValueError(f"{path}:{line_number}: invalid qrels label: {label}")
            labels.append(
                {
                    "patient_id": f"{year}-{number}",
                    "year": year,
                    "nct_id": nct_id,
                    "label": label,
                }
            )
    return labels


def split_criteria(text: str | None) -> tuple[str | None, str | None, str]:
    """Split explicit eligibility sections while retaining the order of repeated blocks.

    Preamble supplies inclusion only when no inclusion header occurs. Unsectioned
    text and header-only text have no populated sections.
    """
    text = _clean_criteria(text)
    if text is None:
        return None, None, "no_criteria"

    blocks: dict[str, list[str]] = {"inclusion": [], "exclusion": []}
    preamble: list[str] = []
    buffer: list[str] = []
    current: str | None = None
    has_header = False
    has_inclusion_header = False

    def flush() -> None:
        block = _clean_criteria("\n".join(buffer))
        if block is not None:
            if current is None:
                preamble.append(block)
            else:
                blocks[current].append(block)
        buffer.clear()

    for line in text.split("\n"):
        header = _HEADER.match(line)
        if header is not None:
            flush()
            kind = header.group("kind").lower()
            current = "inclusion" if kind == "inclusion" else "exclusion"
            has_header = True
            has_inclusion_header |= current == "inclusion"
            line = header.group("body") or ""

        inline = _INLINE_EXCLUSION.search(line)
        if inline is not None:
            buffer.append(line[: inline.start()].rstrip())
            flush()
            current = "exclusion"
            has_header = True
            line = line[inline.end() :].lstrip(" \t:")
        buffer.append(line)
    flush()

    if not has_header:
        return None, None, "unsectioned"
    if not has_inclusion_header:
        blocks["inclusion"] = preamble + blocks["inclusion"]
    inclusion = "\n\n".join(blocks["inclusion"]) or None
    exclusion = "\n\n".join(blocks["exclusion"]) or None
    if inclusion is not None and exclusion is not None:
        status = "both"
    elif inclusion is not None:
        status = "inclusion_only"
    elif exclusion is not None:
        status = "exclusion_only"
    else:
        status = "unsectioned"
    return inclusion, exclusion, status


def parse_age_years(value: str | None) -> float | None:
    """Convert supported age units to years; unknown or unavailable ages are None."""
    if value is None:
        return None
    match = _AGE.fullmatch(value.strip())
    if match is None:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower().rstrip("s")
    return amount / {"year": 1, "month": 12, "week": 52, "day": 365}[unit]


def parse_trial_xml(data: bytes) -> dict:
    """Parse a ClinicalTrials.gov snapshot record without flattening criteria lines."""
    root = ET.fromstring(data)

    def field(path: str) -> str | None:
        return _normalize_text(_element_text(root, path))

    raw_criteria = _element_text(root, "eligibility/criteria/textblock")
    criteria = _clean_criteria(unescape(raw_criteria) if raw_criteria is not None else None)
    inclusion, exclusion, status = split_criteria(criteria)
    conditions = []
    for element in root.findall("condition"):
        condition = _normalize_text("".join(element.itertext()))
        if condition is not None:
            conditions.append(condition)
    return {
        "nct_id": field("id_info/nct_id"),
        "brief_title": field("brief_title"),
        "official_title": field("official_title"),
        "conditions": conditions,
        "brief_summary": field("brief_summary/textblock"),
        "overall_status": field("overall_status"),
        "phase": field("phase"),
        "study_type": field("study_type"),
        "gender": field("eligibility/gender"),
        "minimum_age_years": parse_age_years(field("eligibility/minimum_age")),
        "maximum_age_years": parse_age_years(field("eligibility/maximum_age")),
        "healthy_volunteers": {"Accepts Healthy Volunteers": True, "No": False}.get(
            field("eligibility/healthy_volunteers")
        ),
        "criteria_text": criteria,
        "inclusion_text": inclusion,
        "exclusion_text": exclusion,
        "split_status": status,
    }


def iter_trials(
    zip_paths: Iterable[str | Path], keep: set[str] | None = None
) -> Iterator[dict]:
    """Yield archived trial XMLs, applying the optional NCT filename filter first."""
    for path in zip_paths:
        with ZipFile(path) as archive:
            for member in archive.infolist():
                filename = Path(member.filename)
                if member.is_dir() or filename.suffix.lower() != ".xml":
                    continue
                if keep is not None and filename.stem not in keep:
                    continue
                yield parse_trial_xml(archive.read(member))
