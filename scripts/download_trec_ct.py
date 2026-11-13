"""Download the TREC Clinical Trials 2021/2022 data (topics, qrels, trial snapshot).

Files go to <data_dir>/raw/trec_ct/. Existing files with the expected size are
skipped; downloads go to a .part file first so an interrupted run never leaves a
truncated file behind. A SHA-256 manifest is written next to the files and, with
--write-manifest, into the repository (manifests/trec_ct.sha256) for provenance.

Usage:
    python scripts/download_trec_ct.py [--data-dir DIR] [--write-manifest]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from clinreason.config import load_config

TREC = "https://www.trec-cds.org"
NIST = "https://trec.nist.gov/data/trials"

FILES: dict[str, str] = {
    "topics2021.xml": f"{TREC}/topics2021.xml",
    "topics2022.xml": f"{TREC}/topics2022.xml",
    "qrels2021.txt": f"{NIST}/qrels2021.txt",
    "qrels2022.txt": f"{NIST}/qrels2022.txt",
    **{
        f"ClinicalTrials.2021-04-27.part{i}.zip": (
            f"{TREC}/2021_data/ClinicalTrials.2021-04-27.part{i}.zip"
        )
        for i in range(1, 6)
    },
}

REPO_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "trec_ct.sha256"


def remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        length = resp.headers.get("Content-Length")
    return int(length) if length else None


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)
    tmp.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="override config data_dir")
    parser.add_argument(
        "--write-manifest", action="store_true", help="also write manifests/trec_ct.sha256"
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else Path(load_config()["data_dir"])
    out_dir = data_dir / "raw" / "trec_ct"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for name, url in FILES.items():
        dest = out_dir / name
        expected = remote_size(url)
        if dest.exists() and (expected is None or dest.stat().st_size == expected):
            print(f"skip      {name} (already present)")
        else:
            print(f"download  {name} ({(expected or 0) / 1e6:.1f} MB)", flush=True)
            download(url, dest)
            if expected is not None and dest.stat().st_size != expected:
                print(f"ERROR size mismatch for {name}", file=sys.stderr)
                return 1
        lines.append(f"{sha256(dest)}  {name}")

    manifest = "\n".join(lines) + "\n"
    (out_dir / "SHA256SUMS").write_text(manifest)
    if args.write_manifest:
        REPO_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        REPO_MANIFEST.write_text(manifest)
    print(f"done: {len(lines)} files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
