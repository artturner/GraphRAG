#!/usr/bin/env python3
"""Extract per-page text from textbook PDF chapters into a JSON dataset.

Each chapter is expected to be a separate PDF file. Chapter numbers are
inferred from filenames (e.g. chapter_03.pdf, ch3.pdf, 03_intro.pdf) or
supplied via --chapter / the manifest.

Page numbers printed in the PDF are detected with a three-tier fallback:
  1. PDF page-label metadata (most accurate when present)
  2. Scan of the first/last lines of each page for a standalone integer
  3. Sequential count, starting from --first-page (default 1)

Usage:
    # Single chapter
    python scripts/pdf_extract.py data/pdfs/chapter_03.pdf -o data/ch3_pages.json

    # Single chapter with explicit overrides
    python scripts/pdf_extract.py data/pdfs/intro.pdf --chapter 1 --first-page 47 -o intro.json

    # Whole book via manifest (recommended when --first-page differs per chapter)
    python scripts/pdf_extract.py --manifest data/pdfs/manifest.json -o data/full_book.json

Manifest format (JSON array, paths relative to the manifest file):
    [
      { "file": "chapter_01.pdf", "chapter": 1, "first_page": 1  },
      { "file": "chapter_02.pdf", "chapter": 2, "first_page": 23 },
      { "file": "chapter_03.pdf", "chapter": 3, "first_page": 47 }
    ]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pypdf


# ---------------------------------------------------------------------------
# Chapter detection
# ---------------------------------------------------------------------------

def _detect_chapter(path: Path, override: int | None) -> int | None:
    if override is not None:
        return override
    # chapter_03, chapter-3, chapter 3
    m = re.search(r'ch(?:apter)?[_\s-]*(\d+)', path.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Leading number: 03_federalism, 3-intro
    m = re.match(r'^(\d+)', path.stem)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Page number detection
# ---------------------------------------------------------------------------

def _page_number_from_label(label: str | None) -> int | None:
    if label is None:
        return None
    try:
        n = int(label)
        return n if 1 <= n <= 9999 else None
    except ValueError:
        return None  # roman numerals, letter prefixes, etc.


def _page_number_from_text(text: str) -> int | None:
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    # Check the first 3 and last 3 non-empty lines for a bare integer
    candidates = lines[:3] + lines[-3:]
    for line in candidates:
        if re.fullmatch(r'\d{1,4}', line):
            n = int(line)
            if 1 <= n <= 9999:
                return n
    return None


# ---------------------------------------------------------------------------
# Per-PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(
    pdf_path: Path,
    chapter: int | None,
    first_page: int | None,
) -> list[dict]:
    reader = pypdf.PdfReader(str(pdf_path))

    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []

    pages = []
    sequential = first_page if first_page is not None else 1

    for i, pdf_page in enumerate(reader.pages):
        text = (pdf_page.extract_text() or "").strip()
        if not text:
            sequential += 1
            continue

        if first_page is not None:
            # first_page explicitly set — trust sequential, skip detection
            page_num = sequential
        else:
            label = labels[i] if i < len(labels) else None
            page_num = (
                _page_number_from_label(label)
                or _page_number_from_text(text)
                or sequential
            )

        sequential = page_num + 1

        pages.append({
            "chapter": chapter,
            "page_number": page_num,
            "source": pdf_path.name,
            "text": text,
        })

    return pages


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _collect_pdfs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            paths.append(p)
        else:
            print(f"Warning: skipping {p} (not a PDF)", file=sys.stderr)
    return paths


def _load_manifest(manifest_path: str) -> list[dict]:
    """Load and validate a manifest JSON file.

    Each entry must have a "file" key. "chapter" and "first_page" are optional.
    Relative file paths are resolved against the manifest's directory.
    """
    path = Path(manifest_path)
    base_dir = path.parent

    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading manifest: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(entries, list):
        print("Error: manifest must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    resolved = []
    for i, entry in enumerate(entries):
        if "file" not in entry:
            print(f"Error: manifest entry {i} is missing required 'file' key.", file=sys.stderr)
            sys.exit(1)
        resolved.append({
            "path": base_dir / entry["file"],
            "chapter": entry.get("chapter"),        # None → filename detection
            "first_page": entry.get("first_page"),  # None → label/text detection
        })

    return resolved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-page text from textbook PDFs into a JSON dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs", nargs="*", metavar="PDF_OR_DIR",
        help="PDF file(s) or a directory of PDFs (omit when using --manifest)",
    )
    parser.add_argument(
        "--manifest", metavar="FILE",
        help="JSON manifest file mapping each chapter PDF to its first printed page",
    )
    parser.add_argument(
        "-o", "--output", default="dataset.json",
        help="Output JSON file (default: dataset.json)",
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help="Override chapter number. Only applied when a single PDF is given.",
    )
    parser.add_argument(
        "--first-page", type=int, default=None,
        help="Printed page number of the first PDF page. When set, forces sequential "
             "counting from this number (skips label/text detection). "
             "Only applied when a single PDF is given.",
    )
    args = parser.parse_args()

    if not args.manifest and not args.inputs:
        parser.error("Provide PDF file(s)/directory or --manifest.")

    # Build a unified work list: [(pdf_path, chapter_override, first_page)]
    work: list[tuple[Path, int | None, int | None]] = []

    if args.manifest:
        for entry in _load_manifest(args.manifest):
            work.append((entry["path"], entry["chapter"], entry["first_page"]))
    else:
        pdf_paths = _collect_pdfs(args.inputs)
        if not pdf_paths:
            print("Error: no PDF files found.", file=sys.stderr)
            sys.exit(1)
        single = len(pdf_paths) == 1
        for pdf_path in pdf_paths:
            chapter = args.chapter if single else None
            first_page = args.first_page if single else 1
            work.append((pdf_path, chapter, first_page))

    all_pages: list[dict] = []

    for pdf_path, chapter_override, first_page in work:
        chapter = _detect_chapter(pdf_path, chapter_override)
        print(f"Processing {pdf_path.name}  chapter={chapter}  first_page={first_page} ...", file=sys.stderr)
        pages = extract_pdf(pdf_path, chapter, first_page)
        print(f"  {len(pages)} pages", file=sys.stderr)
        all_pages.extend(pages)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_pages, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(all_pages)} pages → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
