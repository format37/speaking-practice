"""Extract a chapter's reference text from the active book.

The source is auto-detected from the active book folder:

* **epub** (a ``*.epub`` is present): read the table of contents to map a
  human-readable chapter label (e.g. "1.1 JUST A BARREL OF MONKEYS") to its
  XHTML document and extract that chapter's prose.
* **txt** (no epub, a top-level ``*.txt``): treat the whole file as a single
  "chapter" — for short texts read in one pass. The label is the txt's stem.

Either way the chapter is written to the active book's chapters/<label>.txt
with the title as the first line followed by the body. analyze.py's
load_reference() treats the first non-empty line as the title (dropped from
scoring) and the rest as body; for txt the synthetic title line is the label,
so the entire text is kept as scored body.

Usage:
    python extract_chapter.py "1.1 JUST A BARREL OF MONKEYS"   # epub: by label
    python extract_chapter.py                                  # txt: whole file
    python extract_chapter.py --list        # print every chapter label
    python extract_chapter.py --all         # extract every chapter
    # --student/--book override the active ./use selection for one run.
"""
import argparse
import sys
from pathlib import Path

import config

import ebooklib

from ebooklib import epub
from bs4 import BeautifulSoup


def build_label_index(book):
    """Walk the epub TOC and return an ordered list of (label, href) pairs."""
    pairs = []

    def walk(toc):
        for item in toc:
            if isinstance(item, tuple):
                section, children = item
                title = getattr(section, "title", None)
                href = getattr(section, "href", None)
                if title and href:
                    pairs.append((title.strip(), href))
                walk(children)
            elif isinstance(item, list):
                walk(item)
            else:
                title = getattr(item, "title", None)
                href = getattr(item, "href", None)
                if title and href:
                    pairs.append((title.strip(), href))

    walk(book.toc)
    return pairs


def extract_body(book, href):
    """Load the XHTML document for href and return (title, body_text)."""
    file_href = href.split("#")[0]
    item = book.get_item_with_href(file_href)
    if item is None:
        # Fall back to a suffix match (epub hrefs are sometimes relative).
        for doc in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            if doc.get_name().endswith(file_href):
                item = doc
                break
    if item is None:
        sys.exit(f"ERROR: could not locate document for href: {href}")

    soup = BeautifulSoup(item.get_content(), "html.parser")

    # Title: first heading element if present, else the first paragraph.
    title = ""
    heading = soup.find(["h1", "h2", "h3", "h4", "h5", "h6"])
    if heading is not None:
        title = heading.get_text(" ", strip=True)

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text:
            paragraphs.append(text)

    body = "\n\n".join(paragraphs)
    return title, body


def write_chapter(label, title, body):
    """Write the chapter text to the active book's chapters/<label>.txt."""
    config.ensure_dirs()
    out = config.chapter_txt(label)
    header = title.strip() or label
    text = header + "\n\n" + body.strip() + "\n"
    out.write_text(text, encoding="utf-8")
    return out


def resolve_label(pairs, label):
    """Find the href for a label (exact match, then case-insensitive)."""
    for lbl, href in pairs:
        if lbl == label:
            return href
    low = label.lower()
    for lbl, href in pairs:
        if lbl.lower() == low:
            return href
    return None


def extract_txt(txt_path):
    """Whole txt as a single chapter; returns (label, output_path).

    The label is the txt's stem (it must match the audio file's stem). The body
    is the entire file, with the label written as a synthetic title line so
    analyze.load_reference() keeps every line as scored body.
    """
    label = txt_path.stem
    body = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if not body:
        sys.exit(f"ERROR: txt file is empty: {txt_path}")
    out = write_chapter(label, label, body)   # title == label -> dropped, body kept
    return label, out


def run_txt_mode(args, txts):
    """Handle a txt-source book (--list / --all / single)."""
    if args.list:
        for t in txts:
            print(t.stem)
        return
    if args.all:
        for t in txts:
            label, out = extract_txt(t)
            print(f"wrote {out}  ({len(out.read_text(encoding='utf-8'))} chars)")
        return

    # Single text. With one txt the label is its stem regardless of any arg
    # (the label must match the audio stem). With several, an arg selects one.
    if len(txts) == 1:
        chosen = txts[0]
        if args.label and args.label != chosen.stem:
            print(f"note: book has a single text; using {chosen.stem!r} "
                  f"(ignoring label {args.label!r})")
    else:
        chosen = next((t for t in txts if t.stem == args.label), None)
        if chosen is None:
            avail = ", ".join(t.stem for t in txts)
            sys.exit(f"ERROR: multiple texts; pass one of: {avail}"
                     if not args.label else
                     f"ERROR: no text named {args.label!r}; available: {avail}")

    label, out = extract_txt(chosen)
    print(f"Extracted {label!r} (whole text)")
    print(f"  chars: {len(out.read_text(encoding='utf-8'))}")
    print(f"  wrote: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract chapter reference text from the book epub."
    )
    parser.add_argument("label", nargs="?", help="Chapter label (TOC entry).")
    parser.add_argument("--list", action="store_true",
                        help="Print all chapter labels and exit.")
    parser.add_argument("--all", action="store_true",
                        help="Extract every chapter in the TOC.")
    parser.add_argument("--epub", help="Override the source epub path.")
    parser.add_argument("--student", help="Override the active student (./use).")
    parser.add_argument("--book", help="Override the active book (./use).")
    args = parser.parse_args()

    # An explicit --epub lets `--list` run without an active book; everything
    # that writes into the book folder needs the active (student, book) context.
    config.activate(args.student, args.book, require=args.epub is None)

    # Pick the source: an explicit --epub, else the active book's *.epub, else
    # its top-level *.txt (whole-text mode). epub wins when both are present.
    if args.epub:
        epub_path = Path(args.epub)
    else:
        epubs, txts = config.book_sources()
        if not epubs and txts:
            run_txt_mode(args, txts)
            return
        epub_path = config.DEFAULT_EPUB
        if epub_path is None or not epub_path.exists():
            sys.exit(f"ERROR: no .epub or .txt found in the active book folder: "
                     f"{config.BOOK_DIR}\n  add a .epub (chapter mode) or a "
                     f".txt (whole-text mode).")

    if not epub_path.exists():
        sys.exit(f"ERROR: epub not found: {epub_path}")

    book = epub.read_epub(str(epub_path))
    pairs = build_label_index(book)

    if args.list:
        for label, _ in pairs:
            print(label)
        return

    if args.all:
        for label, href in pairs:
            title, body = extract_body(book, href)
            if not body.strip():
                print(f"SKIP (no prose): {label}")
                continue
            out = write_chapter(label, title, body)
            print(f"wrote {out}  ({len(body.split())} words)")
        return

    if not args.label:
        sys.exit("ERROR: provide a chapter label, or use --list / --all.\n"
                 "Run with --list to see available labels.")

    href = resolve_label(pairs, args.label)
    if href is None:
        sys.exit(f"ERROR: chapter label not found in TOC: {args.label!r}\n"
                 f"Run `python extract_chapter.py --list` to see valid labels.")

    title, body = extract_body(book, href)
    if not body.strip():
        sys.exit(f"ERROR: no prose body found for chapter: {args.label!r}")

    out = write_chapter(args.label, title, body)
    print(f"Extracted {args.label!r}")
    print(f"  title: {title!r}")
    print(f"  words: {len(body.split())}")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
