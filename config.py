"""Shared paths and per-student/-book configuration for the speaking toolkit.

Data layout (everything under ``data/``, which is gitignored)::

    data/
      .active                          # YAML: the active {student, book} selection
      <student>/
        config.yaml                    # at least `language:`  (e.g. en / ja / ru)
        <book>/
          <book>.epub                  # source epub (any *.epub in the folder)
          chapters/<label>.txt         # extracted reference text
          audio/<label>.wav            # recorded readings
          transcripts/<label>/<label>.{json,txt}
          reports/<label>/...          # per-chapter csv + png
          reports/_progress/           # cumulative output (sessions.csv + png)

One *student* speaks one *language* (from their ``config.yaml``); each student
reads one or more *books*. A book directory holds the same per-chapter sub-tree
the toolkit always produced — it is just scoped under ``<student>/<book>/`` now.

Call :func:`activate` once at startup (the CLIs do this in ``main()``); it
resolves the active selection — explicit args win, else ``data/.active`` — and
populates the module-level path globals below. The ``<thing>(label)`` helpers
and the path constants are only valid *after* ``activate()`` has run.
"""
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

try:
    import yaml
except ImportError:  # pragma: no cover - surfaced lazily with a clear message
    yaml = None

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
ACTIVE_FILE = DATA / ".active"          # YAML pointer set by ./use (set_active)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# Final fallback language when neither --language nor a student config sets one.
DEFAULT_LANGUAGE = os.getenv("PRACTICE_LANGUAGE", "en")

# LLM error-review backend (opt-in 'analyze.py --review' denoising layer):
#   "claude" (default) -> Claude Agent SDK via your local Claude subscription;
#                         no API key and no per-token billing.
#   "openai"           -> OpenAI Chat Completions; needs OPENAI_KEY (paid API).
REVIEW_BACKEND = os.getenv("REVIEW_BACKEND", "claude")
# Optional model override for the Claude backend (default: the SDK's model).
CLAUDE_REVIEW_MODEL = os.getenv("CLAUDE_REVIEW_MODEL") or None
# OpenAI (only needed when REVIEW_BACKEND=openai).
OPENAI_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

# --------------------------------------------------------------------------- #
# Active selection + populated path globals (set by activate())
# --------------------------------------------------------------------------- #
STUDENT = None
BOOK = None
LANGUAGE = None
BOOK_DIR = None
CHAPTERS_DIR = None       # extracted reference text (.txt)
AUDIO_DIR = None          # recorded readings (.wav)
TRANSCRIPTS_DIR = None    # per-chapter Deepgram output (json + txt)
REPORTS_DIR = None        # per-chapter csv + png
PROGRESS_DIR = None       # cumulative cross-session output
SESSIONS_CSV = None
DEFAULT_EPUB = None


# --------------------------------------------------------------------------- #
# Student / book discovery
# --------------------------------------------------------------------------- #
def student_dir(student):           return DATA / student
def student_config_path(student):   return student_dir(student) / "config.yaml"
def book_dir(student, book):        return student_dir(student) / book


def load_student_config(student):
    """Parse ``data/<student>/config.yaml`` into a dict ({} if absent)."""
    p = student_config_path(student)
    if not p.exists():
        return {}
    if yaml is None:
        sys.exit("ERROR: PyYAML is required to read student config.yaml "
                 "(pip install pyyaml).")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001 - report the offending file
        sys.exit(f"ERROR: could not parse {p}: {e}")
    return data if isinstance(data, dict) else {}


def list_students():
    """Student names = direct child dirs of data/ that have a config.yaml."""
    if not DATA.exists():
        return []
    return sorted(d.name for d in DATA.iterdir()
                  if d.is_dir() and not d.name.startswith(".")
                  and (d / "config.yaml").exists())


def list_books(student):
    """Book folders = direct child dirs of data/<student>/."""
    sd = student_dir(student)
    if not sd.exists():
        return []
    return sorted(d.name for d in sd.iterdir()
                  if d.is_dir() and not d.name.startswith("."))


# --------------------------------------------------------------------------- #
# Active pointer (data/.active)
# --------------------------------------------------------------------------- #
def read_active():
    """Return ``(student, book)`` from data/.active, or ``(None, None)``."""
    if not ACTIVE_FILE.exists():
        return None, None
    text = ACTIVE_FILE.read_text(encoding="utf-8")
    if yaml is not None:
        try:
            data = yaml.safe_load(text) or {}
        except Exception:
            data = {}
        if isinstance(data, dict) and ("student" in data or "book" in data):
            return data.get("student"), data.get("book")
    # Tolerate a plain two-line "student\nbook" file as a fallback.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[0] if lines else None), (lines[1] if len(lines) > 1 else None)


def write_active(student, book):
    DATA.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        ACTIVE_FILE.write_text(
            yaml.safe_dump({"student": student, "book": book},
                           sort_keys=False, allow_unicode=True),
            encoding="utf-8")
    else:
        ACTIVE_FILE.write_text(f"{student}\n{book or ''}\n", encoding="utf-8")


def set_active(student, book=None, create_book=True):
    """Validate and persist the active (student, book); used by ``./use``.

    Auto-picks the book when the student has exactly one. Creates an empty book
    folder when ``book`` is new (so you can drop an epub in and start).
    Returns ``(student, book, created)``.
    """
    if student not in list_students():
        avail = ", ".join(list_students()) or "(none)"
        sys.exit(f"ERROR: unknown student {student!r}. Available: {avail}\n"
                 f"Add one by creating data/{student}/config.yaml "
                 f"(with a `language:` line).")
    books = list_books(student)
    if book is None:
        if len(books) == 1:
            book = books[0]
        elif not books:
            sys.exit(f"ERROR: student {student!r} has no books yet. Start one "
                     f"with  ./use {student} <book-folder>  then drop its .epub "
                     f"into data/{student}/<book-folder>/.")
        else:
            joined = "\n  ".join(books)
            sys.exit(f"ERROR: student {student!r} has multiple books; pick one:"
                     f"\n  {joined}")
    bdir = book_dir(student, book)
    created = False
    if not bdir.exists():
        if not create_book:
            sys.exit(f"ERROR: no such book {book!r} for student {student!r}.")
        bdir.mkdir(parents=True, exist_ok=True)
        # Seed audio/ so recordings have a home before the pipeline first runs
        # (the rest of the sub-tree is created on demand by ensure_dirs()).
        (bdir / "audio").mkdir(exist_ok=True)
        created = True
    write_active(student, book)
    return student, book, created


# --------------------------------------------------------------------------- #
# Activation: resolve (student, book) -> path globals
# --------------------------------------------------------------------------- #
def _find_epub(bdir, book):
    epubs = sorted(bdir.glob("*.epub"))
    return epubs[0] if epubs else (bdir / f"{book}.epub")


def activate(student=None, book=None, *, require=True):
    """Resolve the active (student, book) and populate the path globals.

    Explicit ``student``/``book`` win; otherwise fall back to ``data/.active``.
    ``LANGUAGE`` is read from the student's ``config.yaml`` (``language:``),
    falling back to ``PRACTICE_LANGUAGE``. With ``require=True`` (default),
    exits with a helpful message when no student/book can be resolved, or when
    the resolved selection (from args or a stale ``data/.active``) doesn't exist.
    """
    global STUDENT, BOOK, LANGUAGE, BOOK_DIR, CHAPTERS_DIR, AUDIO_DIR
    global TRANSCRIPTS_DIR, REPORTS_DIR, PROGRESS_DIR, SESSIONS_CSV, DEFAULT_EPUB

    a_student, a_book = read_active()
    student = student or a_student
    book = book or a_book

    if require and not student:
        sys.exit("ERROR: no active student. Run  ./use <student> <book>  "
                 "(or pass --student/--book). Run  ./use  to list students.")
    if require and not book:
        sys.exit(f"ERROR: no active book for student {student!r}. Run "
                 f"./use {student} <book>  (or pass --book).")
    # Validate the resolved selection so a typo'd --student/--book or a stale
    # .active fails here with the same guidance ./use gives — not later with a
    # confusing phantom path.
    if require and student and student not in list_students():
        avail = ", ".join(list_students()) or "(none)"
        sys.exit(f"ERROR: unknown student {student!r}. Available: {avail}\n"
                 f"Add one by creating data/{student}/config.yaml "
                 f"(with a `language:` line), or run  ./use  to list students.")
    if require and student and book and not book_dir(student, book).exists():
        books = ", ".join(list_books(student)) or "(none)"
        sys.exit(f"ERROR: no such book {book!r} for student {student!r}. "
                 f"Books: {books}\nStart it with  ./use {student} \"{book}\"  "
                 f"(creates the folder), then add its .epub.")

    STUDENT, BOOK = student, book
    LANGUAGE = (load_student_config(student).get("language") if student
                else None) or DEFAULT_LANGUAGE

    if student and book:
        BOOK_DIR = book_dir(student, book)
        CHAPTERS_DIR = BOOK_DIR / "chapters"
        AUDIO_DIR = BOOK_DIR / "audio"
        TRANSCRIPTS_DIR = BOOK_DIR / "transcripts"
        REPORTS_DIR = BOOK_DIR / "reports"
        PROGRESS_DIR = REPORTS_DIR / "_progress"
        SESSIONS_CSV = PROGRESS_DIR / "sessions.csv"
        DEFAULT_EPUB = _find_epub(BOOK_DIR, book)
    return STUDENT, BOOK


def _require_active():
    if BOOK_DIR is None:
        sys.exit("ERROR: no active book context — call config.activate() first "
                 "(or run ./use <student> <book>).")


def ensure_dirs():
    _require_active()
    for d in (CHAPTERS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, REPORTS_DIR, PROGRESS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def chapter_txt(label):     _require_active(); return CHAPTERS_DIR / f"{label}.txt"
def audio_path(label):      _require_active(); return AUDIO_DIR / f"{label}.wav"
def transcript_dir(label):  _require_active(); return TRANSCRIPTS_DIR / label
def transcript_json(label): return transcript_dir(label) / f"{label}.json"
def report_dir(label):      _require_active(); return REPORTS_DIR / label


# --------------------------------------------------------------------------- #
# Tiny CLI: backs ./use and lets run.sh resolve paths (paths come from here).
# --------------------------------------------------------------------------- #
def _print_status():
    s, b = read_active()
    print(f"active: student={s or '(none)'}   book={b or '(none)'}")
    students = list_students()
    if not students:
        print("\nNo students yet. Create data/<name>/config.yaml "
              "(with a `language:` line).")
        return
    print("\nstudents:")
    for st in students:
        lang = load_student_config(st).get("language", "?")
        smark = "*" if st == s else " "
        print(f" {smark} {st}  [language={lang}]")
        books = list_books(st)
        if not books:
            print("      (no books yet)")
        for bk in books:
            bmark = "*" if (st == s and bk == b) else " "
            print(f"    {bmark} {bk}")


_PATH_CONSTS = {
    "book-dir": "BOOK_DIR", "chapters-dir": "CHAPTERS_DIR",
    "audio-dir": "AUDIO_DIR", "transcripts-dir": "TRANSCRIPTS_DIR",
    "reports-dir": "REPORTS_DIR", "progress-dir": "PROGRESS_DIR",
    "sessions-csv": "SESSIONS_CSV", "epub": "DEFAULT_EPUB",
}
_PATH_FNS = {
    "chapter-txt": "chapter_txt", "audio": "audio_path",
    "transcript-dir": "transcript_dir", "transcript-json": "transcript_json",
    "report-dir": "report_dir",
}


def _cli(argv):
    import argparse
    p = argparse.ArgumentParser(
        prog="config.py", description="speaking toolkit data-layout helper")
    sub = p.add_subparsers(dest="cmd")
    u = sub.add_parser("use", help="set the active student/book (or show status)")
    u.add_argument("student", nargs="?")
    u.add_argument("book", nargs="?")
    sub.add_parser("status", help="show the active context and the roster")
    a = sub.add_parser("active", help="print the active student/book (machine-readable)")
    a.add_argument("--field", choices=["student", "book", "language"])
    pa = sub.add_parser("path", help="print a resolved data path for the active context")
    pa.add_argument("kind")
    pa.add_argument("label", nargs="?")
    args = p.parse_args(argv)

    if args.cmd in (None, "status") or (args.cmd == "use" and not args.student):
        _print_status()
        return 0

    if args.cmd == "use":
        student, book, created = set_active(args.student, args.book)
        lang = load_student_config(student).get("language", DEFAULT_LANGUAGE)
        print(f"active: {student} / {book}   [language={lang}]")
        if created:
            print(f"  created empty book folder — drop the .epub into "
                  f"{book_dir(student, book)}/")
        return 0

    if args.cmd == "active":
        s, b = read_active()
        if args.field == "student":
            print(s or "")
        elif args.field == "book":
            print(b or "")
        elif args.field == "language":
            print((load_student_config(s).get("language") if s else None)
                  or DEFAULT_LANGUAGE)
        else:
            print(f"{s or ''}\t{b or ''}")
        return 0

    if args.cmd == "path":
        activate()
        if args.kind in _PATH_CONSTS:
            print(globals()[_PATH_CONSTS[args.kind]])
            return 0
        if args.kind in _PATH_FNS:
            if args.label is None:
                sys.exit(f"ERROR: path {args.kind} needs a <label>.")
            print(globals()[_PATH_FNS[args.kind]](args.label))
            return 0
        sys.exit(f"ERROR: unknown path kind {args.kind!r}. Known: "
                 + ", ".join(list(_PATH_CONSTS) + list(_PATH_FNS)))

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
