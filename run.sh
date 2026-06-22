#!/usr/bin/env bash
#
# Run the full per-chapter pipeline:  extract -> transcribe -> analyze (--review).
#
# It targets the active student/book selected with  ./use <student> <book>  —
# run  ./use  to see the current selection and the roster.
#
# Usage:
#   ./run.sh "1.1 JUST A BARREL OF MONKEYS"                 # chapter label MUST be quoted
#   ./run.sh                                                # whole-text (txt) book: no label needed
#   ./run.sh "1.1 JUST A BARREL OF MONKEYS" --review-refresh # extra args pass to analyze.py
#   FORCE=1 ./run.sh "1.1 ..."                              # re-transcribe even if a transcript exists
#
# The recording must already exist at  <active book>/audio/<chapter label>.wav .
# analyze runs with --review (LLM denoising on your Claude subscription by default).
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

# Resolve the active context once; config.py is the single source of paths.
active="$("$PYTHON" config.py active)"   # tab-separated "student<TAB>book"
echo "==> student/book: ${active/$'\t'/ / }"

# The chapter label is the first non-flag argument. With none — e.g. a short
# whole-text (txt) book read in one pass — auto-resolve the single label.
if [ "$#" -ge 1 ] && [ -n "${1:-}" ] && [ "${1#-}" = "$1" ]; then
    chapter="$1"; shift          # remaining args ("$@") are forwarded to analyze.py
else
    chapter="$("$PYTHON" config.py default-label)"
    if [ -z "$chapter" ]; then
        echo "ERROR: no chapter label given, and the active book is not a single-text book." >&2
        echo "       Pass a label (see '$PYTHON extract_chapter.py --list'), or ./use a txt book." >&2
        exit 2
    fi
    echo "==> auto-resolved chapter: $chapter"
fi

echo "==> [1/3] extract: $chapter"
"$PYTHON" extract_chapter.py "$chapter"

transcript="$("$PYTHON" config.py path transcript-json "$chapter")"
if [ -f "$transcript" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "==> [2/3] transcribe: skipped — transcript exists (set FORCE=1 to redo)"
else
    echo "==> [2/3] transcribe: $chapter"
    "$PYTHON" transcribe.py "$chapter"
fi

echo "==> [3/3] analyze (--review): $chapter"
"$PYTHON" analyze.py "$chapter" --review "$@"

echo "==> done — reports in: $("$PYTHON" config.py path report-dir "$chapter")/"
