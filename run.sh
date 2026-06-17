#!/usr/bin/env bash
#
# Run the full per-chapter pipeline:  extract -> transcribe -> analyze (--review).
#
# Usage:
#   ./run.sh "1.1 JUST A BARREL OF MONKEYS"                 # chapter label MUST be quoted
#   ./run.sh "1.1 JUST A BARREL OF MONKEYS" --review-refresh # extra args pass to analyze.py
#   FORCE=1 ./run.sh "1.1 ..."                              # re-transcribe even if a transcript exists
#
# The recording must already exist at  data/audio/<chapter label>.wav .
# analyze runs with --review (LLM denoising on your Claude subscription by default).
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

if [ "$#" -lt 1 ] || [ -z "${1:-}" ]; then
    echo "Usage: $0 \"<chapter label>\" [extra analyze flags]" >&2
    echo "       run '$PYTHON extract_chapter.py --list' to see chapter labels" >&2
    exit 2
fi

chapter="$1"; shift            # remaining args ("$@") are forwarded to analyze.py

echo "==> [1/3] extract: $chapter"
"$PYTHON" extract_chapter.py "$chapter"

transcript="data/transcripts/$chapter/$chapter.json"
if [ -f "$transcript" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "==> [2/3] transcribe: skipped — transcript exists (set FORCE=1 to redo)"
else
    echo "==> [2/3] transcribe: $chapter"
    "$PYTHON" transcribe.py "$chapter"
fi

echo "==> [3/3] analyze (--review): $chapter"
"$PYTHON" analyze.py "$chapter" --review "$@"

echo "==> done — reports in: data/reports/$chapter/"
