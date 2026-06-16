# Speaking Practice

Daily reading-aloud practice for English (ML-interview prep): read a book chapter
aloud, transcribe it, and compare it against the source text to surface the
words/sounds to drill and track progress over time. The analysis is
language-pluggable (English by default; also `generic` and `ja`/`zh`).

## Quick start

One session = one chapter. Record yourself reading it and save the audio as
`data/audio/<chapter label>.wav`, using the **exact** table-of-contents label —
the label is the join key across every step
(e.g. `data/audio/1.1 JUST A BARREL OF MONKEYS.wav`). Then, from the project root:

```bash
python extract_chapter.py "1.1 JUST A BARREL OF MONKEYS"            # reference text from the epub (--list shows all labels)
python transcribe.py      "1.1 JUST A BARREL OF MONKEYS"            # Deepgram transcription
python analyze.py         "1.1 JUST A BARREL OF MONKEYS" --review   # score the reading + LLM denoising
```

Then review `data/reports/<label>/` — start with **`focus_words.csv`** (ranked
drill list), **`ending_changes.csv`** and **`confusions.csv`** — and
`data/reports/_progress/` for trends across sessions.

`--review` is safe to re-run: it **resumes** (only un-judged errors are re-sent),
so if a run doesn't finish in one pass, just run it again until coverage is full.

## Setup

```bash
pip install -r requirements.txt
sudo apt install ffmpeg          # system dependency (ffmpeg + ffprobe); or your OS package manager
cp .env.example .env             # then add your Deepgram API key (DEEPGRAM_API_KEY)
```

`--review` uses your local **Claude subscription** by default (via the Claude
Agent SDK / authenticated `claude` CLI) — **no API key and no cost**. To use the
paid OpenAI API instead, set `REVIEW_BACKEND=openai` and `OPENAI_KEY`. Without
`--review` (or if the backend is unavailable) the analysis still runs and
denoises using a free, offline name-gazetteer — it never crashes.

## What it measures

Every deviation between your reading and the book is classified — omission,
repetition, insertion, ending-mixup, mispronunciation, substitution — and
**measurement noise is kept out of the score** so chapters stay comparable:

- Invented book names (Sering, Brin, Avrana, …) the recognizer can't know are
  excluded from the accuracy/WER denominator (listed separately in `names.csv`),
  never treated as drill targets. Reports show both **raw** and **denoised** scores.
- With `--review`, an LLM judges each remaining error *with its context* and marks
  it keep/exclude plus a **cause and reason** (`errors_reviewed.csv`).
- The headline signals are **ending changes** (dropped `-s`/`-ed`/possessive, e.g.
  `screens`→`screen`) and **confident confusions** (real-word swaps the ASR clearly
  heard) — the things that matter most for interviews.

Per chapter (`data/reports/<label>/`): `focus_words`, `ending_changes`,
`confusions`, `errors` (+ `errors_reviewed` with `--review`), `names`, `summary`,
`wpm_timeline`, `pauses`, and matching PNGs. Cumulative (`data/reports/_progress/`):
`sessions.csv` plus charts of raw + denoised accuracy/WER, speech rate, and the
words that recur most in your focus list.

## Languages

`transcribe.py` and `analyze.py` take `--language/-l` (default from
`PRACTICE_LANGUAGE`, else `en`). Profiles live in `languages.py`: `en` (full
English, rate in WPM), `generic` (any space-separated language; also the fallback
for unknown codes), and `ja`/`zh` (character-level, rate in CPM). `sessions.csv`
is keyed by `(chapter, language)`, so the same chapter in two languages stays
separate. To add a language, add a `LanguageProfile` and register it in `PROFILES`.

## Repo layout

`data/` is **gitignored** — recordings, transcripts, and reports stay local; the
committed repository is code-only.
