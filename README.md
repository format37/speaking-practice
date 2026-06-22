# Speaking Practice

Daily reading-aloud practice: read a book chapter aloud, transcribe it, and
compare it against the source text to surface the words/sounds to drill and
track progress over time. Multiple **students** are supported, each with their
own language and books. The analysis is language-pluggable (full English profile
by default; also `generic`, `ja`/`zh`, and any Deepgram code such as `ru`).

## Students & books

Data is organized as `data/<student>/<book>/`. Each student has a
`data/<student>/config.yaml` that sets at least their `language`:

```
data/
  alex/      config.yaml (language: en)   Children-of-Time-Adrian-Tchaikovsky/…
  irina/     config.yaml (language: ja)   (no books yet)
  veronika/  config.yaml (language: ru)   (no books yet)
  mila/      config.yaml (language: ru)   (no books yet)
```

Pick who you're working with using `./use` — it stores the active selection in
`data/.active`, which every command then targets:

```bash
./use                                   # show the active context + the roster
./use alex Children-of-Time-Adrian-Tchaikovsky   # switch student + book
./use irina "<book-folder>"             # creates the folder if new; then drop the epub in
```

To start a **new book**: `./use <student> "<book-folder>"` creates
`data/<student>/<book-folder>/` (with an `audio/` subfolder for your recordings);
drop the book's `.epub` into that folder, then run as below. `./use <student>`
with no book name auto-selects the book when the student has exactly one. To add
a **new student**: create `data/<name>/config.yaml` with a `language:` line.

## Quick start

One session = one chapter. Record yourself reading it and save the audio under
the active book as `audio/<chapter label>.wav`, using the **exact**
table-of-contents label — the label is the join key across every step (e.g.
`data/alex/Children-of-Time-Adrian-Tchaikovsky/audio/1.1 JUST A BARREL OF MONKEYS.wav`).
Then, from the project root:

```bash
./use alex Children-of-Time-Adrian-Tchaikovsky   # once, to select the context
./run.sh "1.1 JUST A BARREL OF MONKEYS"
```

That chains the three steps — extract the chapter text from the epub, transcribe
the recording with Deepgram, then analyze it with `--review` (LLM denoising). It
skips transcription if a transcript already exists (`FORCE=1` to redo) and
forwards extra flags to `analyze` (e.g. `./run.sh "1.1 …" --review-refresh`). To
run a step on its own, call `python extract_chapter.py` / `transcribe.py` /
`analyze.py` directly (use `python extract_chapter.py --list` to see chapter
labels). Every script targets the active student/book by default; pass
`--student`/`--book` to override it for one run without switching.

Then review `<book>/reports/<label>/` — start with **`focus_words.csv`** (ranked
drill list), **`ending_changes.csv`** and **`confusions.csv`** — and
`<book>/reports/_progress/` for trends across sessions.

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
  heard) — the highest-signal deviations to drill for clear, fluent reading.

Per chapter (`<book>/reports/<label>/`): `focus_words`, `ending_changes`,
`confusions`, `errors` (+ `errors_reviewed` with `--review`), `names`, `summary`,
`wpm_timeline`, `pauses`, and matching PNGs. Cumulative
(`<book>/reports/_progress/`): `sessions.csv` plus charts of raw + denoised
accuracy/WER, speech rate, and the words that recur most in your focus list.
Progress is per book, so each book tracks its own trend.

## Languages

Each student's `data/<student>/config.yaml` sets their default `language`;
`transcribe.py` and `analyze.py` also take `--language/-l` to override it for a
run (its default is the active student's config language). Profiles live in
`languages.py`: `en` (full English, rate in WPM), `generic` (any space-separated
language; also the fallback for unknown codes), and `ja`/`zh` (character-level,
rate in CPM). Any other Deepgram code (e.g. `ru`) uses the generic tokenizer
while keeping that code for transcription. `sessions.csv` is keyed by
`(chapter, language)` within each book, so the same chapter in two languages
stays separate. To add a first-class language, add a `LanguageProfile` and
register it in `PROFILES`.

## Repo layout

`data/` is **gitignored** — student configs, recordings, transcripts, and
reports all stay local; the committed repository is code-only.
