# Speaking Practice

Daily reading-aloud practice for ML-interview prep: read a book chapter out
loud, transcribe the recording with Deepgram, and compare it against the
reference text to surface mispronounced / dropped words and track progress over
time. English is the default, but the analysis is **language-pluggable** — see
[Languages](#languages).

## Daily workflow

1. **Record** yourself reading one chapter aloud. Save it as
   `data/audio/<chapter label>.wav`, using the **exact** table-of-contents label
   (e.g. `data/audio/1.1 JUST A BARREL OF MONKEYS.wav`). The label is the join
   key across every step.
2. **Extract** the reference text for that chapter from the epub.
3. **Transcribe** the recording with Deepgram.
4. **Analyze** transcript vs. reference to score the reading.
5. **Review** the focus words (errors to drill) and your cumulative progress.

## Setup

```bash
# Python dependencies
pip install -r requirements.txt

# System dependency: ffmpeg + ffprobe (must be on PATH)
sudo apt install ffmpeg          # or your OS package manager

# Deepgram API key
cp .env.example .env             # then paste your key into .env
# (you can copy the key from /home/alex/projects/deepgram-stt/.env)
```

## Commands

All scripts live flat at the project root and take the chapter label as the
single argument. Run them from the project root.

```bash
# 1) Extract the reference chapter text from the epub
#    (use --list to print every chapter label)
python extract_chapter.py "1.1 JUST A BARREL OF MONKEYS"

# 2) Transcribe the recording (needs data/audio/1.1 JUST A BARREL OF MONKEYS.wav)
python transcribe.py "1.1 JUST A BARREL OF MONKEYS"

# 3) Analyze transcript vs. reference, write the report, update progress
python analyze.py "1.1 JUST A BARREL OF MONKEYS"
```

`transcribe.py` and `analyze.py` both take `--language/-l` (default
`PRACTICE_LANGUAGE`, falling back to `en`) — see [Languages](#languages).

## Outputs

Everything derived is written under `data/`:

- `data/book/chapters/<label>.txt` — extracted reference text for the chapter.
- `data/transcripts/<label>/` — per-chapter Deepgram output (JSON + plain text).
- `data/reports/<label>/` — per-chapter results. CSVs: `errors` (every
  word-level diff, typed as omission / repetition / insertion / ending-mixup /
  mispronunciation / substitution), `focus_words` (ranked drill list),
  `ending_changes` and `confusions` (the two clean headline signals), `names`
  (invented proper nouns excluded from the metrics), plus `wpm_timeline`,
  `pauses` and `summary`. PNGs: `error_breakdown`, `focus_words`,
  `ending_changes`, `confusions`, `confidence_hist`, `wpm_timeline`.
- `data/reports/_progress/` — cumulative across sessions: `sessions.csv` (one row
  per chapter) and `progress_*.png` charts tracking accuracy/WER, speech rate,
  error trends, and the words that recur most often in your focus list.

## Denoising / focus on real errors

The book invents proper nouns (Sering, Brin, Avrana, …) that the ASR cannot
know and renders as garbage. Those phantom "errors" have no pronunciation
ground truth, so they inflate the score and crowd out the real issues. To keep
the numbers honest and comparable across chapters:

- **The phoneme-group chart was removed.** It bucketed errors by *spelling*
  (e.g. a "th" bucket lumping `through` / `within` / `that`) rather than by the
  sound that actually changed, so it was misleading.
- **Names are excluded from the metrics** via a free, offline *gazetteer* —
  tokens that appear capitalized mid-sentence in several places are treated as
  invented names. They leave the accuracy/WER denominator entirely (so correctly
  read names don't inflate the score either) and are reported separately in
  `names.csv`, never as drill targets. This makes chapters comparable. Reports
  show both the raw and the **denoised** accuracy / WER.
- **The headline is the two clean signals:** `ending_changes` (dropped or
  changed grammatical endings — `screens`→`screen`, `ignored`→`ignores`; these
  matter for interviews) and `confusions` (confident substitutions of one real
  word for another, where the ASR — at confidence ≥ 0.85 — clearly heard a
  different real word).

All of the above runs by default with no API key.

**Opt-in LLM review.** Add `--review` (alias `--llm-review`) to have an LLM judge
each candidate error *with its context* and mark it keep/exclude plus a cause and
a short reason, which then drives the denoised metrics:

```bash
python analyze.py "1.1 JUST A BARREL OF MONKEYS" --review
python analyze.py "1.1 JUST A BARREL OF MONKEYS" --review --review-refresh  # ignore cache
```

Two backends (select with `--review-backend` or the `REVIEW_BACKEND` env var):

- **`claude`** (default) — uses your local **Claude subscription** via the Claude
  Agent SDK (the authenticated `claude` CLI). **No API key and no per-token cost.**
- **`openai`** — the paid OpenAI API (model `gpt-5.5`, override with `OPENAI_MODEL`);
  set `OPENAI_KEY` in `.env`.

Verdicts are cached **per item** per chapter, so an interrupted review *resumes*
(only the un-judged errors are re-sent) and re-runs are cheap. If the chosen
backend is unavailable it prints a clear message and falls back to the free
gazetteer — it never crashes.

## Languages

The analysis is language-pluggable. Pass `--language/-l <code>` to
`transcribe.py` and `analyze.py` (default `config.DEFAULT_LANGUAGE`, set via the
`PRACTICE_LANGUAGE` env var, falling back to `en`). The code selects a
**language profile** that drives every language-specific step (tokenizing,
normalizing, number handling, error classification, sound groups, and the
reading-rate band); alignment, metrics, CSV, and plotting stay shared.

Built-in profiles:

- **`en`** — full English profile (the original, verified behavior): English
  inflectional ending-mixups, confusable pairs, the 8 phoneme/sound groups
  tuned for a Russian-native learner, proper-noun flagging, and word-level
  rate in **WPM** (comfortable band ~130–160).
- **`generic`** — any space-separated, Latin-script language. Casefolding,
  number normalization, and proper-noun flagging are on, but without the
  English-specific suffix/confusable/phoneme rules (a generic ending heuristic,
  no sound groups). Word-level rate in WPM (~130–160). Any unknown language code
  falls back to this profile, using that code as the Deepgram language.
- **`ja`** (and `zh`) — **character-level** analysis: text is tokenized per
  significant character, no casefolding/number-normalization/proper-noun
  flagging, no ending-mixups or sound groups, and the reading rate is reported
  in **CPM** (characters per minute, comfortable band ~250–400). Word-level
  Japanese analysis via an optional MeCab tokenizer is a possible future
  addition — there is intentionally **no mandatory heavy dependency**.

The cumulative `sessions.csv` is language-agnostic and keyed by
`(chapter, language)`, so the same chapter read in two languages keeps separate
rows.

To add a language, add a `LanguageProfile` in `languages.py` (and register it in
`PROFILES`). Reuse the `en`, `generic`, or `ja` profile as a starting point
depending on whether your language is word- or character-based.

## Repo layout

`data/` is **gitignored** — recordings, transcripts, and reports stay local so
the committed repository remains code-only.
