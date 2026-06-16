# Speaking Practice

Daily English reading-aloud practice for ML-interview prep: read a book chapter
out loud, transcribe the recording with Deepgram, and compare it against the
reference text to surface mispronounced / dropped words and track progress over time.

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

## Outputs

Everything derived is written under `data/`:

- `data/book/chapters/<label>.txt` — extracted reference text for the chapter.
- `data/transcripts/<label>/` — per-chapter Deepgram output (JSON + plain text).
- `data/reports/<label>/` — per-chapter results. CSVs: `errors` (every
  word-level diff, typed as omission / repetition / insertion / ending-mixup /
  mispronunciation / substitution), `focus_words` (ranked drill list), plus
  `wpm_timeline`, `pauses` and `summary`. PNGs: `error_breakdown`,
  `focus_words`, `confidence_hist`, `wpm_timeline`, `phoneme_groups`.
- `data/reports/_progress/` — cumulative across sessions: `sessions.csv` (one row
  per chapter) and `progress_*.png` charts tracking accuracy/WER, speech rate,
  error trends, and the words that recur most often in your focus list.

## Repo layout

`data/` is **gitignored** — recordings, transcripts, and reports stay local so
the committed repository remains code-only.
