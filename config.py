"""Shared paths and configuration for the speaking-practice toolkit."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
BOOK_DIR = DATA / "book"
CHAPTERS_DIR = BOOK_DIR / "chapters"        # extracted reference text (.txt)
AUDIO_DIR = DATA / "audio"                   # recorded readings (.wav)
TRANSCRIPTS_DIR = DATA / "transcripts"       # per-chapter Deepgram output (json + txt)
REPORTS_DIR = DATA / "reports"               # per-chapter csv + png
PROGRESS_DIR = REPORTS_DIR / "_progress"     # cumulative cross-session output
SESSIONS_CSV = PROGRESS_DIR / "sessions.csv"

DEFAULT_EPUB = BOOK_DIR / "Children-of-Time-Adrian-Tchaikovsky.epub"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
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


def ensure_dirs():
    for d in (CHAPTERS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, REPORTS_DIR, PROGRESS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def chapter_txt(label):     return CHAPTERS_DIR / f"{label}.txt"
def audio_path(label):      return AUDIO_DIR / f"{label}.wav"
def transcript_dir(label):  return TRANSCRIPTS_DIR / label
def transcript_json(label): return transcript_dir(label) / f"{label}.json"
def report_dir(label):      return REPORTS_DIR / label
