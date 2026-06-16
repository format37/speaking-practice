"""Transcribe a recorded chapter reading with Deepgram (nova-3).

Single speaker reading aloud -> word-level timing + confidence. The whole
file is sent in one request with a generous timeout; on timeout (or with
--chunk) it falls back to splitting the audio into 30-minute chunks via
ffmpeg, transcribing each, and merging words/transcript with time offsets.

Usage:
    python transcribe.py "1.1 JUST A BARREL OF MONKEYS" [--audio PATH] [--language en] [--chunk]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import config

from deepgram import DeepgramClient

# Timeout for Deepgram API requests (seconds).
# Nova-3 processes audio ~40x real-time, so a 43-min file is well under this.
REQUEST_TIMEOUT = 600

# Chunk duration for the fallback splitter (seconds).
CHUNK_DURATION = 30 * 60


def json_serializer(obj):
    """Handle datetime and other non-serializable objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def response_to_dict(response) -> dict:
    """Convert a Deepgram response into a plain dict (pydantic-aware)."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return response


def get_audio_duration(audio_file: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_file)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0.0
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0.0
    return float(data.get("format", {}).get("duration", 0) or 0)


def split_audio(audio_file: Path, chunk_duration: int, output_dir: Path) -> list:
    """Split audio into chunks with ffmpeg stream copy. Returns chunk paths."""
    duration = get_audio_duration(audio_file)
    if duration == 0:
        raise RuntimeError(f"Cannot determine duration of {audio_file}")

    suffix = audio_file.suffix or ".wav"
    chunks = []
    start = 0
    idx = 0
    while start < duration:
        chunk_path = output_dir / f"chunk_{idx:03d}{suffix}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_file),
                "-ss", str(start), "-t", str(chunk_duration),
                "-acodec", "copy", str(chunk_path),
            ],
            capture_output=True,
        )
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            chunks.append(chunk_path)
        start += chunk_duration
        idx += 1
    return chunks


def transcribe_single(client, audio_data: bytes, language: str):
    """Transcribe a single audio buffer (single speaker, word timing)."""
    return client.listen.v1.media.transcribe_file(
        request=audio_data,
        model="nova-3",
        language=language,
        punctuate=True,
        smart_format=True,
        request_options={"timeout_in_seconds": REQUEST_TIMEOUT},
    )


def merge_chunk_responses(chunk_responses: list) -> dict:
    """Merge (offset, response_dict) entries into one combined response.

    Offsets every word's start/end by the chunk's time offset and
    concatenates the per-chunk transcripts.
    """
    merged = {
        "metadata": {"duration": 0.0, "channels": 1},
        "results": {
            "channels": [{"alternatives": [{"transcript": "", "words": []}]}],
        },
    }

    transcript_parts = []
    for offset, resp in chunk_responses:
        results = resp.get("results", {}) if isinstance(resp, dict) else {}
        channels = results.get("channels", [])
        if channels and channels[0].get("alternatives"):
            alt = channels[0]["alternatives"][0]
            for w in alt.get("words", []) or []:
                shifted = dict(w)
                shifted["start"] = (w.get("start", 0) or 0) + offset
                shifted["end"] = (w.get("end", 0) or 0) + offset
                merged["results"]["channels"][0]["alternatives"][0]["words"].append(shifted)
            transcript_parts.append(alt.get("transcript", "") or "")

        meta = results and resp.get("metadata", {}) or resp.get("metadata", {})
        merged["metadata"]["duration"] += float(meta.get("duration", 0) or 0)

    merged["results"]["channels"][0]["alternatives"][0]["transcript"] = " ".join(
        p for p in transcript_parts if p
    ).strip()
    return merged


def transcribe_chunked(client, audio_file: Path, language: str) -> dict:
    """Split audio into chunks, transcribe each, and merge the results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Splitting audio into {CHUNK_DURATION // 60}-minute chunks...")
        chunks = split_audio(audio_file, CHUNK_DURATION, tmp_path)
        print(f"Split into {len(chunks)} chunk(s)")

        chunk_responses = []
        for i, chunk_path in enumerate(chunks):
            chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
            offset = i * CHUNK_DURATION
            print(f"  Chunk {i + 1}/{len(chunks)} "
                  f"({chunk_size_mb:.1f} MB, offset {format_timestamp(offset)})...")
            with open(chunk_path, "rb") as f:
                chunk_data = f.read()
            t0 = time.time()
            resp = transcribe_single(client, chunk_data, language)
            print(f"    Done in {time.time() - t0:.1f}s")
            chunk_responses.append((offset, response_to_dict(resp)))

    merged = merge_chunk_responses(chunk_responses)
    print(f"Merged {len(chunk_responses)} chunk(s) into a single transcript")
    return merged


def extract_transcript_text(response_dict: dict) -> str:
    """Pull the plain joined transcript out of a response dict."""
    results = response_dict.get("results", {}) if isinstance(response_dict, dict) else {}
    channels = results.get("channels", [])
    if channels and channels[0].get("alternatives"):
        return (channels[0]["alternatives"][0].get("transcript", "") or "").strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a chapter reading with Deepgram (nova-3)."
    )
    parser.add_argument("label", help='Chapter label, e.g. "1.1 JUST A BARREL OF MONKEYS"')
    parser.add_argument("--audio", default=None, help="Override audio path (defaults to config.audio_path(label))")
    parser.add_argument("-l", "--language", default=config.DEFAULT_LANGUAGE,
                        help="Language code (default: config.DEFAULT_LANGUAGE)")
    parser.add_argument("--chunk", action="store_true", help="Force chunked processing")
    args = parser.parse_args()

    label = args.label
    config.ensure_dirs()

    # Validate API key.
    if not config.DEEPGRAM_API_KEY:
        print("Error: DEEPGRAM_API_KEY is not set (add it to the project .env).")
        return 1

    # Resolve audio file.
    audio_file = Path(args.audio) if args.audio else config.audio_path(label)
    if not audio_file.exists():
        print(f"Error: audio file not found: {audio_file}")
        return 1

    client = DeepgramClient()

    duration = get_audio_duration(audio_file)
    file_size_mb = audio_file.stat().st_size / (1024 * 1024)
    print(f"Transcribing: {audio_file} ({file_size_mb:.1f} MB, {duration / 60:.1f} min)")

    use_chunks = args.chunk
    response_dict = None

    if not use_chunks:
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        print(f"Sending request to Deepgram "
              f"(model: nova-3, language: {args.language}, timeout: {REQUEST_TIMEOUT}s)...")
        t0 = time.time()
        try:
            response = transcribe_single(client, audio_data, args.language)
            print(f"Response received in {time.time() - t0:.1f}s")
            response_dict = response_to_dict(response)
        except Exception as e:
            if "timeout" in str(e).lower() or "Timeout" in type(e).__name__:
                print(f"\nTimeout after {time.time() - t0:.0f}s. "
                      f"Retrying with chunked processing...")
                use_chunks = True
            else:
                print(f"Error during transcription: {e}")
                return 1

    if use_chunks:
        response_dict = transcribe_chunked(client, audio_file, args.language)

    # Write outputs under data/ via config helpers.
    json_path = config.transcript_json(label)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False, default=json_serializer)
    print(f"Saved response JSON to: {json_path}")

    transcript_text = extract_transcript_text(response_dict)
    txt_path = config.transcript_dir(label) / f"{label}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript_text + "\n")
    print(f"Saved transcript text to: {txt_path}")

    # Word count for a quick summary.
    results = response_dict.get("results", {})
    channels = results.get("channels", [])
    word_count = 0
    if channels and channels[0].get("alternatives"):
        word_count = len(channels[0]["alternatives"][0].get("words", []) or [])

    print(f"Done: {word_count} words, "
          f"{len(transcript_text.split())} transcript tokens, "
          f"{'chunked' if use_chunks else 'single-request'} mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
