"""LLM-assisted error review + free name gazetteer for the speaking toolkit.

Two public entry points:

* :func:`build_name_gazetteer` -- FREE, offline, deterministic. Finds invented
  proper nouns / names (tokens that appear Capitalized but NOT at sentence start
  in >=2 places). Used as the default name filter AND as the fallback when the
  OpenAI key is missing.
* :func:`review_errors` -- OPT-IN. Sends candidate errors (with context) to
  OpenAI and returns per-item ``{keep, cause, reason}`` verdicts. Batched,
  cached per chapter, and degrades gracefully (returns ``{}``) when no key is
  configured or the API errors out.

Self-contained besides ``openai`` / ``pydantic`` / :mod:`config` / stdlib. The
OpenAI integration mirrors the proven pattern in questionarie-master.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

import config

# ---------------------------------------------------------------------------
# Pinned schema
# ---------------------------------------------------------------------------

CAUSES = [
    "real_mispronunciation",
    "real_ending_change",
    "real_substitution",
    "real_omission",
    "real_disfluency",
    "proper_noun_artifact",
    "asr_mishearing",
    "function_word_artifact",
    "number_or_punctuation",
    "homophone_or_variant",
    "other_noise",
]


class ReviewItem(BaseModel):
    id: int
    keep: bool  # True = genuine reader speaking error worth tracking; False = noise -> EXCLUDE
    cause: str = Field(json_schema_extra={"enum": CAUSES})  # one of CAUSES
    reason: str  # <=12 words


class ReviewResult(BaseModel):
    items: list[ReviewItem]


SYSTEM = (
    "You are a meticulous English-pronunciation coach reviewing an automated "
    "reading assessment. A non-native English speaker (Russian L1, preparing for "
    "English-language tech interviews) read a book chapter aloud; the audio was "
    "transcribed by an ASR and diffed against the original book text to produce "
    "candidate 'errors' (reference word vs. what the ASR heard). For each "
    "candidate, decide whether it is a GENUINE speaking issue worth tracking "
    "(keep=true) or MEASUREMENT NOISE to EXCLUDE from the learner's statistics "
    "(keep=false), and give a cause and a short reason.\n"
    "NOISE (keep=false): invented proper nouns / names / places from the book "
    "that the ASR cannot know and renders as garbage (proper_noun_artifact); "
    "punctuation or number rendering differences like '2' vs 'two' "
    "(number_or_punctuation); tiny function-word swaps typical of ASR rather "
    "than misreading, e.g. to/of, in/on, and/an (function_word_artifact / "
    "asr_mishearing); high-ambiguity homophones like their/there "
    "(homophone_or_variant).\n"
    "REAL (keep=true): dropped or changed grammatical endings - missing plural "
    "-s, possessive, or tense -ed/-ing, e.g. screens->screen, ignored->ignores "
    "(real_ending_change) - these matter for interviews; confident substitutions "
    "of one real common word for another where the ASR clearly heard a different "
    "real word, e.g. contact->contract, climbing->clipping (real_substitution / "
    "real_mispronunciation); genuine omissions of real content words "
    "(real_omission); genuine repeated/stuttered words (real_disfluency).\n"
    "When unsure, keep=true for ordinary in-vocabulary words; set keep=false only "
    "when fairly confident it is an artifact. Judge each item independently using "
    "its context window."
)


# ---------------------------------------------------------------------------
# Strict JSON schema helper (OpenAI is picky)
# ---------------------------------------------------------------------------

def _strict(node):
    """Recursively make a JSON schema OpenAI-strict-mode compatible.

    On every object: ``additionalProperties=false`` and ``required`` = all keys.
    Strip ``title`` everywhere. Leave ``$ref``/``$defs``/``enum`` intact.
    """
    if isinstance(node, dict):
        node.pop("title", None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _strict(v)
    elif isinstance(node, list):
        for v in node:
            _strict(v)
    return node


def review_strict_schema() -> dict:
    """OpenAI strict ``json_schema`` body for :class:`ReviewResult`."""
    return _strict(ReviewResult.model_json_schema())


# ---------------------------------------------------------------------------
# Tolerant JSON parsing (mirrors questionarie-master)
# ---------------------------------------------------------------------------

def _iter_json_objects(text: str):
    """Yield each balanced top-level ``{...}`` substring (string/escape aware)."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start:i + 1]


def _parse_result(content: str) -> ReviewResult:
    """Parse a model reply into a :class:`ReviewResult`, scanning for braces."""
    try:
        return ReviewResult.model_validate_json(content)
    except Exception:
        pass
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return ReviewResult.model_validate(obj)
    except Exception:
        pass
    for candidate in _iter_json_objects(content):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "items" in obj:
            try:
                return ReviewResult.model_validate(obj)
            except Exception:
                continue
    raise ValueError("No valid ReviewResult JSON found in model output")


# ---------------------------------------------------------------------------
# Normalization (kept consistent with analyze.py's word normalization)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_SPLIT_RE = re.compile(r"[.!?]+[\"')\]]*\s+|\n+")


def _norm(word: str) -> str:
    """Lowercase + strip surrounding punctuation, keeping inner ' and -."""
    return word.strip().strip(".,;:!?\"'()[]{}<>").lower()


# ---------------------------------------------------------------------------
# Free name gazetteer
# ---------------------------------------------------------------------------

def _load_english_lexicon():
    """Load a system English word list once, lowercased; None if unavailable.

    Used to tell invented book names (Brin, Sering, Avrana) from ordinary
    capitalized words (Doctor, Earth, World, Eye) so the latter stay drillable.
    """
    for p in ("/usr/share/dict/words", "/usr/share/dict/american-english",
              "/usr/share/dict/british-english"):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                return {w.strip().lower() for w in f if w.strip().isalpha()}
        except OSError:
            continue
    return None


_ENGLISH_LEXICON = _load_english_lexicon()

# Common always-capitalized English words kept as real (only used when no system
# lexicon is available): pronoun, honorifics, calendar, nationalities, celestial.
_NAME_STOPLIST = {
    "i", "mr", "mrs", "ms", "dr", "sir", "madam", "mister", "lord", "lady",
    "god", "doctor", "professor", "captain", "colonel", "sergeant", "major",
    "general", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
    "english", "russian", "american", "european", "asian", "african",
    "earth", "world", "sun", "moon", "sky", "heaven", "ocean", "sea",
}


def build_name_gazetteer(text_or_texts) -> set[str]:
    """Return the set of likely INVENTED proper-noun forms (normalized, lowercased).

    Heuristic, FREE, offline, deterministic. A candidate appears Capitalized
    mid-sentence (not sentence-initial, not all-caps) in >=2 places. A candidate
    is treated as an invented name only if it is NOT an ordinary English word:
    decided via the system lexicon when present, otherwise via "never appears
    lowercased in the text" plus a small stoplist. This keeps common capitalized
    words (Doctor, Earth, World, Eye, "I") as real drill targets and excludes only
    book-specific invented names (Brin, Sering, Avrana...).

    ``text_or_texts`` may be a single string or an iterable of strings.
    """
    if isinstance(text_or_texts, str):
        texts = [text_or_texts]
    else:
        texts = [t for t in text_or_texts if t]

    cap_mid: dict[str, int] = {}
    lower: dict[str, int] = {}
    for text in texts:
        if not text:
            continue
        for sentence in _SENT_SPLIT_RE.split(text):
            tokens = _WORD_RE.findall(sentence)
            for idx, tok in enumerate(tokens):
                norm = _norm(tok)
                if not norm:
                    continue
                if tok.islower():
                    lower[norm] = lower.get(norm, 0) + 1
                    continue
                if idx == 0:
                    continue  # sentence-initial capitalization is uninformative
                if not tok[0].isupper():
                    continue
                if tok.isupper() and len(tok) > 1:
                    continue  # all-caps headings/acronyms are not names
                cap_mid[norm] = cap_mid.get(norm, 0) + 1

    names: set[str] = set()
    for norm, c in cap_mid.items():
        if c < 2 or len(norm) < 2 or norm == "i":
            continue
        if _ENGLISH_LEXICON is not None:
            if norm in _ENGLISH_LEXICON:
                continue  # ordinary English word, not an invented name
        elif lower.get(norm, 0) > 0 or norm in _NAME_STOPLIST:
            continue
        names.add(norm)
    return names


# ---------------------------------------------------------------------------
# Context windows
# ---------------------------------------------------------------------------

def _context_window(ref_tokens, ref_pos) -> str:
    """~5 display ref tokens around ``ref_pos`` (the ref word +/- ~2)."""
    if not ref_tokens:
        return ""
    if ref_pos is None:
        return ""
    try:
        pos = int(ref_pos)
    except (TypeError, ValueError):
        return ""
    n = len(ref_tokens)
    pos = max(0, min(pos, n - 1))
    lo = max(0, pos - 2)
    hi = min(n, pos + 3)
    return " ".join(str(t) for t in ref_tokens[lo:hi])


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _item_sig(it) -> str:
    """Stable PER-ITEM signature. Busts automatically on re-transcription /
    edited text. Per-item (not whole-set) keying lets an interrupted review
    resume: already-judged errors are reused, only the rest are re-queried.
    """
    payload = json.dumps(
        [
            it.get("type"),
            it.get("ref_word"),
            it.get("hyp_word"),
            round(float(it.get("confidence") or 0.0), 3),
            it.get("ref_pos"),
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path, model: str) -> dict:
    """Return ``{item_sig: verdict}`` for ``model``; ``{}`` on miss or old format."""
    try:
        data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("model") != model:
        return {}
    by_sig = data.get("by_sig")
    return by_sig if isinstance(by_sig, dict) else {}


def _save_cache(cache_path: Path, model: str, by_sig: dict) -> None:
    try:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps({"model": model, "by_sig": by_sig}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass  # caching is best-effort; never crash the analysis


# ---------------------------------------------------------------------------
# Public review entry point
# ---------------------------------------------------------------------------

def review_errors(
    items,
    ref_tokens,
    *,
    model=None,
    batch_size=40,
    cache_path=None,
    refresh=False,
) -> dict[int, dict]:
    """Review candidate errors with an LLM; return ``{id: {keep, cause, reason}}``.

    ``items``: list of dicts ``{id, type, ref_word, hyp_word, confidence, ref_pos}``.
    ``ref_tokens``: the reference DISPLAY token list (original casing) for context.

    Returns a dict keyed by item ``id`` (the row index into ``items``). Any id the
    model omits or returns malformed defaults to ``keep=True`` (in-vocabulary safe
    default); the caller forces gazetteer names to ``keep=False``.

    Graceful degradation:
      * No ``OPENAI_KEY`` -> log one line, return ``{}`` (caller falls back to gazetteer).
      * Per-batch API/parse failure -> log + skip that batch only (its ids fall
        through to defaults); other batches still run.
    """
    items = list(items)
    if not items:
        return {}
    used_model = model or config.OPENAI_MODEL

    # ----- per-item cache: reuse known verdicts, query only the rest -------
    by_sig: dict = {}
    if cache_path is not None and not refresh:
        by_sig = _load_cache(Path(cache_path), used_model)

    verdicts: dict[int, dict] = {}
    to_query = []
    for it in items:
        sig = _item_sig(it)
        if sig in by_sig:
            verdicts[int(it["id"])] = by_sig[sig]
        else:
            to_query.append((sig, it))

    if not to_query:
        if cache_path is not None:
            print(f"review: using cached verdicts ({len(verdicts)} items)")
        return verdicts

    if not config.OPENAI_KEY:
        print("OPENAI_KEY not set; falling back to free name-gazetteer denoising")
        return verdicts  # whatever was already cached (possibly empty)

    try:
        import openai
    except ImportError:
        print("openai package not installed; falling back to gazetteer denoising")
        return verdicts

    try:
        client = openai.OpenAI(api_key=config.OPENAI_KEY)
    except Exception as exc:  # pragma: no cover - construction rarely fails
        print(f"review: could not init OpenAI client ({exc}); falling back to gazetteer")
        return verdicts

    schema = review_strict_schema()
    query_items = [it for _, it in to_query]
    sig_by_id = {int(it["id"]): sig for sig, it in to_query}
    new_count = 0

    n_batches = (len(query_items) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = query_items[b * batch_size:(b + 1) * batch_size]
        payload = []
        for it in batch:
            payload.append(
                {
                    "id": int(it["id"]),
                    "type": it.get("type"),
                    "reference": it.get("ref_word"),
                    "heard": it.get("hyp_word"),
                    "asr_confidence": it.get("confidence"),
                    "context": _context_window(ref_tokens, it.get("ref_pos")),
                }
            )
        user_text = (
            "Review these candidate reading errors. Return one verdict per id.\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        try:
            resp = client.chat.completions.create(
                model=used_model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "review",
                        "schema": schema,
                        "strict": True,
                    },
                },
                max_completion_tokens=25000,
            )
            content = resp.choices[0].message.content
            result = _parse_result(content)
        except (
            openai.AuthenticationError,
            openai.RateLimitError,
            openai.BadRequestError,
            openai.APIError,
        ) as exc:
            print(f"review: batch {b + 1}/{n_batches} API error ({exc}); skipping batch")
            continue
        except Exception as exc:
            print(f"review: batch {b + 1}/{n_batches} failed ({exc}); skipping batch")
            continue

        for ri in result.items:
            cause = ri.cause if ri.cause in CAUSES else "other_noise"
            v = {
                "keep": bool(ri.keep),
                "cause": cause,
                "reason": (ri.reason or "")[:120],
            }
            rid = int(ri.id)
            verdicts[rid] = v
            if rid in sig_by_id:
                by_sig[sig_by_id[rid]] = v
            new_count += 1

        # Persist after EACH batch so an interruption (quota/network) still saves
        # progress; the next run resumes instead of re-paying for everything.
        if cache_path is not None and by_sig:
            _save_cache(Path(cache_path), used_model, by_sig)

    print(
        f"review: obtained {len(verdicts)} verdicts "
        f"({new_count} new) across {n_batches} batch(es)"
    )
    return verdicts
