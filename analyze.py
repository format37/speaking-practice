"""Compare a read-aloud transcript against the original book chapter.

Produces per-session CSV reports and PNG diagrams plus cumulative cross-session
progress, telling a learner which words/sounds to drill and how their fluency is
progressing. Language-specific behaviour (normalization, tokenization, ending /
mispronunciation / sound-group classification, word- vs character-level rate) is
delegated entirely to a LanguageProfile from ``languages.get_profile()``; this
module owns only the language-agnostic alignment, metrics, CSV and plotting.

Usage:
    python analyze.py "1.1 JUST A BARREL OF MONKEYS"
    python analyze.py "1.1 ..." --language en
    python analyze.py "1.1 ..." --reference ref.txt --transcript hyp.json \\
        --label custom --no-progress
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt
import pandas as pd

import config
import languages

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
LOW_CONF = 0.6
PAUSE_GAP = 0.5            # seconds: gap > this is counted as a pause
HESITATION_GAP = 1.0

# error categories
CORRECT = "correct"
OMISSION = "omission"
INSERTION = "insertion"
REPETITION = "repetition"
ENDING = "ending_mixup"
MISP = "mispronunciation"
SUBST = "substitution"
ERROR_CATEGORIES = [OMISSION, INSERTION, REPETITION, ENDING, MISP, SUBST]

# Notes shown alongside flagged sound groups. The profile decides which group
# keys exist (empty for non-English); keys we do not know simply get no note.
PHONEME_NOTES = {
    "th": "th -> tongue between teeth, not /s/ or /z/",
    "w_v": "w -> round lips (no teeth); v -> teeth on lip",
    "r": "soft English r, do not roll/trill it",
    "final_voiced": "keep final b/d/g/v/z voiced, do not devoice",
    "final_cluster": "release every consonant in the final cluster",
    "h_x": "light breathy h, not a hard /x/",
    "ng": "nasal ng, do not add a hard g",
    "vowel_len": "hold long vowels (ee, oo) clearly",
}

ERROR_NOTE = {
    SUBST: "read a different word - check meaning & spelling",
    MISP: "close miss - drill the pronunciation",
    ENDING: "grammatical ending slip - watch -s/-ed/-ing",
    OMISSION: "skipped - slow down and read every word",
    INSERTION: "added an extra word",
    REPETITION: "stutter/re-read - aim for a smooth first take",
}

# Language-agnostic cumulative schema. The `language` column lets the same
# chapter be tracked in several languages; sound-group rates are NOT stored here
# (the EN progress chart re-reads each session's phoneme_groups.csv instead).
SESSION_COLUMNS = [
    "date", "chapter", "language", "n_ref_units", "accuracy", "wer",
    "omissions", "insertions", "repetitions", "ending_mixups",
    "mispronunciations", "substitutions",
    "overall_rate", "articulation_rate", "mean_confidence",
    "n_pauses", "duration_min",
]


# --------------------------------------------------------------------------- #
# Filename safety
# --------------------------------------------------------------------------- #
def slug(label):
    """Sanitize a label into a filesystem-safe slug for file names."""
    return re.sub(r'[/\\:*?"<>|]+', "_", label).strip() or "session"


# --------------------------------------------------------------------------- #
# Reference / transcript loading (profile-driven)
# --------------------------------------------------------------------------- #
_SENTENCE_END = (".", "?", "!")


def _strip_leading_nonalnum(s):
    """Drop leading quotes/parens etc. so the first *letter* is tested."""
    i = 0
    while i < len(s) and not s[i].isalnum():
        i += 1
    return s[i:]


def tokenize_text(profile, text):
    """Tokenize plain text -> parallel arrays (norm, display, start, end, conf).

    Uses ``profile.normalize`` per display token and ``profile.tokenize`` is not
    used here directly because we must keep the norm and display arrays the same
    length (one display token -> its norm tokens) so opcode indices stay aligned.
    For the char profile, ``normalize`` of a whitespace-delimited token still
    yields per-character norms via the same loop. start/end/conf are None for
    reference text.
    """
    norm, disp = [], []
    char_unit = profile.unit == "char"
    for raw in text.split():
        for n in _normalize_parts(profile, raw):
            norm.append(n)
            disp.append(n if char_unit else raw)
    none = [None] * len(norm)
    return norm, disp, list(none), list(none), list(none)


def _normalize_parts(profile, raw):
    """Return the list of normalized tokens a single raw display token yields.

    ``profile.normalize`` returns a single normalized string for a word-unit
    profile, or — for compound splitting / char profiles — we expand through the
    profile's tokenizer so e.g. hyphen compounds (en) and per-char (ja) both
    produce the right parallel-array length. We call ``profile.tokenize`` on the
    single raw token to obtain its normalized pieces.
    """
    return profile.tokenize(raw)


def load_reference(profile, path):
    """Read reference chapter, drop the title, flag proper nouns.

    Returns (title, ref_arrays, body) where ref_arrays is
    (norm, disp, start, end, conf, is_proper). Proper-noun flagging only happens
    when ``profile.detect_proper_nouns``: a token is proper iff its first letter
    (after stripping leading quotes/parens) is uppercase AND it is not
    sentence-initial (first body token, or following a token ending in .?! or a
    paragraph break). The flag is replicated to every normalized piece of a split
    compound so it survives tokenization.
    """
    if not path.exists():
        sys.exit(f"ERROR: reference file not found: {path}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    title, body_lines, started = "", [], False
    for ln in lines:
        if not started and ln.strip():
            title = ln.strip()
            started = True
            continue
        body_lines.append(ln)
    body = "\n".join(body_lines).strip() or raw  # fallback if only one line

    norm, disp, propers = [], [], []
    sentence_start = True
    for para in body.split("\n\n"):
        sentence_start = True            # paragraph break resets the flag
        for raw_tok in para.split():
            stripped = _strip_leading_nonalnum(raw_tok)
            is_proper = bool(
                profile.detect_proper_nouns and stripped
                and stripped[0].isupper() and not sentence_start)
            char_unit = profile.unit == "char"
            for n in _normalize_parts(profile, raw_tok):
                norm.append(n)
                disp.append(n if char_unit else raw_tok)
                propers.append(is_proper)
            sentence_start = raw_tok.rstrip("\"')]").endswith(_SENTENCE_END)
    none = [None] * len(norm)
    ref = (norm, disp, list(none), list(none), list(none), propers)
    return title, ref, body


def _extract_words(data):
    """Defensively pull the words[] list out of a Deepgram-style json."""
    if isinstance(data, dict):
        res = data.get("results", data)
        chans = res.get("channels") if isinstance(res, dict) else None
        if chans:
            alts = chans[0].get("alternatives") if chans else None
            if alts and isinstance(alts, list):
                w = alts[0].get("words")
                if w is not None:
                    return w
        # flattened / merged shapes
        if isinstance(res, dict) and res.get("words") is not None:
            return res["words"]
        if data.get("words") is not None:
            return data["words"]
        alts = data.get("alternatives")
        if alts and isinstance(alts, list) and alts[0].get("words") is not None:
            return alts[0]["words"]
    return None


def load_transcript(profile, path):
    """Load transcript json -> parallel arrays with timing & confidence.

    Delegates the word/char split to ``profile.split_hypothesis`` so that
    char-unit profiles explode each Deepgram word into its significant chars,
    all sharing that word's start/end/confidence.
    """
    if not path.exists():
        sys.exit(f"ERROR: transcript file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        sys.exit(f"ERROR: could not parse transcript json {path}: {exc}")
    words = _extract_words(data)
    if words is None:
        sys.exit(f"ERROR: no words[] found in transcript {path}")
    norm, disp, starts, ends, confs = [], [], [], [], []
    for unit in profile.split_hypothesis(words):
        norm.append(unit["norm"])
        disp.append(unit["display"])
        starts.append(unit["start"])
        ends.append(unit["end"])
        confs.append(unit["conf"])
    return norm, disp, starts, ends, confs


def resolve_transcript_path(args, label):
    """Resolve transcript path from --transcript, the canonical name, or newest."""
    if args.transcript:
        return config.Path(args.transcript)
    if label is None:
        sys.exit("ERROR: provide --transcript or a chapter label to locate "
                 "the transcript json.")
    canonical = config.transcript_json(label)
    if canonical.exists():
        return canonical
    tdir = config.transcript_dir(label)
    if tdir.exists():
        cands = sorted(tdir.glob("*.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        for c in cands:
            try:
                data = json.loads(c.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if _extract_words(data) is not None:
                return c
    return canonical  # let load_transcript raise the clear error


# --------------------------------------------------------------------------- #
# Alignment & classification
# --------------------------------------------------------------------------- #
def collapse_stutters(hyp):
    """Collapse immediate duplicate hyp tokens (stutters) before alignment.

    Returns (clean_hyp, keep_idx, repetition_items). A stutter is a hyp token
    equal to, or a >=2-char prefix of, the immediately following hyp token
    ("the the", "c- cat"). Such tokens are removed from the alignment stream and
    recorded directly as REPETITION so they cannot pollute difflib opcodes.
    """
    hn, hd, hs, he, hc = hyp
    keep_idx = []           # original indices kept for alignment
    rep_items = []
    i, n = 0, len(hn)
    while i < n:
        tok = hn[i]
        if i + 1 < n:
            nxt = hn[i + 1]
            is_dup = (tok == nxt)
            is_partial = (len(tok) >= 2 and tok != nxt and nxt.startswith(tok))
            if is_dup or is_partial:
                rep_items.append({
                    "ref_word": None, "hyp_word": hd[i], "type": REPETITION,
                    "ref_pos": None, "hyp_pos": i,
                    "time_start": hs[i], "confidence": hc[i],
                })
                i += 1
                continue
        keep_idx.append(i)
        i += 1
    clean = (
        [hn[k] for k in keep_idx],
        [hd[k] for k in keep_idx],
        [hs[k] for k in keep_idx],
        [he[k] for k in keep_idx],
        [hc[k] for k in keep_idx],
    )
    return clean, keep_idx, rep_items


def pair_replace_block(rn, hn, i1, i2, j1, j2):
    """Best-similarity 1:1 pairing of a replace block's sub-sequences.

    Returns (pairs, ref_leftover, hyp_leftover) where pairs is a list of
    (ref_idx, hyp_idx) into the original arrays. Pairs are chosen greedily by
    descending SequenceMatcher ratio so close pairs match across length skews.
    """
    refs = list(range(i1, i2))
    hyps = list(range(j1, j2))
    cand = []
    for ri in refs:
        for hj in hyps:
            ratio = SequenceMatcher(None, rn[ri], hn[hj]).ratio()
            cand.append((ratio, ri, hj))
    cand.sort(key=lambda t: (-t[0], abs((t[1] - i1) - (t[2] - j1)), t[1], t[2]))
    used_ref, used_hyp, pairs = set(), set(), []
    for ratio, ri, hj in cand:
        if ri in used_ref or hj in used_hyp:
            continue
        used_ref.add(ri)
        used_hyp.add(hj)
        pairs.append((ri, hj))
    pairs.sort(key=lambda p: p[0])
    ref_leftover = [r for r in refs if r not in used_ref]
    hyp_leftover = [h for h in hyps if h not in used_hyp]
    return pairs, ref_leftover, hyp_leftover


def align_and_classify(profile, ref, hyp):
    """Align normalized streams and classify every item.

    ref is (norm, disp, start, end, conf, is_proper); hyp is the 5-tuple. Returns
    a list of item dicts with keys ref_word, hyp_word, type, ref_pos, hyp_pos,
    time_start, confidence, is_proper. Replace-pair classification is delegated
    to ``profile.classify_replace_pair``.
    """
    rn, rd, _, _, _, r_proper = ref
    (hn, hd, hs, he, hc), _keep, rep_items = collapse_stutters(hyp)
    items = []
    prev_read_norm = None  # last hyp token actually emitted (for stutters)

    def add(ref_word, hyp_word, typ, ref_pos, hyp_pos):
        nonlocal prev_read_norm
        ts = hs[hyp_pos] if hyp_pos is not None else None
        cf = hc[hyp_pos] if hyp_pos is not None else None
        items.append({
            "ref_word": ref_word, "hyp_word": hyp_word, "type": typ,
            "ref_pos": ref_pos, "hyp_pos": hyp_pos,
            "time_start": ts, "confidence": cf,
            "is_proper": bool(r_proper[ref_pos]) if ref_pos is not None else False,
        })
        if hyp_pos is not None:
            prev_read_norm = hn[hyp_pos]

    sm = SequenceMatcher(a=rn, b=hn, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                add(rd[i1 + k], hd[j1 + k], CORRECT, i1 + k, j1 + k)
        elif tag == "delete":
            for k in range(i1, i2):
                add(rd[k], None, OMISSION, k, None)
        elif tag == "insert":
            for k in range(j1, j2):
                tok = hn[k]
                if prev_read_norm is not None and tok == prev_read_norm:
                    add(None, hd[k], REPETITION, None, k)
                elif (prev_read_norm is not None and len(tok) >= 2
                      and prev_read_norm.startswith(tok)):
                    add(None, hd[k], REPETITION, None, k)
                else:
                    nxt = rn[i1] if i1 < len(rn) else None
                    if nxt and len(tok) >= 2 and nxt.startswith(tok) and tok != nxt:
                        add(None, hd[k], REPETITION, None, k)
                    else:
                        add(None, hd[k], INSERTION, None, k)
        elif tag == "replace":
            la, lb = i2 - i1, j2 - j1
            pairs, ref_leftover, hyp_leftover = pair_replace_block(
                rn, hn, i1, i2, j1, j2)
            for ri, ci in pairs:
                add(rd[ri], hd[ci],
                    profile.classify_replace_pair(rn[ri], hn[ci]), ri, ci)
            for ri in ref_leftover:
                add(rd[ri], None, OMISSION, ri, None)
            for ci in hyp_leftover:
                tok = hn[ci]
                if prev_read_norm is not None and tok == prev_read_norm:
                    add(None, hd[ci], REPETITION, None, ci)
                else:
                    add(None, hd[ci], INSERTION, None, ci)
            assert len(pairs) + len(ref_leftover) == la
            assert len(pairs) + len(hyp_leftover) == lb
    items.extend(rep_items)
    return items


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(items, n_ref):
    counts = {c: 0 for c in [CORRECT] + ERROR_CATEGORIES}
    for it in items:
        counts[it["type"]] += 1
    correct = counts[CORRECT]
    accuracy = correct / n_ref if n_ref else 0.0
    S = counts[ENDING] + counts[MISP] + counts[SUBST]
    D = counts[OMISSION]
    I = counts[INSERTION]
    wer = (S + D + I) / n_ref if n_ref else 0.0
    confs = [it["confidence"] for it in items
             if it["confidence"] is not None and it["type"] != OMISSION]
    mean_conf = sum(confs) / len(confs) if confs else float("nan")
    min_conf = min(confs) if confs else float("nan")
    return {
        "n_ref": n_ref, "correct": correct, "accuracy": accuracy,
        "S": S, "D": D, "I": I, "wer": wer,
        "mean_confidence": mean_conf, "min_confidence": min_conf,
        **{c: counts[c] for c in [CORRECT] + ERROR_CATEGORIES},
    }


def compute_speech_rate(hyp, n_correct_ref):
    """Speaking time, rate, pauses, articulation rate, sliding-window timeline.

    ``rate`` is in the profile's units per minute (WPM for word, CPM for char):
    overall_rate uses ``n_correct_ref`` correctly-read reference units; the 30s
    window and articulation rate count hyp unit starts.
    """
    _, _, starts, ends, _ = hyp
    pairs = [(s, e) for s, e in zip(starts, ends)
             if s is not None and e is not None]
    out = {
        "speaking_time": 0.0, "minutes": 0.0, "overall_rate": 0.0,
        "articulation_rate": 0.0, "n_pauses": 0, "total_pause": 0.0,
        "longest_pause": 0.0, "n_hyp_units": len(pairs),
        "timeline": [], "pauses": [],
    }
    if len(pairs) < 1:
        return out
    first_start = pairs[0][0]
    last_end = pairs[-1][1]
    speaking_time = max(0.0, last_end - first_start)
    minutes = speaking_time / 60.0
    out["speaking_time"] = speaking_time
    out["minutes"] = minutes

    _, hd = hyp[0], hyp[1]
    pause_total, longest, n_pauses, pause_rows = 0.0, 0.0, 0, []
    for idx in range(1, len(pairs)):
        gap = pairs[idx][0] - pairs[idx - 1][1]
        if gap > PAUSE_GAP:
            n_pauses += 1
            pause_total += gap
            longest = max(longest, gap)
            pause_rows.append({
                "start": pairs[idx - 1][1], "end": pairs[idx][0],
                "duration": gap,
                "after_word": hd[idx - 1] if idx - 1 < len(hd) else "",
            })
    out["n_pauses"] = n_pauses
    out["total_pause"] = pause_total
    out["longest_pause"] = longest
    out["pauses"] = pause_rows

    out["overall_rate"] = (n_correct_ref / minutes) if minutes > 0 else 0.0
    artic_span = speaking_time - pause_total
    out["articulation_rate"] = (len(pairs) / (artic_span / 60.0)
                                if artic_span > 0 else 0.0)

    win, step = 30.0, 5.0
    timeline = []
    t = first_start
    while t < last_end:
        w_end = min(t + win, last_end)
        span = w_end - t
        n_in = sum(1 for s, _ in pairs if t <= s < w_end)
        if span >= 5.0:
            rate = n_in / (span / 60.0) if span > 0 else 0.0
            timeline.append((t, w_end, rate, n_in))
        t += step
    out["timeline"] = timeline
    return out


# --------------------------------------------------------------------------- #
# Focus words & sound groups
# --------------------------------------------------------------------------- #
def build_focus_words(profile, items):
    """Aggregate per reference word -> focus_words rows (DataFrame).

    Proper nouns are kept (with is_proper=True) so names stay visible, but the
    caller excludes them from the headline list and plot. Sound-group notes come
    from ``profile.sound_groups_for`` (empty for non-EN profiles).
    """
    agg = {}  # norm -> dict
    for it in items:
        if it["ref_pos"] is None:   # insertions/repetitions have no ref word
            continue
        norm = _normalize_parts(profile, it["ref_word"])
        norm = norm[0] if norm else (it["ref_word"] or "")
        d = agg.setdefault(norm, {
            "word": norm, "display": it["ref_word"], "occurrences": 0,
            "n_errors": 0, "types": {}, "confs": [],
            "is_proper": bool(it.get("is_proper")),
        })
        d["occurrences"] += 1
        d["is_proper"] = d["is_proper"] or bool(it.get("is_proper"))
        if it["confidence"] is not None:
            d["confs"].append(it["confidence"])
        if it["type"] != CORRECT:
            d["n_errors"] += 1
            d["types"][it["type"]] = d["types"].get(it["type"], 0) + 1

    rows = []
    for norm, d in agg.items():
        occ = d["occurrences"]
        confs = d["confs"]
        mean_c = sum(confs) / len(confs) if confs else float("nan")
        min_c = min(confs) if confs else float("nan")
        t = d["types"]
        sub_rate = t.get(SUBST, 0) / occ if occ else 0
        misp_rate = t.get(MISP, 0) / occ if occ else 0
        end_rate = t.get(ENDING, 0) / occ if occ else 0
        omit_rate = t.get(OMISSION, 0) / occ if occ else 0
        conf_pen = (1 - mean_c) if not math.isnan(mean_c) else 0.0
        focus_score = (math.log1p(occ) *
                       (1.0 * sub_rate + 0.8 * misp_rate +
                        0.5 * end_rate + 0.4 * omit_rate) + conf_pen)
        low_conf = (not math.isnan(mean_c)) and mean_c < LOW_CONF
        if d["n_errors"] == 0 and not low_conf:
            continue
        dom_type = max(t, key=t.get) if t else ""
        pg = profile.sound_groups_for(norm, norm)
        note = build_note(dom_type, pg)
        rows.append({
            "word": norm, "display": d["display"], "occurrences": occ,
            "n_errors": d["n_errors"],
            "error_types": ";".join(f"{k}:{v}" for k, v in sorted(t.items())),
            "mean_confidence": round(mean_c, 4) if not math.isnan(mean_c) else "",
            "min_confidence": round(min_c, 4) if not math.isnan(min_c) else "",
            "focus_score": round(focus_score, 4),
            "phoneme_group": ";".join(pg),
            "is_proper": bool(d["is_proper"]),
            "note": note,
            "_dom_type": dom_type,
        })
    df = pd.DataFrame(rows, columns=[
        "word", "display", "occurrences", "n_errors", "error_types",
        "mean_confidence", "min_confidence", "focus_score",
        "phoneme_group", "is_proper", "note", "_dom_type"])
    if not df.empty:
        df = df.sort_values("focus_score", ascending=False).reset_index(drop=True)
    return df


def build_note(dom_type, pg):
    parts = []
    if dom_type in ERROR_NOTE:
        parts.append(ERROR_NOTE[dom_type])
    for g in pg:
        if g in PHONEME_NOTES:
            parts.append(PHONEME_NOTES[g])
            break
    return " | ".join(parts) if parts else "review"


def build_phoneme_table(profile, items):
    """Per sound-group word/error counts -> DataFrame + err_rate dict.

    Returns an empty DataFrame (and {}) when the profile yields no sound groups,
    so non-EN sessions skip the phoneme outputs entirely.
    """
    stat = {}  # group -> stats; discovered lazily from the profile
    order = []
    for it in items:
        if it["ref_pos"] is None:
            continue
        norm = _normalize_parts(profile, it["ref_word"])
        norm = norm[0] if norm else ""
        if not norm:
            continue
        is_err = it["type"] != CORRECT
        for g in profile.sound_groups_for(norm, norm):
            if g not in stat:
                stat[g] = {"words": set(), "errs": 0, "confs": [], "examples": []}
                order.append(g)
            s = stat[g]
            s["words"].add(norm)
            if it["confidence"] is not None:
                s["confs"].append(it["confidence"])
            if is_err:
                s["errs"] += 1
                if norm not in s["examples"] and len(s["examples"]) < 5:
                    s["examples"].append(norm)
    rows, err_rates = [], {}
    for g in order:
        s = stat[g]
        nw = len(s["words"])
        ne = s["errs"]
        mc = sum(s["confs"]) / len(s["confs"]) if s["confs"] else float("nan")
        err_rates[g] = (ne / nw) if nw else 0.0
        rows.append({
            "group": g, "n_words": nw, "n_errors": ne,
            "mean_confidence": round(mc, 4) if not math.isnan(mc) else "",
            "example_words": ",".join(s["examples"]),
        })
    df = pd.DataFrame(rows, columns=[
        "group", "n_words", "n_errors", "mean_confidence", "example_words"])
    return df, err_rates


# --------------------------------------------------------------------------- #
# Paragraph & sentence analysis
# --------------------------------------------------------------------------- #
def build_paragraphs(profile, body, items, rate):
    """Per-paragraph accuracy/confidence/local rate -> DataFrame."""
    para_of_token, tok = [], 0
    for p_idx, para in enumerate(body.split("\n\n")):
        ntok = len(tokenize_text(profile, para)[0])
        for _ in range(ntok):
            para_of_token.append(p_idx)
            tok += 1
    by_ref = {}
    for it in items:
        if it["ref_pos"] is not None:
            by_ref[it["ref_pos"]] = it
    paras = {}
    for ref_pos, p_idx in enumerate(para_of_token):
        it = by_ref.get(ref_pos)
        d = paras.setdefault(p_idx, {"n": 0, "err": 0, "confs": [],
                                     "starts": [], "ends": []})
        d["n"] += 1
        if it:
            if it["type"] != CORRECT:
                d["err"] += 1
            if it["confidence"] is not None:
                d["confs"].append(it["confidence"])
            if it["time_start"] is not None:
                d["starts"].append(it["time_start"])
    rows = []
    para_texts = body.split("\n\n")
    for p_idx in sorted(paras):
        d = paras[p_idx]
        n = d["n"]
        acc = (n - d["err"]) / n if n else 0.0
        mc = sum(d["confs"]) / len(d["confs"]) if d["confs"] else float("nan")
        local_rate = 0.0
        if len(d["starts"]) >= 2:
            span = max(d["starts"]) - min(d["starts"])
            local_rate = n / (span / 60.0) if span > 0 else 0.0
        preview = " ".join(para_texts[p_idx].split()[:10]) if p_idx < len(para_texts) else ""
        rows.append({
            "para_idx": p_idx, "n_units": n, "n_errors": d["err"],
            "accuracy": round(acc, 4),
            "mean_confidence": round(mc, 4) if not math.isnan(mc) else "",
            "local_rate": round(local_rate, 1), "preview": preview,
        })
    return pd.DataFrame(rows, columns=[
        "para_idx", "n_units", "n_errors", "accuracy",
        "mean_confidence", "local_rate", "preview"])


def build_sentence_hotspots(profile, body, items, session_rate):
    """Split ref on .?! and rank sentences by error density."""
    by_ref = {it["ref_pos"]: it for it in items if it["ref_pos"] is not None}
    sentences = re.split(r"(?<=[.?!])\s+", body.replace("\n", " "))
    rows, tok = [], 0
    for s_idx, sent in enumerate(sentences):
        ntok = len(tokenize_text(profile, sent)[0])
        if ntok == 0:
            continue
        err, confs, starts = 0, [], []
        for k in range(ntok):
            it = by_ref.get(tok + k)
            if it:
                if it["type"] != CORRECT:
                    err += 1
                if it["confidence"] is not None:
                    confs.append(it["confidence"])
                if it["time_start"] is not None:
                    starts.append(it["time_start"])
        tok += ntok
        density = err / ntok if ntok else 0.0
        local_rate = 0.0
        if len(starts) >= 2:
            span = max(starts) - min(starts)
            local_rate = ntok / (span / 60.0) if span > 0 else 0.0
        deviates = (session_rate > 0 and local_rate > 0 and
                    abs(local_rate - session_rate) / session_rate > 0.25)
        rows.append({
            "sentence_idx": s_idx, "n_units": ntok, "n_errors": err,
            "error_density": round(density, 4),
            "local_rate": round(local_rate, 1),
            "rate_deviates": bool(deviates),
            "preview": " ".join(sent.split()[:14]),
        })
    df = pd.DataFrame(rows, columns=[
        "sentence_idx", "n_units", "n_errors", "error_density",
        "local_rate", "rate_deviates", "preview"])
    if not df.empty:
        df = df.sort_values("error_density", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #
def write_errors_csv(path, items):
    rows = [{
        "idx": i, "ref_word": it["ref_word"], "hyp_word": it["hyp_word"],
        "type": it["type"], "ref_pos": it["ref_pos"], "hyp_pos": it["hyp_pos"],
        "time_start": it["time_start"], "confidence": it["confidence"],
        "is_proper": bool(it.get("is_proper")),
    } for i, it in enumerate(it2 for it2 in items if it2["type"] != CORRECT)]
    pd.DataFrame(rows, columns=[
        "idx", "ref_word", "hyp_word", "type", "ref_pos", "hyp_pos",
        "time_start", "confidence", "is_proper"]).to_csv(path, index=False)


def write_summary_csv(path, label, profile, metrics, rate):
    row = {
        "chapter": label,
        "language": profile.code,
        "n_ref_units": metrics["n_ref"],
        "correct": metrics["correct"],
        "accuracy": round(metrics["accuracy"], 4),
        "wer": round(metrics["wer"], 4),
        "omissions": metrics[OMISSION],
        "insertions": metrics[INSERTION],
        "repetitions": metrics[REPETITION],
        "ending_mixups": metrics[ENDING],
        "mispronunciations": metrics[MISP],
        "substitutions": metrics[SUBST],
        "overall_rate": round(rate["overall_rate"], 1),
        "rate_unit": profile.rate_unit_label,
        "articulation_rate": round(rate["articulation_rate"], 1),
        "mean_confidence": (round(metrics["mean_confidence"], 4)
                            if not math.isnan(metrics["mean_confidence"]) else ""),
        "min_confidence": (round(metrics["min_confidence"], 4)
                           if not math.isnan(metrics["min_confidence"]) else ""),
        "n_pauses": rate["n_pauses"],
        "total_pause_s": round(rate["total_pause"], 2),
        "longest_pause_s": round(rate["longest_pause"], 2),
        "duration_min": round(rate["minutes"], 3),
        "n_hyp_units": rate["n_hyp_units"],
    }
    pd.DataFrame([row]).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Per-session plots
# --------------------------------------------------------------------------- #
def plot_rate_timeline(path, label, profile, rate):
    timeline = rate["timeline"]
    unit = profile.rate_unit_label
    lo, hi = profile.comfortable_rate
    fig, ax = plt.subplots(figsize=(10, 4))
    if timeline:
        xs = [(a + b) / 2 for a, b, _, _ in timeline]
        ys = [w for _, _, w, _ in timeline]
        ax.plot(xs, ys, "-o", ms=3, color="#1f77b4", label=f"{unit} (30s window)")
        mean_rate = sum(ys) / len(ys)
        ax.axhline(mean_rate, color="green", ls="--",
                   label=f"mean {mean_rate:.0f}")
        ax.axhspan(lo, hi, color="green", alpha=0.10,
                   label=f"comfortable {lo}-{hi}")
        for pr in rate["pauses"]:
            ax.axvline(pr["start"], color="red", alpha=0.25, lw=1)
    else:
        ax.text(0.5, 0.5, "no timing data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{unit} (units per minute)")
    ax.set_title(f"{unit} timeline - {label}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_error_breakdown(path, label, metrics):
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = ERROR_CATEGORIES
    vals = [metrics[c] for c in cats]
    ax.bar(cats, vals, color="#d62728")
    ax.set_ylabel("count")
    ax.set_title(f"Error breakdown - {label}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


_TYPE_COLOR = {SUBST: "#d62728", MISP: "#ff7f0e", ENDING: "#9467bd",
               OMISSION: "#7f7f7f", INSERTION: "#1f77b4",
               REPETITION: "#e377c2", "": "#2ca02c"}


def plot_focus_words(path, label, fdf):
    """Plot the top drillable focus words, EXCLUDING proper nouns (names)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    drill = fdf[~fdf["is_proper"]] if not fdf.empty else fdf
    if drill.empty:
        ax.text(0.5, 0.5, "no focus words - clean read!",
                ha="center", va="center", transform=ax.transAxes)
    else:
        top = drill.head(15).iloc[::-1]
        colors = [_TYPE_COLOR.get(t, "#333333") for t in top["_dom_type"]]
        ax.barh(top["word"], top["focus_score"], color=colors)
        for y, (_, r) in enumerate(top.iterrows()):
            mc = r["mean_confidence"]
            lbl = f"c={mc}" if mc != "" else "c=NA"
            ax.text(r["focus_score"], y, " " + lbl, va="center", fontsize=7)
    ax.set_xlabel("focus_score (higher = drill more)")
    ax.set_title(f"Top focus words - {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confidence_hist(path, label, items):
    confs = [it["confidence"] for it in items
             if it["confidence"] is not None and it["type"] != OMISSION]
    fig, ax = plt.subplots(figsize=(8, 4))
    if confs:
        ax.hist(confs, bins=20, color="#1f77b4", edgecolor="white")
        ax.axvline(LOW_CONF, color="red", ls="--",
                   label=f"low-confidence {LOW_CONF}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no confidence data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("per-unit confidence")
    ax.set_ylabel("count")
    ax.set_title(f"Confidence distribution - {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_phoneme_groups(path, label, pdf):
    fig, ax = plt.subplots(figsize=(8, 4))
    if pdf.empty or pdf["n_words"].sum() == 0:
        ax.text(0.5, 0.5, "no phoneme-group data", ha="center", va="center",
                transform=ax.transAxes)
    else:
        rates = [(r["n_errors"] / r["n_words"] if r["n_words"] else 0.0)
                 for _, r in pdf.iterrows()]
        ax.bar(pdf["group"], rates, color="#8c564b")
        ax.set_ylabel("error rate (errors / words)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(f"Phoneme-group error rate - {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Cumulative progress
# --------------------------------------------------------------------------- #
def upsert_session(label, profile, metrics, rate):
    """Append/replace this (chapter, language) row in the language-agnostic
    sessions.csv. Migrates a legacy (err_rate_* / n_ref_words / overall_wpm)
    schema on read so old rows survive the new columns.
    """
    config.ensure_dirs()
    row = {
        "date": datetime.date.today().isoformat(),
        "chapter": label,
        "language": profile.code,
        "n_ref_units": metrics["n_ref"],
        "accuracy": round(metrics["accuracy"], 4),
        "wer": round(metrics["wer"], 4),
        "omissions": metrics[OMISSION],
        "insertions": metrics[INSERTION],
        "repetitions": metrics[REPETITION],
        "ending_mixups": metrics[ENDING],
        "mispronunciations": metrics[MISP],
        "substitutions": metrics[SUBST],
        "overall_rate": round(rate["overall_rate"], 1),
        "articulation_rate": round(rate["articulation_rate"], 1),
        "mean_confidence": (round(metrics["mean_confidence"], 4)
                            if not math.isnan(metrics["mean_confidence"]) else ""),
        "n_pauses": rate["n_pauses"],
        "duration_min": round(rate["minutes"], 3),
    }

    if config.SESSIONS_CSV.exists():
        try:
            existing = pd.read_csv(config.SESSIONS_CSV)
        except Exception:
            existing = pd.DataFrame(columns=SESSION_COLUMNS)
        existing = _migrate_sessions(existing)
        mask = ~((existing["chapter"] == label) &
                 (existing["language"] == profile.code))
        existing = existing[mask]
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.reindex(columns=SESSION_COLUMNS)
    tmp = config.SESSIONS_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, config.SESSIONS_CSV)
    return df


def _migrate_sessions(df):
    """Bring an old-schema sessions.csv up to SESSION_COLUMNS in place."""
    if df.empty:
        return df.reindex(columns=SESSION_COLUMNS)
    rename = {}
    if "overall_wpm" in df.columns and "overall_rate" not in df.columns:
        rename["overall_wpm"] = "overall_rate"
    if "n_ref_words" in df.columns and "n_ref_units" not in df.columns:
        rename["n_ref_words"] = "n_ref_units"
    if rename:
        df = df.rename(columns=rename)
    drop = [c for c in df.columns if c.startswith("err_rate_")]
    if drop:
        df = df.drop(columns=drop)
    if "language" not in df.columns:
        df["language"] = "en"     # legacy rows were English-only
    return df


def _rolling(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()


def plot_progress(df, profile):
    if df.empty:
        return
    df = df.sort_values("date").reset_index(drop=True)
    x = list(range(len(df)))
    xlabels = [f"{d}\n{c[:18]}" for d, c in zip(df["date"], df["chapter"])]

    # accuracy & WER
    fig, ax = plt.subplots(figsize=(10, 4))
    acc = pd.to_numeric(df["accuracy"], errors="coerce")
    wer = pd.to_numeric(df["wer"], errors="coerce")
    ax.plot(x, acc, "-o", color="green", label="accuracy")
    ax.plot(x, wer, "-o", color="red", label="WER")
    ax.plot(x, _rolling(acc), "--", color="green", alpha=0.5, label="acc roll(3)")
    ax.plot(x, _rolling(wer), "--", color="red", alpha=0.5, label="wer roll(3)")
    if len(df) >= 2:
        ax.annotate(f"acc {acc.iloc[0]:.2f}->{acc.iloc[-1]:.2f}",
                    xy=(x[-1], acc.iloc[-1]), fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=0)
    ax.set_ylabel("rate")
    ax.set_title("Progress: accuracy & WER over sessions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PROGRESS_DIR / "progress_accuracy_wer.png", dpi=120)
    plt.close(fig)

    # rate + articulation
    unit = profile.rate_unit_label
    lo, hi = profile.comfortable_rate
    fig, ax = plt.subplots(figsize=(10, 4))
    rate = pd.to_numeric(df["overall_rate"], errors="coerce")
    art = pd.to_numeric(df["articulation_rate"], errors="coerce")
    ax.axhspan(lo, hi, color="green", alpha=0.10,
               label=f"comfortable {lo}-{hi}")
    ax.plot(x, rate, "-o", color="#1f77b4", label=f"overall {unit}")
    ax.plot(x, art, "-o", color="#ff7f0e", label="articulation rate")
    ax.plot(x, _rolling(rate), "--", color="#1f77b4", alpha=0.5,
            label=f"{unit} roll(3)")
    if len(df) >= 2:
        ax.annotate(f"{unit} {rate.iloc[0]:.0f}->{rate.iloc[-1]:.0f}",
                    xy=(x[-1], rate.iloc[-1]), fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel(f"{unit} (units per minute)")
    ax.set_title("Progress: speaking rate over sessions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.PROGRESS_DIR / "progress_wpm.png", dpi=120)
    plt.close(fig)

    # stacked errors
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = [0.0] * len(df)
    cats = ["omissions", "insertions", "repetitions", "ending_mixups",
            "mispronunciations", "substitutions"]
    palette = ["#7f7f7f", "#1f77b4", "#e377c2", "#9467bd", "#ff7f0e", "#d62728"]
    for cat, col in zip(cats, palette):
        vals = pd.to_numeric(df[cat], errors="coerce").fillna(0).tolist()
        ax.bar(x, vals, bottom=bottom, label=cat, color=col)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("error count")
    ax.set_title("Progress: error categories per session")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(config.PROGRESS_DIR / "progress_errors.png", dpi=120)
    plt.close(fig)


def plot_progress_phoneme_groups():
    """Per-group error-rate trend, rebuilt by re-reading every session's
    phoneme_groups.csv (sound-group rates are no longer stored in sessions.csv).

    Aggregates all chapters' EN csvs regardless of the current run's language;
    that is fine since non-EN sessions never write that csv. Skips silently if
    no phoneme_groups.csv exists.
    """
    glob = sorted(config.REPORTS_DIR.glob("*/phoneme_groups.csv"))
    series = {}  # group -> list of (key, err_rate)
    for pcsv in glob:
        try:
            d = pd.read_csv(pcsv)
        except Exception:
            continue
        if d.empty or "group" not in d.columns:
            continue
        key = pcsv.parent.name
        for _, r in d.iterrows():
            nw = float(r.get("n_words", 0) or 0)
            ne = float(r.get("n_errors", 0) or 0)
            rate = (ne / nw) if nw else 0.0
            series.setdefault(str(r["group"]), []).append((key, rate))
    if not series:
        return
    keys = sorted({k for pts in series.values() for k, _ in pts})
    xidx = {k: i for i, k in enumerate(keys)}
    fig, ax = plt.subplots(figsize=(10, 4))
    for g, pts in series.items():
        pts = sorted(pts, key=lambda kv: xidx[kv[0]])
        xs = [xidx[k] for k, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, "-o", ms=3, label=g)
    ax.set_xticks(list(range(len(keys))))
    ax.set_xticklabels([k[:18] for k in keys], fontsize=7, rotation=0)
    ax.set_ylabel("error rate")
    ax.set_title("Progress: phoneme-group error rate trend")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(config.PROGRESS_DIR / "progress_phoneme_groups.png", dpi=120)
    plt.close(fig)


def plot_progress_focus_words():
    """Aggregate top recurring focus words across per-session focus_words.csv,
    excluding proper nouns (names should not become recurring drills)."""
    agg = {}  # word -> {score, sessions, group}
    for fcsv in config.REPORTS_DIR.glob("*/focus_words.csv"):
        try:
            d = pd.read_csv(fcsv)
        except Exception:
            continue
        if d.empty or "word" not in d.columns:
            continue
        for _, r in d.iterrows():
            if "is_proper" in d.columns and bool(r.get("is_proper")):
                continue
            w = str(r["word"])
            e = agg.setdefault(w, {"score": 0.0, "sessions": 0, "group": ""})
            e["score"] += float(r.get("focus_score", 0) or 0)
            e["sessions"] += 1
            grp = r.get("phoneme_group")
            if not e["group"] and isinstance(grp, str):
                e["group"] = grp.split(";")[0]
    fig, ax = plt.subplots(figsize=(9, 6))
    if not agg:
        ax.text(0.5, 0.5, "no recurring focus words yet",
                ha="center", va="center", transform=ax.transAxes)
    else:
        items = sorted(agg.items(),
                       key=lambda kv: (kv[1]["sessions"], kv[1]["score"]),
                       reverse=True)[:15][::-1]
        words = [w for w, _ in items]
        scores = [v["score"] for _, v in items]
        colors = ["#8c564b" if v["group"] else "#1f77b4" for _, v in items]
        ax.barh(words, scores, color=colors)
        for y, (w, v) in enumerate(items):
            ax.text(scores[y], y, f" x{v['sessions']}", va="center", fontsize=7)
    ax.set_xlabel("cumulative focus_score")
    ax.set_title("Progress: top recurring focus words (drill these)")
    fig.tight_layout()
    fig.savefig(config.PROGRESS_DIR / "progress_focus_words.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv):
    p = argparse.ArgumentParser(description="Analyze a read-aloud session.")
    p.add_argument("chapter", nargs="?", default=None,
                   help="chapter label (TOC heading); optional if "
                        "--reference/--transcript/--label are given")
    p.add_argument("--language", "-l", default=config.DEFAULT_LANGUAGE,
                   help="analysis language profile (en, generic, ja, ...); "
                        f"default {config.DEFAULT_LANGUAGE}")
    p.add_argument("--reference", help="override reference .txt path")
    p.add_argument("--transcript", help="override transcript .json path")
    p.add_argument("--label", help="override label for output dirs/progress key")
    p.add_argument("--no-progress", action="store_true",
                   help="do not append/update the cumulative sessions row")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    profile = languages.get_profile(args.language)

    if args.reference is None and args.chapter is None:
        sys.exit("ERROR: provide a chapter label (or --reference) to locate "
                 "the reference text.")
    label = args.label or args.chapter
    if label is None:
        sys.exit("ERROR: provide a chapter label (or --label) for output dirs.")

    ref_path = (config.Path(args.reference) if args.reference
                else config.chapter_txt(args.chapter))
    title, ref, body = load_reference(profile, ref_path)

    hyp_path = resolve_transcript_path(args, args.chapter)
    hyp = load_transcript(profile, hyp_path)

    rdir = config.report_dir(label)
    rdir.mkdir(parents=True, exist_ok=True)

    n_ref = len(ref[0])
    items = align_and_classify(profile, ref, hyp)
    metrics = compute_metrics(items, n_ref)
    n_correct_ref = metrics["correct"] + metrics[ENDING] + metrics[MISP] + metrics[SUBST]
    rate = compute_speech_rate(hyp, n_correct_ref)

    fdf = build_focus_words(profile, items)
    pdf, err_rates = build_phoneme_table(profile, items)
    has_sound_groups = not pdf.empty
    para_df = build_paragraphs(profile, body, items, rate)
    sent_df = build_sentence_hotspots(profile, body, items, rate["overall_rate"])

    # ---- CSV reports ----
    write_errors_csv(rdir / "errors.csv", items)
    fdf.drop(columns=["_dom_type"]).to_csv(rdir / "focus_words.csv", index=False)
    pd.DataFrame(rate["timeline"],
                 columns=["t_start", "t_end", "rate", "n_units"]
                 ).to_csv(rdir / "wpm_timeline.csv", index=False)
    pd.DataFrame(rate["pauses"],
                 columns=["start", "end", "duration", "after_word"]
                 ).to_csv(rdir / "pauses.csv", index=False)
    write_summary_csv(rdir / "summary.csv", label, profile, metrics, rate)
    if has_sound_groups:
        pdf.to_csv(rdir / "phoneme_groups.csv", index=False)
    para_df.to_csv(rdir / "paragraphs.csv", index=False)
    sent_df.to_csv(rdir / "sentence_hotspots.csv", index=False)

    # ---- PNG diagrams (skip heavy plots only if no scored tokens) ----
    if n_ref > 0:
        plot_rate_timeline(rdir / "wpm_timeline.png", label, profile, rate)
        plot_error_breakdown(rdir / "error_breakdown.png", label, metrics)
        plot_focus_words(rdir / "focus_words.png", label, fdf)
        plot_confidence_hist(rdir / "confidence_hist.png", label, items)
        if has_sound_groups:
            plot_phoneme_groups(rdir / "phoneme_groups.png", label, pdf)

    # ---- progress ----
    if not args.no_progress:
        sdf = upsert_session(label, profile, metrics, rate)
        plot_progress(sdf, profile)
        plot_progress_focus_words()
        if has_sound_groups:
            plot_progress_phoneme_groups()

    # ---- console summary ----
    drill = fdf[~fdf["is_proper"]] if not fdf.empty else fdf
    top5 = drill.head(5)["word"].tolist() if not drill.empty else []
    top_groups = (pdf.sort_values("n_errors", ascending=False).head(3)["group"].tolist()
                  if not pdf.empty else [])
    worst_sent = sent_df.iloc[0]["preview"] if not sent_df.empty else "(none)"
    mc = metrics["mean_confidence"]
    unit = profile.rate_unit_label
    print(f"== {label} ({profile.name}) ==")
    print(f"scored ref {profile.unit}s : {n_ref}")
    print(f"accuracy         : {metrics['accuracy']*100:.1f}%")
    print(f"WER              : {metrics['wer']*100:.1f}%")
    print(f"overall {unit:<8} : {rate['overall_rate']:.0f} "
          f"(articulation {rate['articulation_rate']:.0f})")
    print(f"mean confidence  : {('%.3f' % mc) if not math.isnan(mc) else 'n/a'}")
    print(f"pauses           : {rate['n_pauses']} "
          f"(longest {rate['longest_pause']:.1f}s)")
    print(f"top focus words  : {', '.join(top5) if top5 else '(none)'}")
    print(f"top sound groups : {', '.join(top_groups) if top_groups else '(none)'}")
    print(f"worst sentence   : {worst_sent}")
    print(f"output dir       : {rdir}")


if __name__ == "__main__":
    main()
