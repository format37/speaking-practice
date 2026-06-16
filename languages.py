"""Language-pluggable analysis profiles for the read-aloud toolkit.

This module isolates every language-specific decision behind a small, stable
``LanguageProfile`` API so ``analyze.py`` can stay language-agnostic. Three
profiles ship here:

* ``en``      - reproduces the verified English behavior verbatim (inflectional
                suffix set, confusable pairs, Russian-L1 phoneme buckets).
* ``generic`` - whitespace-tokenized, casefolding, number-normalizing,
                proper-noun-aware language with only generic heuristics (no
                English-specific suffix/confusable/phoneme knowledge).
* ``ja``      - character-level analysis (also used for ``zh``); no proper
                nouns, no number normalization, no phoneme groups.

Standard library only (``difflib`` for similarity ratios).

Public API consumed by analyze.py:
    get_profile(code) -> LanguageProfile
    PROFILES: dict[str, LanguageProfile]
    LanguageProfile.{code,name,unit,rate_unit_label,comfortable_rate,
                     casefold,normalize_numbers,detect_proper_nouns}
    LanguageProfile.normalize(token) -> str
    LanguageProfile.tokenize(text) -> list[str]
    LanguageProfile.split_hypothesis(words) -> list[dict]
    LanguageProfile.is_ending_mixup(ref_norm, hyp_norm) -> bool
    LanguageProfile.classify_replace_pair(ref_norm, hyp_norm) -> str
    LanguageProfile.sound_groups_for(ref_norm, hyp_norm) -> list[str]
"""
import re
import unicodedata
from difflib import SequenceMatcher

# --------------------------------------------------------------------------- #
# Error-category labels (kept identical to analyze.py semantics)
# --------------------------------------------------------------------------- #
ENDING = "ending_mixup"
MISP = "mispronunciation"
SUBST = "substitution"

# --------------------------------------------------------------------------- #
# Shared normalization pieces
# --------------------------------------------------------------------------- #
_CURLY = str.maketrans({"‘": "'", "’": "'", "“": '"',
                        "”": '"', "–": "-", "—": "-"})


# --------------------------------------------------------------------------- #
# Integer -> number-words helper (0..9999)
# --------------------------------------------------------------------------- #
# Joined with NO spaces / NO hyphens so the result is a single normalized token
# (normalize() runs after this and would otherwise re-split a hyphen and break
# the one-display-token -> norm-tokens index alignment in tokenize_text).
_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)


def int_to_words(n):
    """Return a single hyphen/space-free English word form for 0 <= n <= 9999.

    Returns None for values outside that range so callers can leave the digits
    as-is.
    """
    if n < 0 or n > 9999:
        return None
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (_ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + "hundred" + (int_to_words(rest) if rest else "")
    rest = n % 1000
    return _ONES[n // 1000] + "thousand" + (int_to_words(rest) if rest else "")


def _normalize_number_token(token):
    """If token is pure digits in range, map to its number word; else None.

    Also handles a pure-digit run followed by a possessive/plural suffix
    (``'s``, ``s``) so e.g. "2's" -> "twos" and "20s" -> "twentys" instead of
    surviving as a digit-bearing focus word. The apostrophe is dropped and the
    trailing ``s`` re-attached, keeping the result a single normalized token.
    """
    if re.fullmatch(r"\d+", token):
        return int_to_words(int(token))
    m = re.fullmatch(r"(\d+)'?s", token)
    if m:
        words = int_to_words(int(m.group(1)))
        if words is not None:
            return words + "s"
    return None


# --------------------------------------------------------------------------- #
# en-specific knowledge (verbatim from the verified English behavior)
# --------------------------------------------------------------------------- #
INFLECTIONAL_SUFFIXES = {
    "s", "es", "ed", "d", "ing", "en", "er", "est",
    "ies", "ier", "iest", "ly", "'s", "'ll", "'d", "'ve", "'re", "n't",
}
CONFUSABLES = {
    frozenset({"their", "there", "theyre", "they're"}),
    frozenset({"to", "too", "two"}),
    frozenset({"of", "off"}),
    frozenset({"then", "than"}),
    frozenset({"your", "youre", "you're"}),
}
# Russian-L1 phoneme groups (regex on normalized word)
PHONEME_GROUPS = [
    ("th", re.compile(r"th")),
    ("w_v", re.compile(r"[wv]")),
    ("r", re.compile(r"r$|[aeiou]r")),
    ("final_voiced", re.compile(r"[bdgvz]$")),
    ("final_cluster", re.compile(r"[bcdfghjklmnpqrstvwxz]{2}$")),
    ("h_x", re.compile(r"h[aeiou]")),
    ("ng", re.compile(r"ng$|nk")),
    ("vowel_len", re.compile(r"ee|oo")),
]


def _common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# --------------------------------------------------------------------------- #
# LanguageProfile
# --------------------------------------------------------------------------- #
class LanguageProfile:
    """Encapsulates one language's normalization, tokenization, and
    classification decisions. Instances are configured via flags plus optional
    pluggable strategy callables; defaults give the ``generic`` behavior."""

    def __init__(
        self,
        code,
        name,
        *,
        unit="word",
        rate_unit_label="WPM",
        comfortable_rate=(130, 160),
        casefold=True,
        normalize_numbers=True,
        detect_proper_nouns=True,
        suffixes=None,
        confusables=None,
        phoneme_groups=None,
        ending_mode="generic",
    ):
        self.code = code
        self.name = name
        self.unit = unit                      # "word" | "char"
        self.rate_unit_label = rate_unit_label
        self.comfortable_rate = comfortable_rate
        self.casefold = casefold
        self.normalize_numbers = normalize_numbers
        self.detect_proper_nouns = detect_proper_nouns
        self._suffixes = suffixes or set()
        self._confusables = confusables or set()
        self._phoneme_groups = phoneme_groups or []
        # "en"   -> stem>=3 & both tails in suffix set
        # "generic" -> common-prefix-ratio>=0.7 & small tail edit
        # "none" -> always False (char/ja)
        self._ending_mode = ending_mode

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def normalize(self, token):
        """Normalize a single raw display token to its primary norm form.

        Casefolds (if set); maps curly quotes/dashes to ASCII; optionally maps a
        pure-digit token to a canonical number word; strips punctuation while
        keeping intra-word apostrophes.

        Returns "" when nothing significant remains. For word profiles this is
        the FIRST normalized part of a compound; analyze.py splits compounds via
        ``tokenize`` for full alignment, but ``normalize`` answers "what does
        this token reduce to" for single-token comparisons.
        """
        w = token.translate(_CURLY)
        if self.casefold:
            w = w.lower()
        if self.normalize_numbers:
            nw = _normalize_number_token(w)
            if nw is not None:
                return nw
        parts = self._split_compound(w)
        for p in parts:
            cleaned = self._finalize(self._strip_token(p))
            if cleaned:
                return cleaned
        return ""

    def _split_compound(self, w):
        """Split hyphen/slash compounds for word profiles; no-op for char."""
        if self.unit == "char":
            return [w]
        return re.split(r"[-/]+", w)

    def _strip_token(self, p):
        """Strip punctuation but keep intra-word apostrophes; drop edge ones."""
        if self.unit == "char":
            # keep all significant (non-space, non-punct) characters
            return "".join(
                ch for ch in p
                if not ch.isspace() and not _is_punct(ch)
            )
        p = re.sub(r"[^a-z0-9']", "", p) if self.casefold else \
            re.sub(r"[^A-Za-z0-9']", "", p)
        return p.strip("'")

    def _finalize(self, cleaned):
        """Number-normalize a token that became a digit run only after its
        attached punctuation was stripped (e.g. "2," -> "2" -> "two",
        "2's" -> "twos"). No-op for non-digit tokens and char profiles, so
        non-numeric normalization stays byte-identical."""
        if cleaned and self.normalize_numbers:
            nw = _normalize_number_token(cleaned)
            if nw is not None:
                return nw
        return cleaned

    def _normalize_parts(self, token):
        """Return ALL normalized parts of a single raw token (compound-split)."""
        w = token.translate(_CURLY)
        if self.casefold:
            w = w.lower()
        if self.normalize_numbers:
            nw = _normalize_number_token(w)
            if nw is not None:
                return [nw]
        if self.unit == "char":
            out = []
            for ch in w:
                if not ch.isspace() and not _is_punct(ch):
                    out.append(ch)
            return out
        out = []
        for p in re.split(r"[-/]+", w):
            cleaned = self._finalize(self._strip_token(p))
            if cleaned:
                out.append(cleaned)
        return out

    # ------------------------------------------------------------------ #
    # Tokenization
    # ------------------------------------------------------------------ #
    def tokenize(self, text):
        """Tokenize plain text into NORMALIZED tokens.

        word unit: whitespace split, each split into its compound parts.
        char unit: significant characters (whitespace/punctuation dropped).
        """
        out = []
        if self.unit == "char":
            for ch in text.translate(_CURLY):
                if ch.isspace() or _is_punct(ch):
                    continue
                out.append(ch.lower() if self.casefold else ch)
            return out
        for raw in text.split():
            out.extend(self._normalize_parts(raw))
        return out

    # ------------------------------------------------------------------ #
    # Hypothesis splitting (Deepgram words -> norm units with timing)
    # ------------------------------------------------------------------ #
    def split_hypothesis(self, words):
        """Explode Deepgram word dicts into normalized units with timing.

        Each input ``w`` is a dict with at least one of
        ``punctuated_word``/``word`` plus ``start``/``end``/``confidence``.

        Returns a list of dicts: {norm, display, start, end, conf}.
        word unit: one output per normalized part of the word's display text.
        char unit: one output per significant character, all sharing the parent
        word's start/end/confidence (no synthesized intra-word sub-timings).
        """
        out = []
        for w in words:
            raw = str(w.get("punctuated_word") or w.get("word") or "")
            start = w.get("start")
            end = w.get("end")
            conf = w.get("confidence")
            char_unit = self.unit == "char"
            for n in self._normalize_parts(raw):
                out.append({
                    "norm": n, "display": n if char_unit else raw,
                    "start": start, "end": end, "conf": conf,
                })
        return out

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def is_ending_mixup(self, ref_norm, hyp_norm):
        """Whether ref/hyp differ only by an inflectional ending."""
        a, b = ref_norm, hyp_norm
        if a == b:
            return False
        if self._ending_mode == "none":
            return False
        if self._ending_mode == "en":
            cp = _common_prefix_len(a, b)
            if cp < 3:
                return False
            ta, tb = a[cp:], b[cp:]
            if ta == tb:
                return False
            ta_ok = (ta == "") or (ta in self._suffixes)
            tb_ok = (tb == "") or (tb in self._suffixes)
            return ta_ok and tb_ok
        # generic: shared prefix is most of the shorter word, small tail edit
        cp = _common_prefix_len(a, b)
        minlen = min(len(a), len(b))
        if minlen == 0:
            return False
        if cp / minlen < 0.7:
            return False
        ta, tb = a[cp:], b[cp:]
        return max(len(ta), len(tb)) <= 3

    def _both_in_confusable(self, a, b):
        sa = a.replace("'", "")
        sb = b.replace("'", "")
        for grp in self._confusables:
            cg = {x.replace("'", "") for x in grp}
            if sa in cg and sb in cg:
                return True
        return False

    def classify_replace_pair(self, ref_norm, hyp_norm):
        """Classify a ref/hyp replace pair into ending/misp/subst."""
        a, b = ref_norm, hyp_norm
        if self._both_in_confusable(a, b):
            return MISP
        if self.is_ending_mixup(a, b):
            return ENDING
        ratio = SequenceMatcher(None, a, b).ratio()
        minlen = min(len(a), len(b))
        if ratio >= 0.65 and minlen >= 4:
            return MISP
        return SUBST

    def sound_groups_for(self, ref_norm, hyp_norm):
        """Return phoneme/sound group tags implicated by a ref token.

        Mirrors the original phoneme_groups_for(): groups are matched against
        the REFERENCE normalized token. Empty for generic/char profiles.
        """
        if not self._phoneme_groups:
            return []
        return [g for g, rx in self._phoneme_groups if rx.search(ref_norm)]


# --------------------------------------------------------------------------- #
# Punctuation test (used by char/generic stripping)
# --------------------------------------------------------------------------- #
def _is_punct(ch):
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


# --------------------------------------------------------------------------- #
# Concrete profiles
# --------------------------------------------------------------------------- #
EN_PROFILE = LanguageProfile(
    code="en",
    name="English",
    unit="word",
    rate_unit_label="WPM",
    comfortable_rate=(130, 160),
    casefold=True,
    normalize_numbers=True,
    detect_proper_nouns=True,
    suffixes=INFLECTIONAL_SUFFIXES,
    confusables=CONFUSABLES,
    phoneme_groups=PHONEME_GROUPS,
    ending_mode="en",
)

GENERIC_PROFILE = LanguageProfile(
    code="generic",
    name="Generic",
    unit="word",
    rate_unit_label="WPM",
    comfortable_rate=(130, 160),
    casefold=True,
    normalize_numbers=True,
    detect_proper_nouns=True,
    suffixes=set(),
    confusables=set(),
    phoneme_groups=[],
    ending_mode="generic",
)

JA_PROFILE = LanguageProfile(
    code="ja",
    name="Japanese",
    unit="char",
    rate_unit_label="CPM",
    comfortable_rate=(250, 400),
    casefold=False,
    normalize_numbers=False,
    detect_proper_nouns=False,
    suffixes=set(),
    confusables=set(),
    phoneme_groups=[],
    ending_mode="none",
)

ZH_PROFILE = LanguageProfile(
    code="zh",
    name="Chinese",
    unit="char",
    rate_unit_label="CPM",
    comfortable_rate=(250, 400),
    casefold=False,
    normalize_numbers=False,
    detect_proper_nouns=False,
    suffixes=set(),
    confusables=set(),
    phoneme_groups=[],
    ending_mode="none",
)

PROFILES = {
    "en": EN_PROFILE,
    "generic": GENERIC_PROFILE,
    "ja": JA_PROFILE,
    "zh": ZH_PROFILE,
}


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def get_profile(code):
    """Return the LanguageProfile for ``code`` (case-insensitive).

    'en' -> English; 'ja'/'zh' -> character-level CJK profile. Any unknown code
    falls back to the GENERIC profile, but carries the requested code so the
    downstream Deepgram language code is still that requested language.
    """
    if not code:
        return EN_PROFILE
    key = str(code).strip().lower()
    if key in PROFILES:
        return PROFILES[key]
    # Unknown: generic behavior, but preserve the requested Deepgram code.
    fallback = LanguageProfile(
        code=key,
        name=key,
        unit="word",
        rate_unit_label="WPM",
        comfortable_rate=(130, 160),
        casefold=True,
        normalize_numbers=True,
        detect_proper_nouns=True,
        suffixes=set(),
        confusables=set(),
        phoneme_groups=[],
        ending_mode="generic",
    )
    return fallback
