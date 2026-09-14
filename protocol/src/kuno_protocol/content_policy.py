"""The shared prompt content policy: one deterministic list, enforced by the gateway and the worker.

KunoWorld bans all sexual content (NSFW) in both Private and Standard mode: pornography, nudity,
sexual acts, fetish content, sexualised depictions and erotic roleplay, and above all sexual
content involving minors and sexual deepfakes of real people. The gateway runs `check_prompt`
wherever it can read a prompt (Standard mode); the worker runs the same function inside the
enclave for every job, so both enforce exactly the same list. In Private mode nobody outside
the enclave sees the prompt, so the enclave's copy is the only one that runs.

This module is deliberately dependency-free (standard library only) so that the gateway,
the worker and the validators can all import it.

    check_prompt(prompt, negative_prompt=None) -> None      raises ContentPolicyViolation(category)
    mentions_minor(prompt) -> bool                          the prompt names a child or teenager

Categories (`ContentPolicyViolation.category`, for counters and strikes only):
    sexual_minors    sexual content involving minors, including co-occurrence and absolute terms
    sexual_deepfake  sexual content together with deepfake / face-swap / real-person / "nudify" terms
    sexual           every other sexual, nude, pornographic, fetish or erotic request

The message never varies, so nothing about the prompt can leak through an exception, a log line
or an API error.

How matching works
------------------
1. Normalisation (`normalize_tokens`). NFKC, then zero-width and other format characters are
   removed, casefolded, Cyrillic/Greek/small-capital homoglyphs mapped to Latin, accents stripped,
   leetspeak digits and symbols folded (0->o, 1->i, 3->e, 4->a, 5->s, @->a, $->s, ...), and ages
   ("12yo", "13-year-old", "12 years old") under 18 become a minor marker. Runs of three or more
   single letters are merged ("n u d e", "n.u.d.e", "n🍑u🍑d🍑e" -> "nude").
2. Matching (`_TermSet`). A term matches a single token exactly, a whole-token phrase
   ("in the buff"), or up to four adjacent tokens joined ("nu de", "ch ild"). A join never
   includes a common short word, so "the pen is red" does not become "penis". Repeated letters
   are squeezed only when the text has them ("nuuude" -> "nude", but "bobs" stays "bobs").
   Some stems match as prefixes ("porn", "erotic", "masturbat", ...).
3. Allow-context (`AMBIGUOUS`). A handful of words are sexual in most prompts but ordinary in a
   few fixed phrases: "breast cancer awareness", "chicken breast", "nude colour palette",
   "the naked eye", "biological sex of a bird", "same-sex couple", "blue tits at a feeder",
   "sexy sports car". Such a word is allowed only when one of its context words sits within
   two content words of it, on the named side, with no person word or other sexual term in
   between ("nude woman palette" still blocks). Everything else about the prompt is still
   checked, so "chicken breast, naked woman" blocks on "naked".
4. Rules, in order:
   - an absolute term (CSAM phrases, "jailbait", "lolicon", ...) in the prompt or the negative
     prompt -> sexual_minors;
   - any sexual term in the prompt, or a removal verb next to clothing ("take off her clothes",
     "without clothes"), or a nudify term: within 12 tokens of a minor term -> sexual_minors;
     with a deepfake or real-person term anywhere -> sexual_deepfake; otherwise -> sexual;
   - a minor term within 12 tokens of a word that is only unacceptable next to a minor
     ("explicit", "provocative", "suggestive") -> sexual_minors;
   - clothing or "censored"/"sfw" in the negative prompt (it steers the model toward nudity)
     -> sexual_minors if the prompt names a minor, otherwise sexual.
   Sexual words in the negative prompt are otherwise fine: "nsfw, nude" there asks the model
   to avoid them.

Trade-offs
----------
- This is a first line, not the whole defence. It is English-centric (with a few common
  Spanish/German terms and algospeak), catches known phrasings rather than intent, and cannot
  see images, names of real people or euphemisms it does not list. The worker backs it with a
  prompt classifier (Qwen3Guard, multilingual) and frame classifiers over the rendered video.
- Known false positives, accepted: "the explicit" is fine but "explicit ... kids" blocks;
  "seductive/sensual" outside the few listed product contexts block (a perfume ad passes, a
  "sensual dance" does not); "orgy of colour" blocks; "without clothing brands visible" in the
  prompt blocks; clothing words in a negative prompt block even for a landscape; "breastfeeding
  mother" passes but "breast" near a person word without context blocks. Classical nude art
  blocks: nudity is banned regardless of artistic intent.
- Known false negatives, accepted: allow-context can be abused by a prompt whose only sexual
  word is an ambiguous one placed in a listed phrase ("naked eye-catching ..."); words split so
  that one piece is a common short word ("t its"); misspellings beyond repeated letters; non-English and coded
  language. The classifiers and the frame check exist for these.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "CATEGORIES",
    "ContentPolicyViolation",
    "POLICY_MESSAGE",
    "check_prompt",
    "mentions_minor",
    "normalize_tokens",
]

POLICY_MESSAGE = "request violates the acceptable use policy"
CATEGORIES = ("sexual_minors", "sexual_deepfake", "sexual")


class ContentPolicyViolation(Exception):
    """The request breaks the content policy.

    `category` is one of CATEGORIES, for tests, aggregate counters and account strikes. The
    message is fixed: it never contains or depends on the prompt.
    """

    def __init__(self, category: str = "policy"):
        super().__init__(POLICY_MESSAGE)
        self.category = category


# ---------------------------------------------------------------- normalisation

_CONFUSABLES = str.maketrans(
    {
        # Cyrillic and Greek letters that render like Latin ones.
        "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "ё": "e", "һ": "h", "н": "h", "і": "i", "ї": "i",
        "ј": "j", "к": "k", "м": "m", "о": "o", "р": "p", "ԛ": "q", "ѕ": "s", "т": "t", "у": "y", "х": "x",
        "п": "n", "г": "r", "ԝ": "w", "ү": "y", "є": "e", "ѵ": "v", "ӏ": "l",
        "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p", "τ": "t",
        "υ": "u", "χ": "x", "ω": "w", "ɡ": "g", "ı": "i", "ł": "l", "ø": "o", "đ": "d", "ß": "ss", "ɑ": "a",
        # Small capitals, which NFKC leaves alone.
        "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ɢ": "g", "ʜ": "h", "ɪ": "i", "ᴊ": "j", "ᴋ": "k",
        "ʟ": "l", "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ʀ": "r", "ꜱ": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v",
        "ᴡ": "w", "ʏ": "y", "ᴢ": "z",
    }
)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
                       "@": "a", "$": "s", "!": "i", "|": "l", "+": "t"})
_SPLIT = re.compile(r"[^\w@$!|+]+|_+")
_AGE = re.compile(r"^(\d{1,2})(?:yo|yr|yrs|y|year|years|yearold|yearsold)?$")
_AGE_UNITS = {"yo", "y", "yr", "yrs", "year", "years", "yearold", "yearsold", "old"}
MINOR_TOKEN = "\x00minor"


def _collapse(word: str) -> str:
    """Squeezes repeated letters ("nuuude" -> "nude")."""
    return re.sub(r"(.)\1+", r"\1", word)


def normalize_tokens(text: str) -> list[str]:
    """Folds text into comparable tokens. Used for matching only; never log the result."""
    return _token_variants(text)[0]


def _token_variants(text: str) -> list[list[str]]:
    """The tokens, plus a second reading when a spelled-out run starts with a one-letter word.

    "a n.u.d.e woman" merges to "anude" in the first reading; the second keeps "a" apart and so
    finds "nude". Both are checked, so "a.h.e.g.a.o" (which needs the "a") is still caught.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = unicodedata.normalize("NFKD", text.casefold().translate(_CONFUSABLES))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    tokens: list[str] = []
    raw = [t for t in _SPLIT.split(text) if t]
    for index, token in enumerate(raw):
        age = _AGE.match(token)
        if age and (token != age.group(1) or (index + 1 < len(raw) and raw[index + 1] in _AGE_UNITS)):
            tokens.append(MINOR_TOKEN if int(age.group(1)) < 18 else token)
            continue
        if token.isdigit():
            tokens.append(token)
            continue
        folded = "".join(ch for ch in token.translate(_LEET) if ch.isalpha())
        if folded:
            tokens.append(folded)
    first = _merge_single_letters(tokens)
    second = _merge_single_letters(tokens, split_leading_word=True)
    return [first] if second == first else [first, second]


_ONE_LETTER_WORDS = frozenset("ai")


def _merge_single_letters(tokens: list[str], split_leading_word: bool = False) -> list[str]:
    """Joins spelled-out words: "c s a m" and "c.s.a.m" both become "csam".

    With `split_leading_word`, a run of four or more that starts with "a" or "i" keeps that
    letter as its own word ("a n u d e" -> "a", "nude").
    """
    merged: list[str] = []
    run: list[str] = []
    for token in tokens + [""]:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if split_leading_word and len(run) >= 4 and run[0] in _ONE_LETTER_WORDS:
            merged.append(run.pop(0))
        if len(run) >= 3:
            merged.append("".join(run))
        else:
            merged.extend(run)
        run = []
        if token:
            merged.append(token)
    return merged


# ---------------------------------------------------------------- term lists (normalised spelling)

ABSOLUTE_TERMS = [
    "csam", "cp porn", "child porn", "child pornography", "child sexual", "child nude", "child nudes", "kiddie porn",
    "kiddy porn", "kid porn", "preteen porn", "preteen nude", "preteen sex", "underage porn", "underage sex",
    "underage nude", "pedo porn", "pedophile porn", "jailbait", "lolicon", "shotacon", "toddlercon",
]
ABSOLUTE_PREFIXES = ["childporn", "kiddieporn", "kiddyporn", "underageporn", "pedoporn", "lolicon", "shotacon"]

MINOR_TERMS = [
    "child", "children", "kid", "kids", "kiddie", "kiddy", "minor", "minors", "underage", "under age", "preteen", "pre teen",
    "preteens", "tween", "tweens", "toddler", "toddlers", "infant", "infants", "schoolgirl", "schoolgirls", "schoolboy",
    "schoolboys", "teen", "teens", "teenage", "teenager", "teenagers", "young girl", "young boy", "little girl",
    "little boy", "loli", "shota", "middle schooler", "elementary schooler", MINOR_TOKEN,
]

# Sexual in every context this policy knows about: blocks anywhere in the prompt.
SEXUAL_TERMS = [
    # pornography and explicit media
    "porn", "porno", "pron", "nsfw", "hentai", "ecchi", "ahegao", "yiff", "futanari", "smut", "smutty", "lewd",
    "softcore", "onlyfans", "pornhub", "xvideos", "xhamster", "brazzers", "camgirl", "camgirls", "rule 34", "xxx",
    # sex and sexualisation
    "sex", "sexo", "sexe", "seggs", "segs", "secks", "sexuality", "sexually", "sexting",
    "lingerie", "fetish", "fetishes", "bondage", "bdsm", "kink", "milf", "slut", "sluts", "slutty", "whore",
    "nympho", "horny", "erotica", "raunchy", "seductive", "seductively", "sensual", "sensually", "sultry",
    "stripper", "strippers", "striptease", "strip tease", "strip club", "stripclub", "lap dance", "lapdance",
    # nudity
    "nude", "nudes", "nudity", "naked", "nakedness", "nackt", "desnuda", "desnudo", "topless", "unclothed",
    "in the nude", "in the buff", "birthday suit", "skinny dipping", "skinnydipping", "nip slip", "nipslip",
    "upskirt", "downblouse", "cameltoe", "camel toe", "wardrobe malfunction",
    # anatomy in a sexual register
    "breast", "breasts", "boobs", "boobies", "tits", "titties", "nipple", "nipples", "areola", "areolas", "areolae",
    "pussy", "penis", "penises", "vagina", "vaginas", "vulva", "clitoris", "testicles", "scrotum", "labia",
    "phallus", "pubic", "dildo", "dildos", "butt plug", "buttplug", "sex toy",
    # sexual acts
    "orgy", "orgies", "intercourse", "coitus", "blowjob", "handjob", "footjob", "rimjob", "cumshot", "bukkake",
    "gangbang", "incest", "bestiality", "rape", "raped", "raping", "grope", "groped", "groping", "sexy",
    "sexual", "sexualized", "sexualised",
]
SEXUAL_PREFIXES = [
    "porn", "erotic", "masturbat", "molest", "undress", "genital", "fornicat", "orgasm", "copulat",
    "nudis", "sexualiz", "sexualis", "zoophil", "necrophil", "disrob", "nudif", "deepnud",
]

# Only unacceptable next to a minor term: too common elsewhere to block outright.
MINOR_ONLY_TERMS = ["explicit", "provocative", "provocatively", "suggestive", "suggestively"]

CLOTHING_TERMS = ["clothes", "clothing", "clothed", "dressed", "underwear", "bra", "panties", "swimsuit", "bikini", "garments"]
# In a negative prompt these ask for nudity or uncensored output.
NEGATIVE_STEERING_TERMS = CLOTHING_TERMS + ["censored", "censorship", "censor", "sfw"]
REMOVAL_TERMS = ["remove", "removes", "removing", "removed", "strip", "strips", "stripping", "stripped",
                 "take off", "takes off", "taking off", "took off", "without", "pull off", "pulls off", "pulling off"]
REMOVAL_WINDOW = 3

DEEPFAKE_TERMS = ["deepfake", "deepfakes", "deep fake", "faceswap", "face swap", "celebrity", "celebrities", "real person",
                  "real people", "nudify", "nudifier", "undress app", "clothes remover"]
# "nudify"-style tools exist to undress real people: always a sexual deepfake.
DEEPFAKE_SEXUAL_PREFIXES = ["nudif", "deepnud"]

COOCCURRENCE_WINDOW = 12
MAX_JOIN = 4
CONTEXT_WINDOW = 2

# Joins never include these, so ordinary sentences don't spell out a term ("the pen is red").
_JOIN_STOPWORDS = frozenset(
    "a an the of and or for in on at to by as is it be do go he if me my no so up us we am are was were but not you "
    "all any can her his its our out one who how why yes pen hi oh ok".split()
)
# Skipped when counting the distance to an allow-context word.
_FILLERS = frozenset("a an the of and or for in on to with at its my your our their this that these those".split())
# Scanning for an allow-context word stops at these: the ambiguous word then describes a person.
PERSON_TERMS = frozenset(
    "woman women man men girl girls boy boys lady ladies guy guys person people model models body bodies wife husband "
    "girlfriend boyfriend actress actor she he her him herself himself lover lovers figure female male".split()
)


@dataclass(frozen=True)
class AllowContext:
    """Where an ambiguous term is ordinary: a context word within CONTEXT_WINDOW content words
    `before` or `after` it, or (for `after_of`) after it when "of" immediately follows the term."""

    before: frozenset[str] = frozenset()
    after: frozenset[str] = frozenset()
    after_of: frozenset[str] = frozenset()


def _stem(word: str) -> str:
    """Crude plural folding for context words only ("colours" -> "colour", "breasts" -> "breast")."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def _words(text: str) -> frozenset[str]:
    return frozenset(_stem(w) for w in text.split())


_ANIMALS = ("bird chick chicken hen fish animal species plant reptile turtle tortoise egg puppy kitten cat dog bee "
            "insect butterfly snake lizard parrot cattle calf foal horse fetus foetus baby lobster crab frog")
_BREAST = AllowContext(
    before=_words("chicken turkey duck goose quail pheasant poultry grilled roasted roast fried baked boneless skinless "
                  "smoked robin red grill sliced"),
    after=_words("cancer stroke pocket plate feeding milk pump exam screening awareness meat fillet bone"),
)
AMBIGUOUS: dict[str, AllowContext] = {
    "sex": AllowContext(
        before=_words("same opposite biological both either mixed fetal foetal"),
        after=_words("determination ratio chromosome difference identification linked reversal biological"),
        after_of=_words(_ANIMALS),
    ),
    "sexual": AllowContext(after=_words("orientation harassment health reproduction reproductive selection dimorphism dimorphic maturity")),
    "sexy": AllowContext(after=_words("car sportscar vehicle truck motorcycle motorbike yacht boat jet font typography logo")),
    "nude": AllowContext(
        after=_words("colour color palette tone shade lipstick lip lipgloss gloss makeup eyeshadow blush nail polish "
                     "manicure heel shoe pump sandal beige pink fabric paint wall foundation hue"),
    ),
    "naked": AllowContext(after=_words("eye flame truth ambition bulb lightbulb mole molerat wire blade cake tree branch")),
    "breast": _BREAST,
    "breasts": _BREAST,
    "nipple": AllowContext(before=_words("bottle grease feeding")),
    "nipples": AllowContext(before=_words("bottle grease feeding")),
    "tits": AllowContext(
        before=_words("blue great coal marsh willow crested penduline bearded tailed"),
        after=_words("bird nest feeder flock chick egg"),
    ),
    "pussy": AllowContext(after=_words("cat willow")),
    "horny": AllowContext(after=_words("toad lizard beetle coral")),
    "topless": AllowContext(after=_words("bus car convertible jeep tour roadster boat")),
    "kink": AllowContext(after=_words("hose cable wire pipe chain")),
    "stripper": AllowContext(before=_words("wire paint wallpaper cable")),
    "strippers": AllowContext(before=_words("wire paint wallpaper cable")),
    "sultry": AllowContext(after=_words("weather day night summer afternoon evening heat air jazz")),
    "seductive": AllowContext(after=_words("perfume fragrance chocolate dessert scent cologne")),
    "sensual": AllowContext(after=_words("perfume fragrance chocolate dessert scent cologne")),
    "intercourse": AllowContext(before=_words("social")),
    "rape": AllowContext(after=_words("seed field oil")),
    # Roman numerals: "Super Bowl XXX", "Chapter XXX".
    "xxx": AllowContext(before=_words("bowl chapter volume part page act olympiad")),
}


# ---------------------------------------------------------------- matching


@dataclass(frozen=True)
class Hit:
    term: str
    start: int
    end: int  # exclusive


class _TermSet:
    def __init__(self, terms: Iterable[str], prefixes: Sequence[str] = ()):
        self.exact: dict[str, str] = {}
        self.collapsed: dict[str, str] = {}
        self.phrases: dict[tuple[str, ...], str] = {}
        for term in terms:
            canonical = term.replace(" ", "")
            self.exact[canonical] = term
            squeezed = _collapse(canonical)
            if len(squeezed) >= 3:
                self.collapsed.setdefault(squeezed, term)
            if " " in term:
                self.phrases[tuple(term.split())] = term
        self.prefixes = tuple(prefixes)
        self.collapsed_prefixes = tuple(_collapse(p) for p in prefixes)

    def lookup(self, candidate: str) -> str | None:
        if candidate in self.exact:
            return self.exact[candidate]
        if candidate.startswith(self.prefixes):
            return candidate
        squeezed = _collapse(candidate)
        if squeezed != candidate:  # only text with repeated letters is squeezed
            if squeezed in self.collapsed:
                return self.collapsed[squeezed]
            if squeezed.startswith(self.collapsed_prefixes):
                return candidate
        return None

    def hits(self, tokens: Sequence[str]) -> list[Hit]:
        """One hit per start position: a whole-token phrase, or a join of up to MAX_JOIN tokens."""
        found: list[Hit] = []
        for start in range(len(tokens)):
            hit = None
            for length in range(min(MAX_JOIN, len(tokens) - start), 1, -1):
                term = self.phrases.get(tuple(tokens[start : start + length]))
                if term:
                    hit = Hit(term, start, start + length)
                    break
            if hit is None:
                candidate = ""
                for end in range(start, min(start + MAX_JOIN, len(tokens))):
                    if end > start and (tokens[end] in _JOIN_STOPWORDS or tokens[start] in _JOIN_STOPWORDS):
                        break
                    candidate += tokens[end]
                    term = self.lookup(candidate)
                    if term:
                        hit = Hit(term, start, end + 1)
                        break
            if hit is not None:
                found.append(hit)
        return found


_ABSOLUTE = _TermSet(ABSOLUTE_TERMS, ABSOLUTE_PREFIXES)
_MINOR = _TermSet(MINOR_TERMS)
_SEXUAL = _TermSet(SEXUAL_TERMS, SEXUAL_PREFIXES)
_MINOR_ONLY = _TermSet(MINOR_ONLY_TERMS)
_CLOTHING = _TermSet(CLOTHING_TERMS)
_NEGATIVE_STEERING = _TermSet(NEGATIVE_STEERING_TERMS)
_REMOVAL = _TermSet(REMOVAL_TERMS)
_DEEPFAKE = _TermSet(DEEPFAKE_TERMS, DEEPFAKE_SEXUAL_PREFIXES)


def _allowed(hit: Hit, tokens: Sequence[str], blocked_positions: set[int]) -> bool:
    """True when an ambiguous term sits in one of its allow-contexts."""
    context = AMBIGUOUS.get(hit.term)
    if context is None:
        return False

    def scan(positions: Iterable[int], words: frozenset[str]) -> bool:
        seen = 0
        for pos in positions:
            token = tokens[pos]
            if token in PERSON_TERMS or token == MINOR_TOKEN or pos in blocked_positions:
                return False
            if token in _FILLERS:
                continue
            if _stem(token) in words:
                return True
            seen += 1
            if seen >= CONTEXT_WINDOW:
                return False
        return False

    if context.after and scan(range(hit.end, len(tokens)), context.after):
        return True
    if context.before and scan(range(hit.start - 1, -1, -1), context.before):
        return True
    if context.after_of and hit.end < len(tokens) and tokens[hit.end] == "of":
        return scan(range(hit.end + 1, len(tokens)), context.after_of)
    return False


def _sexual_hits(tokens: Sequence[str]) -> list[Hit]:
    raw = _SEXUAL.hits(tokens)
    positions = {p for hit in raw for p in range(hit.start, hit.end)}
    return [hit for hit in raw if not _allowed(hit, tokens, positions - set(range(hit.start, hit.end)))]


def _removal_of_clothing(tokens: Sequence[str]) -> list[Hit]:
    """ "take off her clothes", "without clothing": a removal verb shortly before a clothing word."""
    clothing = _CLOTHING.hits(tokens)
    return [
        removal for removal in _REMOVAL.hits(tokens)
        if any(0 <= c.start - removal.end <= REMOVAL_WINDOW for c in clothing)
    ]


def _near(a: Sequence[Hit], b: Sequence[Hit], window: int = COOCCURRENCE_WINDOW) -> bool:
    return any(abs(x.start - y.start) <= window for x in a for y in b)


def check_prompt(prompt: str, negative_prompt: str | None = None) -> None:
    """Raises ContentPolicyViolation when the prompt breaks the content policy; returns None otherwise."""
    prompts = _token_variants(prompt or "")
    negatives = _token_variants(negative_prompt or "")
    # The gravest category wins when readings disagree.
    found = [_category(tokens, negative) for tokens in prompts for negative in negatives]
    for category in CATEGORIES:
        if category in found:
            raise ContentPolicyViolation(category)


def _category(tokens: Sequence[str], negative: Sequence[str]) -> str | None:
    try:
        _check_tokens(tokens, negative)
    except ContentPolicyViolation as exc:
        return exc.category
    return None


def _check_tokens(tokens: Sequence[str], negative: Sequence[str]) -> None:
    if _ABSOLUTE.hits(tokens) or _ABSOLUTE.hits(negative):
        raise ContentPolicyViolation("sexual_minors")
    minors = _MINOR.hits(tokens)
    sexual = _sexual_hits(tokens) + _removal_of_clothing(tokens)
    if sexual:
        if _near(minors, sexual):
            raise ContentPolicyViolation("sexual_minors")
        if _DEEPFAKE.hits(tokens):
            raise ContentPolicyViolation("sexual_deepfake")
        raise ContentPolicyViolation("sexual")
    if minors and _near(minors, _MINOR_ONLY.hits(tokens)):
        raise ContentPolicyViolation("sexual_minors")
    # A negative prompt steers away from what it lists: "clothing" there pushes toward nudity.
    if _NEGATIVE_STEERING.hits(negative):
        raise ContentPolicyViolation("sexual_minors" if minors else "sexual")


def mentions_minor(prompt: str) -> bool:
    """True when the prompt (not the negative prompt, which lists what to avoid) names a minor."""
    return any(_MINOR.hits(tokens) for tokens in _token_variants(prompt or ""))
