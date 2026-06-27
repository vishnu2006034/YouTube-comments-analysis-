import json
import logging
import os
import re
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("yt-analyzer-local")

def sanitize_for_json(text: str) -> str:
    """Remove or escape characters that often break LLM JSON generation."""
    if not text:
        return ""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Replace literal newlines with space to keep the prompt clean
    text = text.replace("\n", " ").replace("\r", " ")
    # Strip extreme whitespace
    text = " ".join(text.split())
    return text

@dataclass
class IntervalAnalysis:
    interval: str
    sentiment_analysis: str
    recommendation: str
    sentiment_breakdown: Dict[str, float]
    top_topics: List[Dict[str, str]]
    viewer_reactions: List[str]
    pain_points: List[str]
    engagement_drivers: List[str]
    content_opportunities: List[str]
    creator_recommendation: str


_TIMESTAMP_PATTERNS = [
    # 1:23, 12:44, 01:02, optionally with AM/PM suffix
    r"(?P<m>\d{1,2})\s*:\s*(?P<s>\d{2})\s*(?P<ampm>[aApP][mM])?",
]


def _to_seconds(mins: int, secs: int) -> int:
    return mins * 60 + secs


def extract_timestamps_seconds(text: str) -> List[int]:
    if not text:
        return []

    # Normalize: many comments have newlines/extra symbols
    t = str(text)

    found: List[int] = []
    for pat in _TIMESTAMP_PATTERNS:
        for m in re.finditer(pat, t):
            try:
                mins = int(m.group("m"))
                secs = int(m.group("s"))
                if 0 <= secs <= 59:
                    found.append(_to_seconds(mins, secs))
            except Exception:
                continue

    # Deduplicate + sort
    return sorted(set(found))


def build_intervals(
    timestamps_seconds: List[int],
    interval_seconds: int = 60,
    max_intervals: int = 20,
) -> List[Tuple[int, int]]:
    """Create [start, end) intervals from earliest timestamp."""
    if not timestamps_seconds:
        return [(0, interval_seconds)]

    start = min(timestamps_seconds)
    # align down to interval boundary
    start = (start // interval_seconds) * interval_seconds

    # end bound heuristic
    end_ts = max(timestamps_seconds)
    num = max(1, int((end_ts - start) / interval_seconds) + 1)
    num = min(num, max_intervals)

    intervals = []
    for i in range(num):
        s = start + i * interval_seconds
        e = s + interval_seconds
        intervals.append((s, e))
    return intervals


def format_interval(start_s: int, end_s: int) -> str:
    # display as 0-15s, 15-30s ...
    return f"{start_s}-{end_s}s"


def assign_comments_to_intervals(
    comments: List[str],
    interval_seconds: int = 60,
) -> Dict[str, List[str]]:
    """Assign each comment to an interval.

    If a comment contains multiple timestamps, use the first one.

    If a comment has no timestamp, assign it to the nearest interval boundary by
    distributing those comments across the timeline (instead of ignoring them).
    """

    all_ts: List[int] = []
    comment_ts: List[Optional[int]] = []

    for c in comments:
        ts = extract_timestamps_seconds(c)
        if ts:
            all_ts.extend(ts)
            comment_ts.append(ts[0])
        else:
            comment_ts.append(None)

    intervals = build_intervals(all_ts, interval_seconds=interval_seconds)

    interval_map: Dict[str, List[str]] = {}
    for s, e in intervals:
        interval_map[format_interval(s, e)] = []

    # If no timestamps in any comment, everything goes into first interval
    if not all_ts:
        first_key = next(iter(interval_map.keys()))
        interval_map[first_key] = [c for c in comments if c]
        return interval_map

    # Helper to find containing interval; if outside, clamp to last.
    def _place_for_ts(ts: int) -> str:
        for s, e in intervals:
            if s <= ts < e:
                return format_interval(s, e)
        return format_interval(intervals[-1][0], intervals[-1][1])

    no_ts_comments: List[str] = []

    for c, ts in zip(comments, comment_ts):
        if not c:
            continue
        if ts is None:
            no_ts_comments.append(c)
            continue
        key = _place_for_ts(ts)
        interval_map[key].append(c)

    # Distribute timestamp-less comments across intervals (round-robin)
    if no_ts_comments:
        keys = [format_interval(s, e) for s, e in intervals]
        for idx, c in enumerate(no_ts_comments):
            interval_map[keys[idx % len(keys)]].append(c)

    # Remove empty intervals for compactness
    interval_map = {k: v for k, v in interval_map.items() if v}
    return interval_map


def _extract_top_themes(texts: List[str], limit: int = 3) -> Tuple[List[str], List[str]]:
    """Return (positive_themes, negative_themes) derived from keyword groups."""

    joined = "\n".join(texts).lower()

    # Praise themes
    positive_groups: Dict[str, List[str]] = {
        "warmth / kindness": [
            "wholesome",
            "love",
            "blessed",
            "happy",
            "feel good",
            "felt good",
            "stay happy",
        ],
        "clear storytelling": [
            "clear",
            "story",
            "explained",
            "explaining",
            "good explanation",
            "makes sense",
        ],
        "consistency / updates": [
            "update",
            "updates",
            "consistent",
            "upload",
            "uploads",
            "keep it up",
        ],
        "supportive community": [
            "support",
            "congrats",
            "congratulations",
            "wishing",
            "best",
            "amazing",
            "great",
        ],
        "quality / entertainment": ["funny", "entertaining", "awesome", "cute"],
    }

    # Concern themes
    negative_groups: Dict[str, List[str]] = {
        "misleading / fake claims": ["fraud", "fake", "scam", "mislead", "manipulat"],
        "disrespect / cringe": ["cringe", "hate", "dislike", "angry"],
        "disappointment / regret": ["regret", "disappointed", "worst"],
        "problem / issues": ["problem", "issue", "broken", "doesn't work", "doesnt work"],
        "diversion / shady motives": ["divert", "agenda", "suspicious"],
    }

    pos_hits: List[Tuple[str, int]] = []
    for theme, keys in positive_groups.items():
        score = sum(1 for k in keys if k in joined)
        if score > 0:
            pos_hits.append((theme, score))

    neg_hits: List[Tuple[str, int]] = []
    for theme, keys in negative_groups.items():
        score = sum(1 for k in keys if k in joined)
        if score > 0:
            neg_hits.append((theme, score))

    pos_hits.sort(key=lambda x: x[1], reverse=True)
    neg_hits.sort(key=lambda x: x[1], reverse=True)

    pos_themes = [t for t, _ in pos_hits[:limit]]
    neg_themes = [t for t, _ in neg_hits[:limit]]

    return pos_themes, neg_themes


# ────────────────────────────────────────────────────────────────────
# Stopwords for keyword extraction
# ────────────────────────────────────────────────────────────────────
_STOP_WORDS = frozenset(
    "i me my myself we our ours ourselves you your yours yourself yourselves "
    "he him his himself she her hers herself it its itself they them their "
    "theirs themselves what which who whom this that these those am is are was "
    "were be been being have has had having do does did doing a an the and but "
    "if or because as until while of at by for with about against between "
    "through during before after above below to from up down in out on off "
    "over under again further then once here there when where why how all any "
    "both each few more most other some such no nor not only own same so than "
    "too very s t can will just don should now d ll m o re ve y ain aren "
    "couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn "
    "wasn weren won wouldn also would could one like got get go really even "
    "still much know thing think gonna much lol lmao yeah yes no oh wow ok "
    "okay omg gonna gotta im dont youre its thats hes shes "
    "video comment like channel watch watching watched".split()
)


def _extract_keywords(texts: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    """Extract the most frequent meaningful words from comment texts.

    Returns: list of (word, count) tuples sorted by frequency descending.
    """
    word_counts: Dict[str, int] = {}
    for text in texts:
        words = re.findall(r"[a-z']+", text.lower())
        for w in words:
            if len(w) < 3 or w in _STOP_WORDS:
                continue
            word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]


def _detect_questions(texts: List[str]) -> List[str]:
    """Extract comments that look like viewer questions."""
    questions: List[str] = []
    question_starters = re.compile(
        r"^(who|what|when|where|why|how|can|could|would|should|is|are|do|does|did|will|any)\b",
        re.IGNORECASE,
    )
    for t in texts:
        stripped = t.strip()
        if "?" in stripped or question_starters.match(stripped):
            q = stripped[:120] + ("…" if len(stripped) > 120 else "")
            questions.append(q)
    return questions[:5]


def _count_sentiment_occurrences(texts: List[str]) -> Dict[str, int]:
    """Count actual occurrences of sentiment-bearing words across all comments."""
    positive_words = [
        "love", "loved", "amazing", "awesome", "great", "beautiful", "perfect",
        "fantastic", "wonderful", "excellent", "brilliant", "incredible", "best",
        "happy", "blessed", "wholesome", "inspiring", "heartwarming",
        "helpful", "thanks", "thank", "good", "nice", "cool", "sweet", "proud",
        "congrats", "congratulations", "superb", "outstanding", "respect",
        "talented", "genius", "masterpiece", "legendary", "fire", "goat",
        "underrated", "gem", "solid", "satisfying", "touching", "emotional",
    ]
    negative_words = [
        "hate", "hated", "bad", "terrible", "awful", "worst", "boring",
        "cringe", "disappointing", "disappointed", "annoying", "overrated",
        "fake", "fraud", "scam", "trash", "garbage", "waste", "stupid",
        "dumb", "ugly", "horrible", "pathetic", "useless",
        "clickbait", "misleading", "manipulative", "unsubscribe", "unsub",
        "dislike", "angry", "frustrated", "confused", "regret", "problematic",
        "broken", "skip", "skipped", "unwatchable", "poorly", "mediocre",
    ]

    pos_count = 0
    neg_count = 0
    for text in texts:
        lower = text.lower()
        for w in positive_words:
            pos_count += lower.count(w)
        for w in negative_words:
            neg_count += lower.count(w)

    return {"positive": pos_count, "negative": neg_count}


def _compute_sentiment_breakdown(texts: List[str]) -> Dict[str, float]:
    """Return percentage breakdown of positive / neutral / negative sentiment."""
    counts = _count_sentiment_occurrences(texts)
    total_signals = counts["positive"] + counts["negative"]
    total_comments = len(texts)

    if total_signals == 0:
        return {"positive_percent": 0.0, "neutral_percent": 100.0, "negative_percent": 0.0}

    # Estimate neutral ratio from comments with no strong signals
    signal_words = ["love", "great", "amazing", "hate", "bad", "worst", "boring", "awesome", "best", "terrible"]
    comments_with_signal = sum(
        1 for t in texts
        if any(w in t.lower() for w in signal_words)
    )
    neutral_ratio = max(0, (total_comments - comments_with_signal)) / max(total_comments, 1)
    neutral_pct = round(neutral_ratio * 100, 1)
    remaining = 100.0 - neutral_pct
    if remaining > 0 and total_signals > 0:
        pos_pct = round((counts["positive"] / total_signals) * remaining, 1)
        neg_pct = round(remaining - pos_pct, 1)
    else:
        pos_pct = 0.0
        neg_pct = 0.0

    return {"positive_percent": pos_pct, "neutral_percent": neutral_pct, "negative_percent": neg_pct}


def _detect_pain_points(texts: List[str]) -> List[str]:
    """Detect specific issues or complaints viewers raised."""
    pain_indicators = re.compile(
        r"(doesn.t work|not working|can.t find|where is|missing|wrong|error|"
        r"please fix|too long|too short|too fast|too slow|confusing|unclear|"
        r"no subtitles|bad audio|bad quality|low quality|clickbait|misleading|"
        r"waste of time|not helpful|doesn.t make sense|hard to follow|"
        r"should have|could have|needs more|lacks|repetitive|dragged|filler)",
        re.IGNORECASE,
    )
    points: List[str] = []
    seen: set = set()
    for t in texts:
        matches = pain_indicators.findall(t)
        for m in matches:
            norm = m.lower().strip()
            if norm not in seen:
                seen.add(norm)
                snippet = t[:150] + ("…" if len(t) > 150 else "")
                points.append(f'Viewers mention "{norm}" — e.g. "{snippet}"')
    return points[:5]


def _detect_engagement_drivers(texts: List[str]) -> List[str]:
    """Detect what's driving viewer engagement in these comments."""
    drivers: List[str] = []
    total = len(texts)
    if total == 0:
        return drivers

    # Direct creator mentions
    creator_mentions = sum(1 for t in texts if any(w in t.lower() for w in ["bro", "sir", "dude", "brother", "sister"]) or t.strip().startswith("@"))
    if creator_mentions > total * 0.1:
        drivers.append(f"Direct creator engagement: {round(creator_mentions/total*100)}% of comments address the creator personally")

    # Emoji-heavy comments
    emoji_pattern = re.compile(r"[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\u2600-\u27bf]")
    emoji_comments = sum(1 for t in texts if emoji_pattern.search(t))
    if emoji_comments > total * 0.15:
        drivers.append(f"High emotional reactions: {round(emoji_comments/total*100)}% of comments contain emojis/reactions")

    # Timestamp references
    ts_comments = sum(1 for t in texts if re.search(r"\d{1,2}:\d{2}", t))
    if ts_comments > total * 0.05:
        drivers.append(f"Moment-specific reactions: {round(ts_comments/total*100)}% reference exact timestamps")

    # Long comments (high investment)
    long_comments = sum(1 for t in texts if len(t) > 200)
    if long_comments > total * 0.1:
        drivers.append(f"Deep engagement: {round(long_comments/total*100)}% of comments are 200+ chars (high viewer investment)")

    return drivers[:4]


def heuristic_sentiment_and_recommendation(
    texts: List[str],
    interval_label: str = "",
    all_interval_data: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, str, Dict[str, float], List[Dict[str, str]], List[str], List[str], List[str], List[str], str]:
    """Content-aware heuristic analysis producing unique, interval-specific insights.

    Returns a 9-tuple:
        (sentiment_analysis, recommendation, sentiment_breakdown,
         top_topics, viewer_reactions, pain_points,
         engagement_drivers, content_opportunities, creator_recommendation)
    """
    if not texts:
        return (
            "No comments in this interval",
            "No data available for analysis.",
            {"positive_percent": 0, "neutral_percent": 100, "negative_percent": 0},
            [], [], [], [], [],
            "Consider adding a hook or CTA at this timestamp to generate discussion.",
        )

    total = len(texts)

    # ── 1. Sentiment breakdown ──────────────────────────────────────
    breakdown = _compute_sentiment_breakdown(texts)
    pos_pct = breakdown["positive_percent"]
    neg_pct = breakdown["negative_percent"]
    neu_pct = breakdown["neutral_percent"]
    sentiment_str = f"Pos: {pos_pct}% | Neutral: {neu_pct}% | Neg: {neg_pct}%"

    # ── 2. Top keywords / topics ────────────────────────────────────
    keywords = _extract_keywords(texts, top_n=8)
    top_topics: List[Dict[str, str]] = []
    for word, count in keywords[:5]:
        freq_pct = round(count / total * 100)
        pos_ctx = sum(1 for t in texts if word in t.lower() and any(p in t.lower() for p in ["love", "great", "amazing", "good", "best", "awesome"]))
        neg_ctx = sum(1 for t in texts if word in t.lower() and any(n in t.lower() for n in ["hate", "bad", "worst", "boring", "terrible"]))
        if pos_ctx > neg_ctx:
            opinion = "Mostly praised"
        elif neg_ctx > pos_ctx:
            opinion = "Mostly criticized"
        else:
            opinion = "Neutral/mixed"
        top_topics.append({
            "topic": word,
            "mention_frequency": f"{freq_pct}% of comments ({count}/{total})",
            "viewer_opinion": opinion,
        })

    # ── 3. Viewer reactions (sample actual comments) ────────────────
    viewer_reactions: List[str] = []
    positive_samples = [t for t in texts if any(w in t.lower() for w in ["love", "amazing", "great", "best", "awesome"])]
    negative_samples = [t for t in texts if any(w in t.lower() for w in ["hate", "bad", "worst", "boring", "disappointed"])]
    question_samples = _detect_questions(texts)

    if positive_samples:
        viewer_reactions.append(f'Positive: "{positive_samples[0][:120]}"')
    if negative_samples:
        viewer_reactions.append(f'Concern: "{negative_samples[0][:120]}"')
    if question_samples:
        viewer_reactions.append(f'Question: "{question_samples[0]}"')
    if not viewer_reactions:
        viewer_reactions.append(f"{total} comments analyzed, mostly neutral/conversational")

    # ── 4. Pain points ──────────────────────────────────────────────
    pain_points = _detect_pain_points(texts)

    # ── 5. Engagement drivers ───────────────────────────────────────
    engagement_drivers = _detect_engagement_drivers(texts)

    # ── 6. Content opportunities ────────────────────────────────────
    content_opportunities: List[str] = []
    questions = _detect_questions(texts)
    if questions:
        content_opportunities.append(
            f"{len(questions)} viewer question(s) detected — consider a follow-up or pinned reply: \"{questions[0]}\""
        )
    non_obvious_kws = [w for w, c in keywords if c >= 3 and w not in ("video", "comment", "like", "channel")]
    if non_obvious_kws:
        content_opportunities.append(
            f"Recurring topics ({', '.join(non_obvious_kws[:3])}) suggest viewer interest — potential standalone video ideas"
        )

    # ── 7. Generate unique, data-driven recommendation ──────────────
    rec_parts: List[str] = []

    # Lead with the dominant signal + actual data
    if neg_pct > 30:
        neg_kws = [w for w, _ in keywords if any(n in w for n in ["bad", "hate", "boring", "fake", "scam", "worst", "confus"])]
        if neg_kws:
            rec_parts.append(
                f"⚠️ {neg_pct}% negative sentiment. Top concern keyword: \"{neg_kws[0]}\". "
                f"Address this directly in a pinned comment or annotation at {interval_label}."
            )
        else:
            rec_parts.append(
                f"⚠️ {neg_pct}% negative sentiment across {total} comments. Review for specific complaints."
            )
    elif pos_pct > 70 and keywords:
        rec_parts.append(
            f"✅ Strong approval ({pos_pct}% positive). \"{keywords[0][0]}\" appears {keywords[0][1]}× — "
            f"this resonates. Feature more of this in future videos."
        )
    elif keywords:
        rec_parts.append(
            f"📊 Mixed reactions ({pos_pct}% pos / {neg_pct}% neg / {neu_pct}% neutral) "
            f"across {total} comments. Top keyword: \"{keywords[0][0]}\" ({keywords[0][1]} mentions)."
        )
    else:
        rec_parts.append(f"📊 {total} comments with {pos_pct}% positive / {neg_pct}% negative sentiment.")

    # Add actionable specifics
    if questions:
        rec_parts.append(
            f"📌 Viewers asking: \"{questions[0][:80]}\" — answering boosts engagement."
        )
    if pain_points:
        rec_parts.append(f"🔧 Issue: {pain_points[0][:100]}")
    if engagement_drivers:
        rec_parts.append(f"💡 {engagement_drivers[0]}")

    # Comparative note across intervals
    if all_interval_data and len(all_interval_data) > 1:
        avg_count = sum(len(v) for v in all_interval_data.values()) / len(all_interval_data)
        if total > avg_count * 1.3:
            rec_parts.append(
                f"📈 This interval has {round((total/avg_count - 1)*100)}% more comments than average — a hot spot."
            )
        elif total < avg_count * 0.5:
            rec_parts.append(
                f"📉 Low comment volume ({total} vs avg {round(avg_count)}). Consider tighter editing here."
            )

    creator_recommendation = " ".join(rec_parts)

    return (
        sentiment_str,
        creator_recommendation,
        breakdown,
        top_topics,
        viewer_reactions,
        pain_points,
        engagement_drivers,
        content_opportunities,
        creator_recommendation,
    )



def validate_gemini_response(text: Optional[str]) -> None:
    """Basic sanity checks before attempting JSON parsing."""
    if text is None:
        raise ValueError("Gemini response had no `.text` field (None).")
    if not isinstance(text, str):
        raise ValueError(
            f"Gemini response `.text` was not a string (type={type(text)})."
        )
    if not text.strip():
        raise ValueError("Gemini response `.text` is empty.")





def clean_markdown_json(raw: str) -> str:
    """Extract JSON from markdown/code fences and trim leading junk."""
    s = raw.strip()

    # Common case: ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        # remove first fence line
        parts = s.split("```", 2)
        if len(parts) >= 2:
            s = parts[1].strip()
        # if it still has trailing fence, remove it
        s = s.replace("```", "").strip()

    # If model added leading text, try to locate first '[' or '{'
    first_bracket = min([i for i in (s.find('['), s.find('{')) if i != -1], default=-1)
    if first_bracket > 0:
        s = s[first_bracket:].strip()

    return s


def safely_extract_json(raw_text: str) -> object:
    """Best-effort JSON extraction. Never throws JSON parsing without context."""
    cleaned = clean_markdown_json(raw_text)

    # Prefer direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try a more permissive heuristic: capture from first '[' to last ']' if it looks like an array
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            return json.loads(candidate)

        # Try object capture
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            return json.loads(candidate)

        # Re-raise with the cleaned text for debugging
        raise


def analyze_intervals(
    comments: List[str],
    interval_seconds: int = 60,
    max_llm_intervals: int = 10,
) -> List[IntervalAnalysis]:
    """Main entrypoint.
    Uses Gemini 1.5 Flash for interval sentiment & recommendations.
    If parsing fails or Gemini errors, we fall back to heuristics.

    To enable Gemini:
      - install `google-generativeai`
      - set env var `GEMINI_API_KEY`
    """
    interval_map = assign_comments_to_intervals(comments, interval_seconds=interval_seconds)
    interval_items = [(k, v) for k, v in interval_map.items() if v]
    interval_items = interval_items[:max_llm_intervals]

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.warning("GEMINI_API_KEY env var not set. Falling back to heuristic analyzer.")
    else:
        raw_text = ""
        try:
            # pyrefly: ignore [missing-import]
            import google.generativeai as genai
            from google.generativeai.types import RequestOptions

            genai.configure(api_key=gemini_key)

            # 1. Define strict schema to guarantee JSON structure
            # Using Gemini's controlled output to match your requested structure
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "interval": {"type": "string"},
                        "sentiment_breakdown": {
                            "type": "object",
                            "properties": {
                                "positive_percent": {"type": "number"},
                                "neutral_percent": {"type": "number"},
                                "negative_percent": {"type": "number"},
                            },
                            "required": ["positive_percent", "neutral_percent", "negative_percent"],
                        },
                        "top_topics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "topic": {"type": "string"},
                                    "mention_frequency": {"type": "string"},
                                    "viewer_opinion": {"type": "string"},
                                },
                                "required": ["topic", "mention_frequency", "viewer_opinion"],
                            },
                        },
                        "viewer_reactions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "pain_points": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "engagement_drivers": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "content_opportunities": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "creator_recommendation": {"type": "string"},
                    },
                    "required": [
                        "interval", "sentiment_breakdown", "top_topics", "viewer_reactions",
                        "pain_points", "engagement_drivers", "content_opportunities", "creator_recommendation"
                    ],
                },
            }

            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
            )

            interval_to_lines: Dict[str, List[str]] = {}
            for interval, texts in interval_items:
                interval_to_lines[interval] = [sanitize_for_json(t) for t in texts[:50]]

            lines: List[str] = []
            for interval, texts in interval_to_lines.items():
                for t in texts:
                    lines.append(f"{interval} | {t}")

            csv_text = "\n".join(lines)
            if not csv_text:
                raise ValueError("No input data to analyze.")

            # 2. Refined Prompt
            prompt = (
                "You are a senior YouTube Growth Strategist, Audience Researcher, and Content Analyst. "
                "Analyze the comments associated with these video intervals.\n\n"
                "Do NOT provide generic statements such as 'Mostly positive', 'Lean into what worked', or 'Supportive community'. "
                "Instead provide actionable insights.\n\n"
                "Requirements:\n"
                "1. Identify recurring themes.\n"
                "2. Be extremely specific. Do not use generic words like 'good' or 'bad'.\n"
                "3. Identify what specifically viewers liked/disliked.\n"
                "4. Detect questions or confusion points.\n"
                "5. Detect retention risks.\n"
                "6. Compare this interval to the context of the rest of the comments if possible.\n"
                "7. Detect content ideas for future videos.\n"
                "8. If the comments for an interval are mostly spam or low-effort, state that clearly rather than inventing sentiment.\n"
                "9. Quantify observations whenever possible (e.g., '10% of commenters asked about...').\n"
                "10. Recommendations must be unique for each interval and highly specific.\n\n"
                f"Comments:\n{csv_text}"
            )

            # 3. Use response_mime_type and response_schema in the call
            # safety_settings are set to BLOCK_NONE to ensure the JSON structure isn't broken by a partial block.
            resp = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                    "temperature": 0.3,  # Increased slightly to allow for more diverse vocabulary/analysis
                    "max_output_tokens": 8192, # Increased for detailed analysis
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )

            if not resp or not resp.text:
                raise ValueError(f"Empty response from Gemini. Finish reason: {resp.candidates[0].finish_reason}")

            raw_text = resp.text
            logger.info("Raw Gemini Response: %s", raw_text)

            # 4. Robust Extraction
            data = safely_extract_json(raw_text)
            logger.info("Parsed JSON: %s", json.dumps(data))

            if not isinstance(data, list):
                raise ValueError("Root of JSON response must be an array.")

            results = [
                IntervalAnalysis(
                    interval=str(item.get("interval", "N/A")),
                    # Populate legacy fields from new structured data
                    sentiment_analysis=f"Pos: {item.get('sentiment_breakdown',{}).get('positive_percent',0)}% | Neu: {item.get('sentiment_breakdown',{}).get('neutral_percent',0)}% | Neg: {item.get('sentiment_breakdown',{}).get('negative_percent',0)}%",
                    recommendation=str(item.get("creator_recommendation", "No recommendation")),
                    # New fields
                    sentiment_breakdown=item.get("sentiment_breakdown", {"positive_percent": 0, "neutral_percent": 0, "negative_percent": 0}),
                    top_topics=item.get("top_topics", []),
                    viewer_reactions=item.get("viewer_reactions", []),
                    pain_points=item.get("pain_points", []),
                    engagement_drivers=item.get("engagement_drivers", []),
                    content_opportunities=item.get("content_opportunities", []),
                    creator_recommendation=str(item.get("creator_recommendation", "No specific recommendation generated."))
                ) for item in data if isinstance(item, dict)
            ]

            if results:
                return results

        except Exception as e:
            logger.error("Gemini Analysis Failed: %s", str(e))
            logger.error(traceback.format_exc())
            # Detailed logging of the failure for debugging
            if raw_text:
                logger.error("Faulty Raw Text: %s", raw_text)

    # 5. Robust Fallback to Heuristics
    logger.warning("Falling back to heuristic analyzer.")
    # Build a dict for cross-interval comparison
    all_interval_map = {k: v for k, v in interval_items}
    fallback_results: List[IntervalAnalysis] = []
    for interval, texts in interval_items:
        (
            sent, rec, s_breakdown, s_topics, s_reactions,
            s_pain, s_drivers, s_opps, s_creator_rec
        ) = heuristic_sentiment_and_recommendation(
            texts,
            interval_label=interval,
            all_interval_data=all_interval_map,
        )
        fallback_results.append(
            IntervalAnalysis(
                interval=interval,
                sentiment_analysis=sent,
                recommendation=rec,
                sentiment_breakdown=s_breakdown,
                top_topics=s_topics,
                viewer_reactions=s_reactions,
                pain_points=s_pain,
                engagement_drivers=s_drivers,
                content_opportunities=s_opps,
                creator_recommendation=s_creator_rec,
            )
        )
    
    if not fallback_results:
        return [IntervalAnalysis(
            interval="0-60s",
            sentiment_analysis="No data",
            recommendation="No comments available for analysis.",
            sentiment_breakdown={"positive_percent": 0, "neutral_percent": 0, "negative_percent": 0},
            top_topics=[],
            viewer_reactions=["No data"],
            pain_points=[],
            engagement_drivers=[],
            content_opportunities=[],
            creator_recommendation="Ensure the video has public comments."
        )]
        
    return fallback_results
