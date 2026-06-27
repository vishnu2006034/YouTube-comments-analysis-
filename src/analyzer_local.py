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


def heuristic_sentiment_and_recommendation(texts: List[str]) -> Tuple[str, str]:
    """Heuristic sentiment + interval-specific recommendation (non-repetitive).

    Output format is kept compatible with IntervalAnalysis.
    """

    joined = "\n".join(texts).lower()

    # crude overall signals (kept for sentiment class)
    negative_markers = [
        "cringe",
        "hate",
        "unsub",
        "unsubscribe",
        "fraud",
        "manipulat",
        "fake",
        "worst",
        "problem",
        "divert",
        "mislead",
        "dislike",
        "angry",
        "regret",
        "disappointed",
        "scam",
    ]
    positive_markers = [
        "congrats",
        "congratulations",
        "happy",
        "wholesome",
        "great",
        "love",
        "blessed",
        "wishing",
        "best",
        "stay happy",
        "felt good",
        "amazing",
        "cute",
        "feel good",
        "satisfied",
        "support",
    ]

    neg = sum(1 for m in negative_markers if m in joined)
    pos = sum(1 for m in positive_markers if m in joined)

    pos_themes, neg_themes = _extract_top_themes(texts, limit=3)

    # Sentiment class
    theme_str = ""
    if neg == 0 and pos == 0:
        sentiment = "Neutral / mixed reactions"
        theme_str = "neutral observations"
    elif pos >= neg:
        sentiment = "Mostly positive / supportive"
        theme_str = ", ".join(pos_themes) if pos_themes else "general support"
    else:
        sentiment = "Mixed with concerns (criticism / suspicion)"
        theme_str = ", ".join(neg_themes) if neg_themes else "specific concerns"

    # Create more dynamic, less repetitive recommendations by mixing sentence structures
    if sentiment == "Mostly positive / supportive":
        theme_1 = pos_themes[0] if pos_themes else "overall positivity"
        recs = [
            f"Viewers are highly engaged with {theme_1}. Double down on this style of content in future segments to maintain high retention.",
            f"The positive response to {theme_1} suggests this is a key value prop. Consider moving a highlight of this to the video intro.",
            f"Strong community sentiment around {theme_1}. Use this momentum to pin a comment or ask a specific question related to it."
        ]
        # Use hash of text to pick a stable but "random" recommendation for this specific content
        rec = recs[hash(joined) % len(recs)] + f" (Detected themes: {theme_str})"

    elif sentiment == "Mixed with concerns (criticism / suspicion)":
        theme_1 = neg_themes[0] if neg_themes else "the main concerns"
        recs = [
            f"Address the {theme_1} concerns in a pinned comment to prevent sentiment from bleeding into the next segment.",
            f"Confusion regarding {theme_1} is visible. For future edits, add text overlays or B-roll to clarify these points visually.",
            f"The skepticism around {theme_1} indicates a trust gap. Transparency here would turn critics into advocates."
        ]
        rec = recs[hash(joined) % len(recs)] + f" (Issues: {theme_str})"

    else:
        pos_part = pos_themes[0] if pos_themes else "some good points"
        neg_part = neg_themes[0] if neg_themes else "uncertainty"
        rec = (
            f"Keep a balanced tone: acknowledge {neg_part} while reinforcing {pos_part}. "
            f"Why: the interval shows both positive and unclear reactions, so one clarifying sentence will likely smooth misunderstandings. "
            "After that, highlight the strongest genuine takeaway from the comments."
        )

    return sentiment, rec


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
    fallback_results: List[IntervalAnalysis] = []
    for interval, texts in interval_items:
        sent, rec = heuristic_sentiment_and_recommendation(texts)
        fallback_results.append(
            IntervalAnalysis(
                interval=interval,
                sentiment_analysis=sent,
                recommendation=rec,
                sentiment_breakdown={"positive_percent": 0, "neutral_percent": 0, "negative_percent": 0},
                top_topics=[],
                viewer_reactions=[f"Heuristic detection: {sent}"],
                pain_points=[],
                engagement_drivers=[],
                content_opportunities=[],
                creator_recommendation=rec
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
