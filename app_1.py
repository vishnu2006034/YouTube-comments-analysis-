import json
import logging
import logging.handlers
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for

# youtube-comment-downloader
from youtube_comment_downloader import YoutubeCommentDownloader

from src.analyzer_local import analyze_intervals


APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def configure_logging() -> None:
    """Configure application logging.

    Logs to both console and a file under `output/`.
    """
    log_dir = OUTPUT_DIR
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Avoid duplicate handlers when reloader runs.
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root.addHandler(file_handler)
    root.addHandler(console_handler)


configure_logging()
logger = logging.getLogger("yt-comments")


def is_probably_youtube_url(url: str) -> bool:
    """Basic URL sanity check for YouTube."""
    if not url:
        return False
    url_l = url.lower().strip()
    return (
        "youtube.com" in url_l
        or "youtu.be" in url_l
    )


def extract_video_id(url: str) -> Optional[str]:
    """Extract 11-character YouTube video ID from supported URL formats."""
    if not url:
        return None

    # Common patterns
    patterns = [
        r"(?:v=)([\w-]{11})",  # watch?v=
        r"youtu\.be/([\w-]{11})",
        r"(?:embed|shorts)/([\w-]{11})",
        r"youtube\.com/.*?/([\w-]{11})",  # fallback-ish
    ]

    for p in patterns:
        m = re.search(p, url, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    # If the user pasted a bare video id
    if re.fullmatch(r"[\w-]{11}", url.strip()):
        return url.strip()

    return None


def search_youtube(query: str) -> Optional[str]:
    """Search YouTube for a query and return the first video ID found.
    
    Tries Google YouTube Data API first, then falls back to scraping results page.
    """
    # 1. Try YouTube Data API if configured
    api_key = os.getenv("YOUTUBE_API_KEY") or os.getenv("utube_KEY")
    if api_key:
        try:
            from googleapiclient.discovery import build
            youtube = build('youtube', 'v3', developerKey=api_key)
            search_response = youtube.search().list(
                q=query,
                part='snippet',
                maxResults=1,
                type='video'
            ).execute()
            if search_response.get('items'):
                return search_response['items'][0]['id']['videoId']
        except Exception as e:
            logger.warning("YouTube API search failed: %s. Falling back to scraping.", e)

    # 2. Fallback to scraping
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        r = requests.get(
            f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}",
            headers=headers,
            timeout=10
        )
        if r.status_code == 200:
            video_ids = re.findall(r'/watch\?v=([\w-]{11})', r.text)
            if video_ids:
                return video_ids[0]
    except Exception as e:
        logger.error("YouTube search scraping failed: %s", e)

    return None


def clean_comments_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove empty comments and duplicates.

    Expected input columns: at least `comment`.
    Output columns: `comment` only (string).
    """
    if df.empty:
        return df

    if "comment" not in df.columns:
        raise ValueError("DataFrame must include a 'comment' column")

    df = df.copy()
    df["comment"] = df["comment"].astype(str)
    df["comment"] = df["comment"].map(lambda s: s.strip() if isinstance(s, str) else "")

    # Remove empty
    df = df[df["comment"].str.len() > 0]

    # Remove duplicates (keep first)
    df = df.drop_duplicates(subset=["comment"], keep="first")

    # Final cleanup
    df = df.reset_index(drop=True)
    return df


@dataclass
class JobStatus:
    state: str  # queued|running|done|error
    progress: int  # 0-100
    video_url: str
    video_id: Optional[str] = None
    message: Optional[str] = None
    extracted_raw: int = 0
    extracted_clean: int = 0
    csv_path: Optional[str] = None
    analysis_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class InMemoryJobStore:
    """Thread-safe in-memory job store for progress polling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobStatus] = {}

    def create(self, video_url: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobStatus(
                state="queued",
                progress=0,
                video_url=video_url,
            )
        return job_id

    def set(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            self._jobs[job_id] = status

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            current = self._jobs[job_id]
            for k, v in kwargs.items():
                setattr(current, k, v)
            self._jobs[job_id] = current

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            return self._jobs.get(job_id)


job_store = InMemoryJobStore()


def download_youtube_comments(
    video_url: str,
    max_comments: int,
    job_id: str,
) -> Tuple[List[str], int]:
    """Download comments using youtube-comment-downloader.

    Returns: (comments_list, extracted_raw_count)
    """
    ycd = YoutubeCommentDownloader()

    logger.info("[job=%s] Starting download for URL=%s", job_id, video_url)

    # youtube-comment-downloader uses generators via dict chunks.
    # It doesn't provide a deterministic progress, so we estimate based on yielded chunks.
    comments: List[str] = []

    extracted_raw = 0
    last_yield_time = time.time()

    # The package may return comment records with keys like 'comment', 'text', etc.
    # We'll normalize to `comment`.
    try:
        for record in ycd.get_comments_from_url(video_url):
            if extracted_raw >= max_comments:
                break

            raw_comment = (
                record.get("comment")
                or record.get("text")
                or record.get("comment_text")
                or ""
            )
            if not isinstance(raw_comment, str):
                raw_comment = str(raw_comment)

            comments.append(raw_comment)
            extracted_raw += 1

            # Progress estimation: map 0..max_comments to 0..90 (cleaning and saving after)
            if extracted_raw % 50 == 0:
                progress = min(90, int((extracted_raw / max_comments) * 90))
                job_store.update(
                    job_id,
                    progress=progress,
                    state="running",
                    message="Downloading comments...",
                )

                now = time.time()
                if now - last_yield_time > 2:
                    logger.info(
                        "[job=%s] Downloaded %d/%d raw comments",
                        job_id,
                        extracted_raw,
                        max_comments,
                    )
                    last_yield_time = now

        return comments, extracted_raw
    except Exception:
        logger.exception("[job=%s] Download failed", job_id)
        raise


def process_job(job_id: str, query_or_url: str, max_comments: int) -> None:
    """Background job runner."""
    job_store.update(
        job_id,
        state="running",
        progress=1,
        message="Resolving input...",
    )

    try:
        video_id = extract_video_id(query_or_url)
        
        if not video_id:
            job_store.update(
                job_id,
                message=f"Searching YouTube for '{query_or_url}'...",
                progress=5,
            )
            video_id = search_youtube(query_or_url)
            
        if not video_id:
            raise ValueError(f"Could not find any YouTube video for query: '{query_or_url}'")

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        job_store.update(
            job_id,
            video_id=video_id,
            video_url=video_url,
            progress=10,
            message="Connecting to YouTube comment stream...",
        )

        comments, extracted_raw = download_youtube_comments(
            video_url,
            max_comments=max_comments,
            job_id=job_id,
        )
        job_store.update(
            job_id,
            extracted_raw=extracted_raw,
            progress=91,
            message="Cleaning comments...",
        )

        df = pd.DataFrame({"comment": comments})
        df_clean = clean_comments_df(df)
        extracted_clean = int(len(df_clean))

        job_store.update(
            job_id,
            extracted_clean=extracted_clean,
            progress=93,
            message="Saving CSV...",
        )

        csv_filename = f"comments_{video_id}.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)

        df_clean.to_csv(csv_path, index=False, encoding="utf-8")

        job_store.update(
            job_id,
            progress=95,
            message="Analyzing comments (interval sentiment)…",
        )

        cleaned_comments = df_clean["comment"].astype(str).tolist()
        analyses = analyze_intervals(cleaned_comments, interval_seconds=60)

        analysis_path = os.path.join(OUTPUT_DIR, f"analysis_{video_id}.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in analyses], f, ensure_ascii=False, indent=2)

        job_store.update(
            job_id,
            state="done",
            progress=100,
            message="Done",
            csv_path=csv_path,
            analysis_path=analysis_path,
            finished_at=time.time(),
        )

    except Exception as e:
        job_store.update(
            job_id,
            state="error",
            progress=100,
            error=str(e),
            message="Error",
            finished_at=time.time(),
        )


def comments_json_response(job_id: str, limit: Optional[int] = None) -> Response:
    status = job_store.get(job_id)
    if not status:
        return jsonify({"error": "Unknown job id"}), 404

    if status.state != "done":
        return jsonify({"error": f"Job not finished (state={status.state})"}), 409

    if not status.csv_path or not os.path.exists(status.csv_path):
        return jsonify({"error": "CSV not found for completed job"}), 500

    df = pd.read_csv(status.csv_path)
    comments = df["comment"].astype(str).tolist()

    if limit is not None:
        comments = comments[:limit]

    payload = {
        "video_id": status.video_id,
        "video_url": status.video_url,
        "total_extracted_raw": status.extracted_raw,
        "total_extracted_clean": status.extracted_clean,
        "comments": comments,
        "csv_path": status.csv_path,
    }
    return jsonify(payload)


# ---------------------- Flask App ----------------------
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home() -> str:
    return render_template("frontend.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> Response:
    user_input = (request.form.get("query") or "").strip()

    if not user_input:
        return render_template(
            "error.html",
            error_title="Empty Input",
            error_message="Please enter a YouTube video URL or a search term (e.g. video title).",
            error_details=None,
            recoverable=False,
        ), 400

    job_id = job_store.create(video_url=user_input)

    try:
        max_comments = int(os.getenv("MAX_COMMENTS", "2000"))
    except (ValueError, TypeError):
        max_comments = 2000

    t = threading.Thread(target=process_job, args=(job_id, user_input, max_comments), daemon=True)
    t.start()

    return redirect(url_for("progress_page", job_id=job_id))


@app.route("/progress/<job_id>", methods=["GET"])
def progress_page(job_id: str) -> str:
    return render_template("progress.html", job_id=job_id)


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id: str) -> Response:
    status = job_store.get(job_id)
    if not status:
        return jsonify({"error": "Unknown job id"}), 404

    return jsonify(status.to_json())


@app.route("/result/<job_id>", methods=["GET"])
def job_result(job_id: str) -> Response:
    limit_raw = request.args.get("limit")
    limit = int(limit_raw) if limit_raw and limit_raw.isdigit() else None
    return comments_json_response(job_id=job_id, limit=limit)


@app.route("/analysis/<job_id>", methods=["GET"])
def job_analysis(job_id: str) -> Response:
    status = job_store.get(job_id)
    if not status:
        return jsonify({"error": "Unknown job id"}), 404

    analysis_path = status.analysis_path
    if status.state != "done" or not analysis_path or not os.path.exists(analysis_path):
        return render_template(
            "error.html",
            error_title="Analysis not ready",
            error_message="The comment analysis is not ready yet. Please wait or extract again.",
            error_details=None,
            recoverable=True,
            retry_url=url_for("progress_page", job_id=job_id),
        ), 409

    with open(analysis_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    csv_path = status.csv_path
    return render_template(
        "analysis.html",
        job_id=job_id,
        video_id=status.video_id,
        count=len(data),
        data=data,
        csv_path=csv_path,
    )


@app.route("/download/<job_id>", methods=["GET"])
def job_csv_download(job_id: str) -> Response:
    status = job_store.get(job_id)
    if not status or status.state != "done" or not status.csv_path:
        return jsonify({"error": "CSV not ready"}), 409

    return send_file(
        status.csv_path,
        as_attachment=True,
        download_name=os.path.basename(status.csv_path),
    )


@app.errorhandler(404)
def not_found(e: Exception) -> Response:
    return render_template(
        "error.html",
        error_title="Not found",
        error_message="The requested page does not exist.",
        error_details=str(e),
        recoverable=False,
    ), 404


@app.errorhandler(500)
def internal_error(e: Exception) -> Response:
    logger.exception("Unhandled exception")
    return render_template(
        "error.html",
        error_title="Server error",
        error_message="An internal error occurred.",
        error_details=str(e),
        recoverable=True,
        retry_url=url_for("home"),
    ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
