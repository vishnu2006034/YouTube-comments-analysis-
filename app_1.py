from flask import Flask, render_template, request, url_for
import requests
from googleapiclient.discovery import build
import pandas as pd
import os

# --- Performance caps (reduce API + prompt size) ---
MAX_COMMENT_PAGES = int(os.getenv("MAX_COMMENT_PAGES", "3"))  # pages of commentThreads()
MAX_TOTAL_COMMENTS = int(os.getenv("MAX_TOTAL_COMMENTS", "300"))  # hard cap on collected comments
MAX_TIMESTAMPED_ROWS = int(os.getenv("MAX_TIMESTAMPED_ROWS", "120"))  # rows included in Gemini prompt

import re
import google.generativeai as genai
import json
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydata.db'
db = SQLAlchemy(app)

YOUTUBE_API_KEY = os.getenv("utube_KEY")
GEMINI_API_KEY = os.getenv("gemini_KEY")
genai.configure(api_key=GEMINI_API_KEY)


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String, unique=True)
    thumbnail = db.Column(db.String, default='default.img')
    moviename = db.Column(db.String, nullable=False)
    sentiments = db.relationship('SentimentAnalysis', backref='movie', lazy=True)


class SentimentAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movie.id'), nullable=False)
    time_strap = db.Column(db.String, nullable=False)
    sentiment = db.Column(db.Text, nullable=False)
    recommendation = db.Column(db.Text, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'movie_id': self.movie_id,
            'time_strap': self.time_strap,
            'sentiment': self.sentiment,
            'recommendation': self.recommendation,
        }


def extract_video_id(query):
    """
    Returns an 11-character YouTube video ID if the query is any recognised
    YouTube URL format, otherwise returns None so the caller can fall back to
    a keyword search.

    Supported formats:
      https://www.youtube.com/watch?v=VIDEO_ID
      https://www.youtube.com/watch?v=VIDEO_ID&t=30s   (extra params)
      https://youtu.be/VIDEO_ID
      https://www.youtube.com/embed/VIDEO_ID
      https://www.youtube.com/shorts/VIDEO_ID
      https://m.youtube.com/watch?v=VIDEO_ID
    """
    patterns = [
        r'(?:v=)([\w-]{11})',            # ?v=ID  (watch URLs)
        r'youtu\.be/([\w-]{11})',         # youtu.be/ID
        r'(?:embed|shorts)/([\w-]{11})',  # embed/ or shorts/ID
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None


def extract_timestamps(text):
    if pd.isnull(text):
        return None
    pattern = r'\b\d{1,2}:\d{2}(?:\s?[APMapm]{2})?\b'
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches if matches else None


@app.route('/', methods=['GET', 'POST'])
def index():
    comments = []
    video_title = ""

    if request.method == 'POST':
        # Give the UI immediate feedback (AJAX-friendly; shown if you add a loader page)

        search_query = request.form['query']

        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        # ── Detect whether the input is a direct YouTube URL ──────────────────
        video_id = extract_video_id(search_query)

        if video_id:
            # Fetch video metadata directly — no search quota used
            try:
                video_response = youtube.videos().list(
                    part='snippet',
                    id=video_id
                ).execute()
            except Exception as e:
                return render_template(
                    'error.html',
                    error_title='YouTube API error',
                    error_message='Could not fetch video details from YouTube. Try again in a moment or use a different video.',
                    error_details=str(e),
                    recoverable=True,
                    retry_url=url_for('index'),
                )

            if not video_response.get('items'):
                return render_template(
                    'error.html',
                    error_title='Video not found',
                    error_message='Please check the URL and try again.',
                    recoverable=False,
                )

            video_title = video_response['items'][0]['snippet']['title']

        else:
            # Fall back to keyword search
            search_response = youtube.search().list(
                q=search_query,
                part='snippet',
                maxResults=1,
                type='video'
            ).execute()

            if not search_response.get('items'):
                return render_template(
                    'error.html',
                    error_title='No video found',
                    error_message='Your search didn\'t return a video. Try a different title/topic.',
                    recoverable=False,
                )

            video_id = search_response['items'][0]['id']['videoId']
            video_title = search_response['items'][0]['snippet']['title']

        # ── Thumbnail ─────────────────────────────────────────────────────────
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        response = requests.get(thumbnail_url)

        if response.status_code == 200:
            with open(f"static/{video_id}_thumbnail.jpg", "wb") as f:
                f.write(response.content)
            thumbnail_path = f"static/{video_id}_thumbnail.jpg"
        else:
            thumbnail_path = None

        # ── Fetch limited comments (performance cap) ─────────────────────────
        next_page_token = None
        page_count = 0

        while True:
            if page_count >= MAX_COMMENT_PAGES or len(comments) >= MAX_TOTAL_COMMENTS:
                break

            comment_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            for item in comment_response.get('items', []):
                if len(comments) >= MAX_TOTAL_COMMENTS:
                    break
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append({'Comment': comment})

            page_count += 1
            next_page_token = comment_response.get('nextPageToken')
            if not next_page_token:
                break


        # ── Extract timestamped comments and build prompt (no Pandas) ────────
        timestamped_rows = []
        for c in comments:
            ts = extract_timestamps(c.get('Comment'))
            if not ts:
                continue
            # Keep the comment only once, even if multiple timestamps are present
            # (Gemini will infer interval impact from the text + timestamps).
            timestamped_rows.append((ts, c.get('Comment', '')[:800]))
            if len(timestamped_rows) >= MAX_TIMESTAMPED_ROWS:
                break

        # Compact text format to minimize prompt size
        # Example: "['0:15'] :: Comment text" per line
        lines = []
        for ts_list, comment_text in timestamped_rows:
            lines.append(f"{ts_list} :: {comment_text}")

        csv_text = "\n".join(lines)

        # If no timestamped comments, avoid calling Gemini with an empty prompt.
        if not csv_text.strip():
            return render_template(
                'error.html',
                error_title='No timestamps found',
                error_message='In the sampled comments, no time references (like 2:15 or 10:30) were detected. Try a different video.',
                recoverable=True,
                retry_url=url_for('index'),
            )



        prompt = f"""
You are a strict JSON generator.

Analyze the sentiment of YouTube comments in the following data. Each line contains timestamp mentions (if any) plus the related comment text.
Segment the result into roughly 60-second intervals from the beginning. Return a pure JSON array of objects only. No explanations or formatting.

Each object must contain:
* "interval": "[Start Time]-[End Time]" (e.g., "0-60s")
* "sentiment_analysis": "[Sentiment summary]"
* "recommendation": "[Suggested action]"

Here is the data:
{csv_text}
"""


        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

        try:
            response = model.generate_content(prompt)


            if not response or not hasattr(response, 'candidates') or not response.candidates:
                return render_template(
                    'error.html',
                    error_title='Gemini response error',
                    error_message='The AI did not return valid analysis. Try again with a different video.',
                    error_details='Missing Gemini candidates/content/parts.',
                    recoverable=True,
                    retry_url=url_for('index'),
                )

            response_text = response.candidates[0].content.parts[0].text.strip()

            raw = response_text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]

            data = json.loads(raw)

            # ── Persist to DB (avoid duplicate video_id) ────────────────────────
            movie = Movie.query.filter_by(video_id=video_id).first()
            if not movie:
                movie = Movie(
                    thumbnail=video_id + '_thumbnail.jpg',
                    video_id=video_id,
                    moviename=video_title,
                )
                db.session.add(movie)
                db.session.commit()

            # Remove previous analysis rows for this movie (so re-runs replace old output)
            SentimentAnalysis.query.filter_by(movie_id=movie.id).delete()
            db.session.commit()

            for item in data:
                row = SentimentAnalysis(
                    movie_id=movie.id,
                    time_strap=item.get('interval', ''),
                    sentiment=item.get('sentiment_analysis', ''),
                    recommendation=item.get('recommendation', '')
                )
                db.session.add(row)

            db.session.commit()

            return render_template(
                'info.html',
                data=data,
                moviename=movie.moviename,
                video_id=video_id,
                thumbnail_url=thumbnail_url
            )


        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            return render_template(
                'error.html',
                error_title='Gemini JSON parsing error',
                error_message='The AI response was not in the expected format. Try again with a different video.',
                error_details=str(e),
                recoverable=True,
                retry_url=url_for('index'),
            )

    content = Movie.query.all()
    return render_template('frontend.html', content=content)


@app.route('/video/<video_id>')
def show_video(video_id):
    video = Movie.query.filter_by(video_id=video_id).first_or_404()
    sentiment_data = SentimentAnalysis.query.filter_by(movie_id=video.id).all()
    return render_template('video_detail.html', video=video, data=sentiment_data)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
