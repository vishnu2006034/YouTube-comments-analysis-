from flask import Flask, render_template, request,url_for
import requests
from googleapiclient.discovery import build
import pandas as pd
import re
import google.generativeai as genai
import json
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydata.db'
db = SQLAlchemy(app)

YOUTUBE_API_KEY ='your youtube apikey'
GEMINI_API_KEY = 'your gemini apikey'
genai.configure(api_key=GEMINI_API_KEY)

class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id=db.Column(db.String, unique=True)
    thumbnail=db.Column(db.String,default='default.img')
    moviename = db.Column(db.String, nullable=False )
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
                'video_id': self.video_id,
                # 'title': self.title,
                'moviename': self.moviename,
                'thumbnail': self.thumbnail,
                # 'comments_csv': self.comments_csv,
                'interval': self.time_strap,
                'sentiment_analysis': self.sentiment,
                'recommendation': self.recommendation
            }
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
        search_query = request.form['query']

        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    
        search_response = youtube.search().list(
            q=search_query,
            part='snippet',
            maxResults=1,
            type='video'
        ).execute()

        if not search_response['items']:
            return render_template('index.html', error="No video found.")

        video_id = search_response['items'][0]['id']['videoId']
        video_title = search_response['items'][0]['snippet']['title']
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        response = requests.get(thumbnail_url)

        if response.status_code == 200:
            with open(f"static/{video_id}_thumbnail.jpg", "wb") as f:
                f.write(response.content)
            thumbnail_path = f"static/{video_id}_thumbnail.jpg"
        else:
            thumbnail_path = None

        next_page_token = None
        while True:
            comment_response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            for item in comment_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append({'Comment': comment})

            next_page_token = comment_response.get('nextPageToken')
            if not next_page_token:
                break

        
        df = pd.DataFrame(comments)
        df['extracted_timestamps'] = df['Comment'].apply(extract_timestamps)
        df_filtered = df[df['extracted_timestamps'].notnull()][['Comment', 'extracted_timestamps']]
        csv_text = df_filtered.to_string(index=False)

    
        prompt = f"""
You are a strict JSON generator.

Analyze the sentiment of YouTube comments in the following CSV data, segmented into roughly 15-second intervals from the beginning. Return a pure JSON array of objects only. No explanations or formatting.

Each object must contain:
* "interval": "[Start Time]-[End Time]" (e.g., "0-15s")
* "sentiment_analysis": "[Sentiment summary]"
* "recommendation": "[Suggested action]"

Here is the data:
{csv_text}
"""

        model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-exp-image-generation")

        try:
            response = model.generate_content(prompt)

            if not response or not hasattr(response, 'candidates') or not response.candidates:
                return render_template('index.html', error="No valid response from Gemini.")

            response_text = response.candidates[0].content.parts[0].text.strip()

            raw = response_text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]

            data = json.loads(raw)

            sentiment_summary = []
            recommendations = []

            for item in data:
                sentiment_summary.append(f"{item.get('interval')}: {item.get('sentiment_analysis')}")
                recommendations.append(f"{item.get('interval')}: {item.get('recommendation')}")

            sentiment_combined = " | ".join(sentiment_summary)
            recommendation_combined = " | ".join(recommendations)

            new_movie = Movie(
                thumbnail=video_id + '_thumbnail.jpg',
                video_id=video_id,
                moviename=video_title,
                
            )
            db.session.add(new_movie)
            db.session.commit()
            for item in data:
                row = SentimentAnalysis(
                    movie_id=new_movie.id,
                    time_strap=item.get('interval', ''),
                    sentiment=item.get('sentiment_analysis', ''),
                    recommendation=item.get('recommendation', '')
                )
                db.session.add(row)

            db.session.commit()

            
            return render_template('info.html', data=data, moviename=search_query,video_id=video_id,thumbnail_url=thumbnail_url)

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            return render_template('front end.html', error=f"Gemini response error: {str(e)}")
    content=Movie.query.all()
    return render_template('front end.html',content=content)

@app.route('/video/<video_id>')
def show_video(video_id):
    video = Movie.query.filter_by(video_id=video_id).first_or_404()
    sentiment_data = SentimentAnalysis.query.filter_by(movie_id=video.id).all()
    return render_template('video_detail.html', video=video,data=sentiment_data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
