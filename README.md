# 🎬 YouTube Time Strip Comment Analyzer & Sentiment-Based Recommendation System

## 📌 Project Overview

This project analyzes YouTube video comments that mention specific **time stamps** (e.g., `0:15`, `1:30`). The video is segmented into **15-second intervals**, and any timestamp-related comment is processed using **Google Gemini API** to extract **sentiment insights** and provide **recommendations** for each interval.

The output stores:
- The **sentiments** across intervals in a single-column text field
- The **recommendations** similarly in another single-column text field

## 🚀 Features

- ⏱ Extracts and maps timestamped comments to 15-second video intervals
- 🤖 Uses **Gemini API (Google Generative AI)** for sentiment analysis and suggestion generation
- 📊 Aggregates results in structured JSON and stores in a SQLite database
- 📁 Sentiment and recommendation outputs are stored in simplified single text fields

## 🧠 Use Cases

- Content creators identifying sections of a video that receive negative feedback
- Video editors targeting specific parts of the content for improvement
- Automated analytics for educational or promotional content

## 🛠️ Technologies Used

- **Flask** – for the web interface
- **Google YouTube Data API** – to fetch video comments
- **Google Generative AI (Gemini)** – to analyze sentiments and generate suggestions
- **Pandas** – for data manipulation
- **SQLite** with SQLAlchemy – for local data storage

## 👥 Contributors

## Contributors
This project was developed as part of a hackathon by Team Hawkeye:
- Vishnu R
- Tharun Kumar V
- Vignesh V
- Suriya K
- Vishnupriya D

## 📁 Project Structure
youtube-time-analyzer/
│
├── templates/
│ ├── front end.html
│ ├── info.html
│
├── app.py
├── requirements.txt
└── README.md


## 💡 How It Works

1. User enters a **search query** on the web interface
2. The first matching YouTube video is fetched via API
3. All top-level comments are extracted and parsed for timestamps
4. Gemini API analyzes the filtered comments and returns:
    - Interval-wise **sentiment summary**
    - Interval-wise **recommendation**
5. Both sentiment and recommendation data are flattened and stored in the database as **single-column strings**

| Time Strap          | Sentiment (text)                       | Recommendation (text)                               |
| 15-30s | audience feel bad here|  Keep it engaging... |


## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishnu2006034/youtube-time-analyzer.git
   cd youtube-time-analyzer

2. Install requirement file
  pip install -r requirements.txt

4. Create your own api
   YOUTUBE_API_KEY=your_youtube_api_key
   GEMINI_API_KEY=your_gemini_api_key


