# 🎬 YouTube Comment Analyzer & Sentiment-Based Recommendation System

## 📌 Project Overview

This project analyzes YouTube video comments that mention specific **timestamps** (e.g., `0:15`, `1:30`). The video is segmented into intervals, and comments are processed using **Google Gemini API** to extract **sentiment insights** and provide **recommendations** for each interval.

If the Gemini API key is not configured, the app falls back to a local heuristic analyzer.

## 🚀 Features

- ⏱ Extracts and maps timestamped comments to video intervals
- 🤖 Uses **Gemini API (Google Generative AI)** for sentiment analysis and suggestion generation
- 📊 Aggregates results in structured JSON
- 📁 Exports cleaned comments as CSV with download support
- 🔄 Real-time progress tracking via polling
- 🛡️ Heuristic fallback when Gemini API is unavailable

## 🧠 Use Cases

- Content creators identifying sections of a video that receive negative feedback
- Video editors targeting specific parts of the content for improvement
- Automated analytics for educational or promotional content

## 🛠️ Technologies Used

- **Flask** – web interface
- **youtube-comment-downloader** – comment extraction (no YouTube API key required)
- **Google Generative AI (Gemini)** – sentiment analysis and suggestions
- **Pandas** – data manipulation

## 👥 Contributors

This project was developed as part of a hackathon by Team Hawkeye:
- Vishnu R
- Tharun Kumar V
- Vignesh V
- Suriya K
- Vishnupriya D

## 📁 Project Structure

```
youtube-comments-analysis/
├── app_1.py                  # Main Flask application
├── src/
│   ├── __init__.py
│   └── analyzer_local.py     # Gemini + heuristic sentiment analyzer
├── templates/
│   ├── frontend.html         # Home page (URL input)
│   ├── progress.html         # Real-time progress tracker
│   ├── analysis.html         # Interval sentiment results
│   └── error.html            # Error display page
├── static/
│   └── youtube.png           # Favicon
├── output/                   # Generated CSVs, analysis JSON, logs
├── .env                      # API keys (not committed)
├── requirements.txt
├── LICENSE
└── README.md
```

## 💡 How It Works

1. User pastes a **YouTube video URL** on the web interface
2. Comments are extracted using `youtube-comment-downloader` (no API key needed)
3. Comments are cleaned (deduplicated, empty removed) and saved to CSV
4. Comments with timestamps are grouped into intervals
5. Gemini API analyzes each interval and returns:
   - Interval-wise **sentiment breakdown** (positive/neutral/negative %)
   - Interval-wise **recommendation** for the creator
6. Results are displayed in a table and the CSV is available for download

| Interval | Sentiment | Recommendation |
|----------|-----------|----------------|
| 0-60s    | Pos: 70% \| Neu: 20% \| Neg: 10% | Keep the engaging intro style... |

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishnu2006034/YouTube-comments-analysis-.git
   cd YouTube-comments-analysis-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**

   Copy `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   Get a key at: https://aistudio.google.com/app/apikey

4. **Run the app**
   ```bash
   python app_1.py
   ```
   Open http://localhost:5000 in your browser.
