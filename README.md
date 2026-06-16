# ActionFlow 
ActionFlow is an AI-powered meeting assistant that automatically extracts action items, assignees, and deadlines from meeting transcripts and audio recordings.

## Features
*  Extract action items from text transcripts
*  Transcribe audio files using Groq Whisper
*  AI-powered action item detection
*  Automatic assignee identification
*  Deadline extraction
*  Editable action items table
*  Export results as CSV and JSON

## Tech Stack
* Python
* Streamlit
* Groq API
* spaCy
* Pandas
* Dateparser

## How to Run

1. Clone the repository
```bash
git clone <repository-url>
cd ActionFlow
```
2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the application
```bash
streamlit run app.py
```
5. Enter your Groq API Key in the sidebar and start extracting action items.

## Use Case
ActionFlow helps teams save time by automatically converting meeting discussions into actionable tasks with clear ownership and deadlines.

FlowZint AI Hackathon 2026
Built as a solo project for the FlowZint AI Hackathon 2026.
