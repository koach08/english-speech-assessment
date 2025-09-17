# English Speech Assessment Tool

A tool designed for educators to evaluate English pronunciation and speaking skills. It leverages Azure Speech Services for pronunciation assessment (Accuracy, Fluency, Prosody) and OpenAI for generating detailed Japanese feedback (180-250 characters, 3 improvement suggestions). Supports YouTube/Google Drive links, MP3/MP4 files, and CSV input for batch processing. Integrates with Moodle for efficient grading. Developed with assistance from ChatGPT, Grok, Claude, Azure, and OpenAI for Hokkaido University and other educational institutions.

## Features
- **Input Options**: YouTube (limited sharing), Google Drive links, MP3/MP4 files, CSV (batch processing).
- **Pronunciation Assessment**: Uses Azure Speech to evaluate Accuracy, Fluency, and Prosody, with optional target text.
- **Feedback Generation**: OpenAI generates Japanese feedback with improvement suggestions, customizable via `config.yaml`.
- **Moodle Integration**: Exports grading_worksheet.csv for easy grade import.
- **Customization**: Adjustable evaluation criteria (`rubric.example.json`) and weights (`config.yaml`).
- **Open Source**: MIT License, free to use and modify.

## Setup Instructions
1. Place folder in Desktop (`/Users/your_username/Desktop/english_assessment`).
2. Open Terminal: `/Applications/Utilities/Terminal.app`
3. Navigate to folder: `cd /Users/your_username/Desktop/english_assessment`
4. Remove old venv: `rm -rf venv`
5. Check Python 3.12: `/opt/homebrew/bin/python3.12 --version` (install if needed: `brew install python@3.12`)
6. Create venv: `/opt/homebrew/bin/python3.12 -m venv venv`
7. Activate venv: `source /Users/your_username/Desktop/english_assessment/venv/bin/activate`
8. Install libraries: `pip install --upgrade pip --no-cache-dir; pip install streamlit yt-dlp requests PyYAML azure-cognitiveservices-speech openai moviepy pydub gdown --no-cache-dir`
9. Install FFmpeg: `brew install ffmpeg`
10. Get Azure/OpenAI keys: Azure[](https://portal.azure.com), OpenAI[](https://platform.openai.com/api-keys)
11. Edit `run_app.sh` with keys:
    ```sh
    #!/bin/bash
    export AZURE_SPEECH_KEY=your_azure_key_here
    export AZURE_SPEECH_REGION=japaneast
    export OPENAI_API_KEY=your_openai_key_here
    source /Users/your_username/Desktop/english_assessment/venv/bin/activate
    cd /Users/your_username/Desktop/english_assessment
    python -m streamlit run app.py
