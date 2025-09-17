#!/bin/bash
    export AZURE_SPEECH_KEY=your_azure_key_here
    export AZURE_SPEECH_REGION=japaneast
    export OPENAI_API_KEY=your_openai_key_here
    source /Users/your_username/Desktop/english_assessment/venv/bin/activate
    cd /Users/your_username/Desktop/english_assessment
    python -m streamlit run app.py