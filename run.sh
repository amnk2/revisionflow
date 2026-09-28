#!/bin/bash
# Starts RevisionFlow and keeps the Whisper and Hugging Face model files in the project's .cache folder.
cd "$(dirname "$0")"
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
source venv/bin/activate
streamlit run app.py
