#!/bin/bash
source .venv/bin/activate
# Use port 8501 as a fallback if $PORT is not set
streamlit run main.py --server.port ${PORT:-8501} --server.headless true --server.enableCORS=false --server.enableXsrfProtection=false