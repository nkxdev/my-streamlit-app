# AI Image Editor (Streamlit)

A Streamlit web app where users:
- upload an image
- describe desired edits in plain language
- get AI-assisted image edits automatically

## Features
- Natural-language editing prompt
- Optional Anthropic-powered instruction parsing
- Safe fallback local parser (works without API key)
- Built-in edits: brightness, contrast, saturation, sharpness, blur, grayscale, rotate, flip
- Download edited image as PNG

## Run locally
```bash
pip install -r requirements.txt
streamlit run /home/runner/work/my-streamlit-app/my-streamlit-app/app.py
```

## Optional API setup
Set `ANTHROPIC_API_KEY` in your environment or provide it in the sidebar inside the app.
