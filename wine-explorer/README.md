# Wine Explorer

A voice-enabled web app for exploring a wine dataset using local LLMs. Ask questions by voice or text and get answers in both text and speech (no cloud APIs required, everything runs on your machine).

## Features

- **Voice input** : record a question via your microphone; transcribed on-device with Whisper
- **Text input** : toggle to a keyboard input as an alternative
- **Streaming answers** : responses appear in real time as the LLM generates them
- **Voice output** : answers are spoken aloud sentence-by-sentence via browser TTS
- **Analytical queries** : questions like "average price" or "how many" trigger LLM-generated pandas code executed against the dataset
- **Semantic search (RAG)** : exploratory questions use hybrid retrieval (fuzzy name matching + embedding similarity)
- **Intent extraction** : structured filters (price, color, region, country) are pulled from natural language before retrieval
- **Graceful fallbacks** : intent extraction falls back to regex; analyst code-gen retries with error feedback; analyst falls back to RAG

## Architecture

```
Browser (voice/text)
  │
  ├─ /transcribe ──► Whisper (faster-whisper)
  │
  └─ /query
       ├─ Analytical? ──► analyst.py ──► LLM code-gen ──► exec(pandas) ──┐
       │                                                                  │
       └─ Exploratory? ──► intent.py ──► rag.py (hybrid retrieval) ──┐   │
                                                                     │   │
                                                          llm.py ◄───┘───┘
                                                            │
                                                     StreamingResponse
                                                            │
                                                    Browser (text + TTS)
```

## Stack

| Layer | Tool |
|---|---|
| Speech-to-text | faster-whisper (Whisper `base`) |
| Embeddings | Ollama `qwen3-embedding:8b` |
| Chat LLM | Ollama `qwen2.5:7b` |
| Intent extraction | Ollama `gemma2:9b` |
| Text-to-speech | Browser Web Speech Synthesis |
| Backend | FastAPI + uvicorn |
| Frontend | Single-page vanilla HTML/JS |

## Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the following models pulled:
  ```
  ollama pull qwen3-embedding:8b
  ollama pull qwen2.5:7b
  ollama pull gemma2:9b
  ```

### 2. Install dependencies

```bash
cd wine-explorer
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 3. Add the wine dataset

Place the wine CSV file at:
```
backend/data/wines.csv
```

### 4. Start the server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

On first launch the server will:
1. Load the Whisper `base` model (~150 MB, downloaded once to `~/.cache`)
2. Embed all wines via Ollama and cache the result — this takes about a minute on the first run

### 5. Open the app

Navigate to `http://localhost:8000` in your browser.

## Usage

- **Voice:** Click the microphone button, ask your question, click again to stop. Review the transcription (edit if needed), then click **Send**.
- **Text:** Click the **Type** button to switch to a text input.
- Answers stream into the chat window and are spoken aloud automatically.

## Example questions

- Which are the best-rated wines under $50?
- What do you have from Burgundy?
- What's the most expensive bottle you have?
- Compare the average price of wine in the USA and France.
- How many red wines are there?
- Which bottles would make a good housewarming gift?

## Project structure

```
wine-explorer/
├── backend/
│   ├── main.py          # FastAPI server, /transcribe and /query endpoints
│   ├── analyst.py       # Analytical queries via LLM code generation + pandas
│   ├── rag.py           # Hybrid retrieval (fuzzy + semantic) with embedding cache
│   ├── intent.py        # Structured filter extraction from natural language
│   ├── llm.py           # Streaming chat responses via Ollama
│   ├── transcribe.py    # Whisper speech-to-text
│   └── data/
│       ├── wines.csv
│       └── embeddings_cache.npz
├── frontend/
│   └── index.html       # Single-page app (vanilla HTML/JS/CSS)
└── requirements.txt
```
