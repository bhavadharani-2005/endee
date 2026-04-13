# AI Resume Matcher

## Problem Statement

Recruiters and candidates often struggle to quickly identify which roles best match a resume. This project solves that problem with semantic search and Retrieval-Augmented Generation (RAG), using Endee as the vector database to compare resume content against job profiles.

## Features

- Accepts resume text through a Streamlit interface
- Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Stores job vectors and metadata in Endee
- Retrieves the top 3 semantically similar jobs
- Produces AI insights with OpenAI when available
- Falls back to deterministic rule-based feedback if the API is unavailable
- Handles empty input and no-match scenarios

## Architecture

```text
Resume Input (Streamlit)
        |
        v
SentenceTransformer Embedding
        |
        v
Endee Index: jobs
        |
        v
Top 3 Semantic Matches
        |
        +--> Matching Skills Extraction
        |
        v
RAG Layer (OpenAI or Rule-Based Fallback)
        |
        v
UI Output: roles, skills, missing skills, suggestions
```

## How Endee Is Used

- `embed.py` initializes an Endee client and creates the `jobs` index if it does not exist
- Job descriptions are embedded and upserted into Endee with metadata:
  - `title`
  - `skills`
  - `description`
- `rag.py` embeds the resume text and queries Endee for the top 3 closest matches
- Retrieved metadata is used as context for generating resume feedback
- The app checks Endee connectivity before indexing or searching and shows a friendly UI message if the service is unavailable

## Setup Instructions

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Endee and make sure the API is reachable.

4. Optional environment variables:

```bash
set ENDEE_API_TOKEN=your_endee_token
set ENDEE_BASE_URL=http://localhost:8080/api/v1
set OPENAI_API_KEY=your_openai_api_key
set OPENAI_MODEL=gpt-4.1-mini
set AI_RESUME_MATCHER_CACHE_DIR=your_writable_cache_directory
```

Notes:
- If `ENDEE_API_TOKEN` is not set, the SDK uses unauthenticated development mode.
- If `ENDEE_BASE_URL` is not set, the app uses Endee's default local endpoint.
- If `OPENAI_API_KEY` is not set or the API call fails, the app uses rule-based suggestions.
- No fixed system path is required for model caching. The app resolves cache directories dynamically in this order:
  - `AI_RESUME_MATCHER_CACHE_DIR`
  - project-local `.cache`
  - OS temp directory

## Run Instructions

1. Ingest sample jobs into Endee:

```bash
python embed.py
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. Paste resume text into the UI and click `Analyze Resume`.
