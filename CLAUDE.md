# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MeshWiki is a Python application that provides offline Wikipedia access over Meshtastic LoRa mesh radio. Users send questions via Meshtastic (prefixed with `?`), the app performs RAG (Retrieval-Augmented Generation) against a local Wikipedia vector database (ChromaDB + sentence-transformers), generates an answer via a local LLM (Ollama), and sends the response back over Meshtastic — chunked to fit the 228-byte packet limit.

**Target use case**: Reliable information access during internet outages (e.g., cyclones in La Réunion).

## Build & Run Commands

```bash
# Install dependencies
pip install -e .

# Run the application
python -m meshwiki

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_chunker.py

# Run a specific test
pytest tests/test_chunker.py::test_function_name -v
```

**External requirements**: Ollama must be running locally (`http://localhost:11434`) with a model pulled (e.g., `ollama pull mistral`). A Meshtastic module must be connected via USB serial or BLE.

## Architecture

```
[Meshtastic radio] ↔ [meshtastic_bridge.py] → [rate_limiter.py] → [rag.py] → [llm.py] → [chunker.py] → [Meshtastic radio]
                                                                      ↓
                                                               [ChromaDB / sentence-transformers]
```

### Module Responsibilities

- **main.py** — Entry point. Loads config, checks Ollama/ChromaDB availability, triggers initial Wikipedia download if no index exists, starts the Meshtastic listener and the background update scheduler.
- **meshtastic_bridge.py** — Listens for `TEXT_MESSAGE_APP` messages with the trigger prefix (`?`), checks rate limits, dispatches to RAG pipeline, sends chunked responses with configurable delay between packets.
- **rag.py** — Encodes the question, performs semantic search in ChromaDB (top 3-5 chunks), constructs system+user prompt with Wikipedia context, calls the LLM, returns the answer.
- **llm.py** — HTTP POST to Ollama's `/api/generate` endpoint. Stream disabled, low temperature (0.1) for factual answers.
- **chunker.py** — Splits responses into ≤220-byte UTF-8 chunks. Numbered `[1/N]` format. Max 5 chunks, truncates with `... [tronqué]` if exceeded. Never splits mid-word.
- **rate_limiter.py** — Thread-safe sliding window rate limiter per Meshtastic node ID. Returns a French denial message with wait time when limit exceeded.
- **wikipedia_indexer.py** — Reads `.zim` files (Kiwix format via `libzim`), extracts/cleans articles, chunks at ~500 tokens with 50-token overlap, generates embeddings (`BAAI/bge-m3`), stores in ChromaDB.
- **wikipedia_updater.py** — Scrapes Kiwix download page for latest `wikipedia_fr_all_mini_*.zim`, downloads with resume support, performs safe collection swap (new → active, old as backup) so the app never has downtime.

### Key Design Constraints

- **228-byte Meshtastic packet limit**: All user-facing messages (responses, error messages, rate limit denials) must fit within this. The chunker uses a 220-byte safety margin.
- **100% offline after setup**: Only the Wikipedia updater needs internet, and it's optional/periodic.
- **Safe re-indexation**: New index is built in a temporary ChromaDB collection; the active collection is only swapped after successful completion. Failed indexation preserves the existing index.
- **Thread safety**: Rate limiter uses `threading.Lock`. Background updates run in a separate thread with lowered priority.

## Configuration

All settings are in `config.yaml` at the project root. Key sections: `meshtastic` (connection, trigger prefix, response delay), `rate_limiting`, `embeddings` (model, chunk size), `vectordb`, `ollama` (model, temperature, max tokens), `updater` (interval, allowed hours, Kiwix patterns).

## Implementation Order

The plan specifies a phased build order, each phase with corresponding tests:
1. chunker → 2. rate_limiter → 3. llm → 4. wikipedia_indexer → 5. rag → 6. wikipedia_updater → 7. meshtastic_bridge → 8. main

## Language & Locale

All user-facing messages (responses, errors, rate limit denials) are in **French**. Code, comments, and variable names are in English. The embedding model is multilingual (`BAAI/bge-m3`, 8192-token context, 1024-dim embeddings).

## Synchronisation avec la version Android

Le projet existe en deux versions : Python (ce repo) et Android (`MeshWiki Android`). Sauf mention contraire, toute modification d'un prompt LLM (system prompt, user prompt) doit être répercutée dans les deux versions. Les prompts se trouvent dans :
- **Python** : `meshwiki/rag.py` (SYSTEM_PROMPT, SYSTEM_PROMPT_KIWIX, SYSTEM_PROMPT_KIWIX_PERMANENT)
- **Android** : `app/src/main/java/com/meshwiki/rag/RagEngine.kt` (SYSTEM_PROMPT, SYSTEM_PROMPT_KIWIX)

## Commit Message Format

```
<Short summary of the main change>

- file1 : change description
- file2 :
  change1
  change2
- file3 : change description
```
