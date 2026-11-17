# Hindi RAG: Agni Ki Udaan

A reproducible Retrieval-Augmented Generation (RAG) pipeline built over the supplied Hindi study guide on A.P.J. Abdul Kalam. It supports Hindi questions and English-to-Hindi cross-lingual retrieval.

## Features

- Hindi PDF extraction using PyMuPDF.
- Unicode NFC normalization for Devanagari text.
- Multilingual semantic embeddings.
- Persistent ChromaDB vector storage.
- Local LLM generation using Ollama and Qwen 2.5 7B.
- Optional OpenAI API-based generation.
- Automatic citations containing page, section, and chunk ID.
- Six-question evaluation workflow with JSON output.

## Architecture

1. **PDF extraction:** PyMuPDF extracts text page by page while retaining page numbers.
2. **Normalization:** Unicode NFC normalization and whitespace cleanup preserve Devanagari characters.
3. **Chunking:** Paragraph-first chunking with approximately 850-character chunks and 120-character overlap. Chunks remain page-scoped.
4. **Embeddings:** `intfloat/multilingual-e5-base` with `passage:` and `query:` prefixes.
5. **Vector database:** ChromaDB stores document chunks, embeddings, and citation metadata locally in `.chroma/`.
6. **Retrieval:** Semantic search retrieves relevant Hindi passages for Hindi and English questions.
7. **Generation:** Ollama with Qwen 2.5 7B generates answers from retrieved evidence. If `OPENAI_API_KEY` is configured, the OpenAI generation path is used instead.

Similarity scores are ranking signals, not calibrated probabilities.

## Requirements

- Python 3.10 or later
- Ollama installed and running for local generation
- Internet access for initial model downloads

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the local generation model:

```bash
ollama pull qwen2.5:7b
```

Set the model:

```bash
export OLLAMA_MODEL=qwen2.5:7b
```

The embedding model downloads automatically from Hugging Face on first run.

## Build the vector index

From the project root:

```bash
python3 -m src.rag ingest --reset
```

## Ask questions

Hindi example:

```bash
python3 -m src.rag ask "कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?"
```

English example:

```bash
python3 -m src.rag ask "Which institution did Kalam attend to study aeronautical engineering?"
```

## Run the evaluation

```bash
python3 -m src.rag test
```

The command prints answers and citations and saves the machine-readable results to:

`outputs/test_run.json`

Each citation includes the source page, section, chunk ID, and retrieval similarity score.

## Optional OpenAI configuration

Local Ollama generation is used when no OpenAI API key is configured.

To use OpenAI instead:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o-mini"
```

Never commit API keys or `.env` files to version control.

## Evaluation and limitations

The six-question test is a reproducible smoke test, not a complete retrieval benchmark.

Further evaluation could include Recall@1, Recall@3, Mean Reciprocal Rank (MRR), alternative multilingual embedding models, and cross-encoder reranking.

The current PDF extraction relies on selectable text. It is not OCR. Table layout may be flattened during extraction, and section detection is heuristic.

Generated answers should be checked against their cited passages. Retrieval similarity scores do not guarantee answer correctness.

## Repository structure

```text
hindi_kalam_rag/
├── data/
│   └── Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf
├── outputs/
│   ├── answers.md
│   └── test_run.json
├── src/
│   ├── __init__.py
│   └── rag.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Reproducibility

Follow the setup instructions, download the required models, build the index, and run the six-question test.

The ChromaDB index can be recreated using `ingest --reset`.
