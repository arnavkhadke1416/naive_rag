# Naive RAG — Multi-PDF Question Answering

A local, command-line Retrieval-Augmented Generation (RAG) application for asking questions across a collection of PDF documents. It extracts text page by page, indexes overlapping chunks from all PDFs in one local FAISS vector store, retrieves relevant passages, and asks Google Gemini to answer using that evidence with filename-and-page citations.

The project is intentionally a small, backend-only example: it has no web UI, HTTP API, user accounts, or chat-history storage.

## Contents

- [What it does](#what-it-does)
- [Architecture and data flow](#architecture-and-data-flow)
- [Technology](#technology)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configure](#configure)
- [Build or rebuild the index](#build-or-rebuild-the-index)
- [Ask questions](#ask-questions)
- [Tests and evaluation](#tests-and-evaluation)
- [Limitations and data handling](#limitations-and-data-handling)
- [Troubleshooting](#troubleshooting)

## What it does

- Discovers every `.pdf` file directly inside the configured documents folder; filenames are not hardcoded.
- Extracts text from each PDF page and retains the source filename and one-based page number as metadata.
- Splits the extracted pages into overlapping character-based chunks and embeds them with a local Sentence Transformers model.
- Combines chunks from the entire collection in a single FAISS index. A question can retrieve evidence from more than one PDF.
- Sends the question and retrieved excerpts to Gemini to generate a grounded answer.
- Requires citations in the answer to match retrieved filenames and page numbers. It abstains if no passages are retrieved or the answer has no valid retrieved-source citations.
- Instructs Gemini to describe conflicting evidence and cite both sources rather than silently choosing between them.

## Architecture and data flow

### Indexing (run when the source PDFs or indexing settings change)

```text
PDF files in documents/
        |
        v
Discover PDFs and extract text page by page (pypdf)
        |
        v
Attach source filename, one-based page, and document identity
        |
        v
Split each page into overlapping chunks
        |
        v
Embed chunks locally (all-MiniLM-L6-v2)
        |
        v
Save one combined FAISS index in vectorstore/
```

Run `python -m scripts.build_index` to build or replace the index. The script reports the PDF, page, and chunk counts. Empty pages are skipped and logged; pages with very little text are reported as potentially suspicious. A collection containing no extractable text cannot be indexed.

### Question answering

```text
Question entered in the terminal
        |
        v
Embed the question using the same local embedding model
        |
        v
Retrieve the top-k passages from the combined FAISS index
        |
        v
Send question + passages to Gemini with grounding/citation instructions
        |
        v
Validate citations against retrieved source filenames and pages
        |
        v
Print answer and source list, or abstain
```

The index is shared across PDFs, but every PDF is **not** sent to Gemini for every question. Only retrieved passages are included in the prompt. Retrieval is similarity-based; the returned similarity score is not a probability of correctness.

## Technology

| Component | Implementation |
| --- | --- |
| Language | Python 3.12 |
| PDF extraction | `pypdf` |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via LangChain Hugging Face integration |
| Vector store | Local FAISS via LangChain Community |
| Answer generation | Google Gemini via LangChain Google GenAI |
| Configuration | `.env` file loaded with `python-dotenv` |
| Tests | `pytest`; pipeline tests use synthetic data and mocks |

The MiniLM model produces 384-dimensional embeddings and runs locally after it has been downloaded. The first index build may need internet access to download the model. Gemini is a separate service used only for answer generation and requires a Gemini API key.

## Project structure

```text
app/
  cli.py                Interactive question-answering terminal
  config.py             Environment-backed settings and validation
  document_loader.py    PDF discovery, page text extraction, metadata
  chunker.py            Page-preserving chunking and chunk identifiers
  embeddings.py         Local embedding and Gemini model factories
  vectorstore.py        FAISS build, persistence, loading, retrieval
  prompts.py            Evidence-grounding and citation instructions
  rag.py                Retrieval, answer validation, abstention
scripts/
  build_index.py        Build the index from all configured PDFs
tests/                  Offline unit tests for pipeline components
  evaluation_questions.json
                        Templates for future real-document evaluation
documents/              Place local PDF source files here
vectorstore/            Generated FAISS index (not committed)
.env.example            Configuration template (no real credentials)
requirements.txt        Pinned Python dependencies
```

## Requirements

- Windows and Python 3.12 (the pinned CPU FAISS dependency is used with Python 3.12).
- A Gemini API key with access to the configured generation model, for answering questions.
- Internet access on first use to download the sentence-transformer model.
- Text-based PDFs. Scanned/image-only pages are not OCR-processed.

## Setup

Run these commands in PowerShell from the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your own key. Keep `.env` private; it is excluded from Git. Add the PDFs you are authorized to process directly to `documents/`.

## Configure

All settings can be placed in `.env`. Paths may be absolute or relative to the project directory.

| Variable | Default | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | unset | Required for answer generation; store the secret only in local `.env`. |
| `GEMINI_MODEL` | `gemini-3.7-flash` | Gemini model used to generate answers. Change it if this model is unavailable to your account. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local model used for both document chunks and questions. |
| `DOCUMENTS_DIR` | `documents` | Directory scanned for PDF files. |
| `FAISS_INDEX_PATH` | `vectorstore` | Directory used to save and load the combined index. |
| `CHUNK_SIZE` | `700` | Maximum chunk size in characters (not tokens). |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks in characters; must be smaller than `CHUNK_SIZE`. |
| `RETRIEVAL_K` | `6` | Maximum number of passages retrieved for a question. |

Use the same embedding model and chunk configuration for indexing and querying. Rebuild the entire index whenever a source PDF, `EMBEDDING_MODEL`, `CHUNK_SIZE`, or `CHUNK_OVERLAP` changes.

## Build or rebuild the index

```powershell
python -m scripts.build_index
```

The application processes PDFs in filename order, extracts text page by page, and stores the resulting vectors and metadata under `FAISS_INDEX_PATH`. The index contains extracted document text and must be treated as sensitive as the source PDFs. Rebuild it after changing the PDF collection or indexing configuration; the application does not incrementally synchronize changes.

The command fails with an actionable error if the configured directory is missing, contains no PDFs, cannot be read, or produces no extractable text. Scanned PDFs may require OCR before they can be indexed.

## Ask questions

After a successful index build:

```powershell
python -m app.cli
```

Enter questions at the `You:` prompt. Ask multiple questions in one session; enter `exit` or `quit` to leave. Example (illustrative only):

```text
Naive RAG — type 'exit' or 'quit' to quit

You: What do the reports say about the same topic?
Assistant: [Answer supported by retrieved excerpts.] [report-a.pdf, page 2]
Sources:
- report-a.pdf, page 2

You: exit
```

If retrieval returns no passages or Gemini does not cite a retrieved source using the expected `[filename.pdf, page N]` form, the application returns an insufficient-evidence response instead of presenting the answer as supported.

## Tests and evaluation

Run the offline automated suite:

```powershell
python -m pytest -q
```

The tests use synthetic documents plus deterministic or mocked embeddings and answer models. They cover configuration, PDF page ingestion and metadata, chunking, FAISS persistence/reload, retrieval across documents, citation validation, abstention, and the terminal loop. They do not make live Gemini API calls.

`tests/evaluation_questions.json` provides evaluation-case templates for single-PDF questions, cross-PDF synthesis, terminology differences, absent or ambiguous answers, tables/lists, multi-page sections, and conflicting sources. Fill in questions and expected filenames/pages using your own actual PDF collection before evaluating retrieval recall, answer accuracy, citation correctness, latency, or cost. No benchmark results or live-provider evaluation are claimed by this project.

## Limitations and data handling

- **No OCR:** scanned/image-only pages may have no extractable text and will be skipped.
- **Retrieval is not guaranteed:** a relevant passage may not be among the top `RETRIEVAL_K` results. Increase `RETRIEVAL_K` or adjust chunk settings only after evaluating against the actual documents.
- **Generation uses an external service:** the question and retrieved excerpts are sent to the configured Gemini service for answer generation. Do not index or query data unless you are permitted to process it this way.
- **Local index contains source-derived text:** keep `vectorstore/` private and rebuild it locally; generated index artifacts are excluded from Git.
- **Trusted index files only:** LangChain FAISS loading enables pickle deserialization for the local metadata file. Never load an index from an untrusted source.
- **No real-corpus evaluation is included:** automated tests validate pipeline behavior, not domain-specific answer quality.
- **No persistent conversation history:** questions and answers are handled in the current terminal session only.

## Troubleshooting

| Problem | Action |
| --- | --- |
| `GEMINI_API_KEY is not set` | Copy `.env.example` to `.env` and add a valid key. |
| FAISS index missing | Add PDFs to `documents/` and run `python -m scripts.build_index`. |
| No PDFs found | Check `DOCUMENTS_DIR`; the loader scans that directory (not subdirectories). |
| No extractable text | Verify the PDFs contain a text layer; OCR scanned files before indexing. |
| Changed embedding/chunk settings | Rebuild the full index, then start the CLI again. |
| Gemini model unavailable | Set `GEMINI_MODEL` to a generation model enabled for your account. |
