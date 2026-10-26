# RAG Pipeline Project

This project contains two implementations of a basic Retrieval-Augmented Generation (RAG) pipeline:

- `rag_pipeline.py`: manually wires document loading, chunking, embeddings, FAISS search, prompt construction, and OpenAI API calls.
- `rag_langchain.py`: implements the same basic flow using LangChain components.

## 1. Requirements

- Python 3.10+ recommended
- An OpenAI API key with access to the selected chat model
- Internet connection for initial package/model downloads and OpenAI API calls

## 2. Setup

Open a terminal in this project folder and create a virtual environment:

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Configure your API key

Copy `.env.example` to `.env` and put your own key in `.env`:

```dotenv
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Use a model name available to your API account. Keep `.env` private and never commit it to Git.

## 4. Add documents

Place text-based PDF files and/or UTF-8 `.txt` files inside `documents/`.

Scanned PDFs may need OCR first. The example loaders do not perform OCR.

## 5. Run the pure-Python implementation

```bash
python rag_pipeline.py
```

## 6. Run the LangChain implementation

```bash
python rag_langchain.py
```

Type a question when prompted. Type `exit` or `quit` to stop.

## Notes

- The embedding model `sentence-transformers/all-MiniLM-L6-v2` runs locally after it has been downloaded.
- The LLM generation step calls the OpenAI API and may incur API charges.
- Both versions rebuild the vector index at startup; they do not persist the index between runs.
- The pure-Python version uses word-based chunks (200 words, 40-word overlap).
- The LangChain version uses character-based chunks (1000 characters, 200-character overlap).
- Retrieval similarity is not a guarantee that the generated answer is correct.
