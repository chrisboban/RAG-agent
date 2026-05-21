# RAG Agent with Conversation Memory

A Retrieval-Augmented Generation (RAG) agent built with LangGraph, FAISS, and Groq. Supports continuous multi-turn conversation with a sliding memory window of the last 3 exchanges.

## How It Works

```
User Input
    |
    v
Router — keyword-based routing
    |
    |-- retrieve --> Retriever — FAISS semantic search + lexical reranking
    |                    |
    |                    v
    +-----------> Generator — answers from docs + conversation history
                       |
                       v
                  Bot Response
```

- **Retriever**: Two-stage hybrid retrieval — FAISS semantic search (top 20) reranked with keyword overlap scoring
- **Generator**: Uses last 6 messages (3 human + 3 AI) as memory. If no docs retrieved, answers from history alone
- **Ingest**: Playwright renders JS pages, BeautifulSoup extracts clean text, chunked and stored in a FAISS index

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/RAG-agent.git
cd RAG-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langchain langchain-community langchain-huggingface langchain-text-splitters
pip install langgraph
pip install faiss-cpu
pip install sentence-transformers
pip install groq
pip install playwright beautifulsoup4
pip install python-dotenv
playwright install chromium
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Add your source URLs

Open `ingest.py` and add the URLs you want the agent to answer questions about:

```python
urls = [
    "https://your-url-here.com/blog/article",
    ...
]
```

### 6. Build the FAISS index

Run this once (or whenever you change your source URLs):

```bash
python ingest.py
```

This scrapes the pages, chunks the text, embeds it, and saves the index to `faiss_index/`.

### 7. Start the agent

```bash
python main.py
```

---

## Project Structure

```
RAG-agent/
├── main.py              # Conversation loop
├── graph.py             # LangGraph pipeline definition
├── state.py             # Shared state schema
├── ingest.py            # Web scraper + FAISS index builder
└── nodes/
    ├── router.py        # Routes query to retrieval or direct answer
    ├── retriever.py     # Hybrid semantic + lexical retrieval
    └── generator.py     # LLM response generation with memory
```

---

## Configuration

| Variable | Location | Default | Description |
|---|---|---|---|
| `MEMORY_WINDOW` | `generator.py` | `6` | Number of past messages sent to LLM (3 pairs) |
| `k` | `retriever.py` | `20` | Number of semantic candidates from FAISS |
| `chunk_size` | `ingest.py` | `500` | Max characters per chunk |
| `chunk_overlap` | `ingest.py` | `100` | Overlap between consecutive chunks |

---

## Notes

- `faiss_index/` is not tracked in git — you must run `ingest.py` after cloning
- The embedding model (`all-MiniLM-L6-v2`) must be the same in both `ingest.py` and `retriever.py`
- Type `exit` or `quit` to end the chat session
