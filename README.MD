# SLCL Board Docs AI Assistant
## Description:
- Uses RAG to retrieve relevant parts of SLCL meeting minutes and uses AI to answer a query based on the documents.

## Quick Start:
- run `streamlit run rag_app.py` to open an interactive web UI (recommended for general use)

## Usage:
- run `python python_files/rag_system.py` to run the single query at the bottom of the file (best used for testing)
- run `python python_files/rebuild_index.py` to rebuild FAISS and BM25 indexes (only necessary if new documents are added)
- run `node scrapeDocs.js` to fetch more recent board docs. If new documents are fetched, rebuild indexes.

## How It Works:
### Rebuilding Indexes
run `python python_files/rebuild_index.py`
- removes ChromaDB index if exists
- calls `create_vectorstore` from python_files/rag_system.py

`create_vectorstore` does the following:
- Loads documents and extracts meeting dates as metadata
- Splits documents into chuncks
- Creates embeddings and saves in chroma_db
- Creates BM25 index

### Querying Documents
The main function to query documents is `query_documents` in python_files/rag_system.py.
The arguments are `question, verbose=False`.
If `verbose=True`, additional details will be printed to the console, including when certain parts of the function run and times each section takes to run.

#### Before `query_documents` is run:
- ChromaDB vectorstore is loaded
- llm instance is created

#### When `query_documents` is run:  
<b>1. Loading indexes and prefiltering</b>
- date_filter is extracted from the query if applicable
- bm25 index is loaded
- chroma_filter is added if applicable
  
<b>2. Performing search</b>
- semantic search is performed (ChromaDB)
- keyword search is performed (BM25)
- apply reciprocal rank function to combine scores from semantic and keyword search

<b>3. Filtering and reranking</b>
- apply date filtering to results
- rerank results with reranker

<b>4. Prompting LLM</b>
- load prompt
- request response

### Fetching Documents
To fetch documents, run node scrapeDocs.js
#### When `main` is run:
<b>1. Retrieve board meeting IDs</b>
- check previously processed meeting IDs.
- retrieve board meeting metadata from website
- process board meeting metadata

<b>2. Retrieve meeting documents for every ID that hasn't been previously processed</b>
- Download agenda (not currently used in RAG)
- Download minutes (used in RAG)
- Update retrieved IDs
