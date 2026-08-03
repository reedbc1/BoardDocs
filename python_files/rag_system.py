import json
import os
import pickle
import re
import time
from datetime import datetime

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

# Configuration
MINUTES_DIR = "minutes"
INDEX_PATH = "chroma_db"
COLLECTION_NAME = "board_minutes"
BM25_INDEX_PATH = "bm25_index.pkl"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2")

# Global BM25 retriever
bm25_retriever = None
bm25_documents = None

def extract_date_from_document(doc):
    """Extract date from document metadata/content and add to metadata."""
    try:
        # Try to parse JSON metadata at the beginning of the document
        content = doc.page_content
        # Match multi-line JSON object at the start (using .*? for non-greedy match)
        json_match = re.search(r'^\{.*?\}', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            metadata = json.loads(json_str)
            if 'date' in metadata:
                date_str = metadata['date']
                # Parse ISO format date (e.g., "2025-11-17T00:00:00Z")
                # Extract just the date part before 'T'
                date_only = date_str.split('T')[0]
                doc.metadata['date'] = date_only
                print(f"Extracted date {date_only} from {doc.metadata.get('source', 'unknown')}")
                return doc
        
        # Fallback: try to extract from filename if available
        source = doc.metadata.get('source', '')
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', source)
        if date_match:
            doc.metadata['date'] = date_match.group(1)
            print(f"Extracted date {date_match.group(1)} from filename {source}")
            return doc
            
    except Exception as e:
        print(f"Warning: Could not extract date from document {doc.metadata.get('source', 'unknown')}: {e}")
    
    return doc

def create_bm25_index(chunks):
    """Create BM25 index from document chunks."""
    print("Creating BM25 index...")
    
    # Tokenize documents for BM25
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index and documents
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'documents': chunks}, f)
    
    print(f"BM25 index saved to {BM25_INDEX_PATH}")
    return bm25, chunks

def load_bm25_index():
    """Load existing BM25 index."""
    with open(BM25_INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['bm25'], data['documents']

def create_vectorstore():
    """Load documents, split into chunks, and create ChromaDB vector store and BM25 index."""
    print("Loading documents...")
    loader = DirectoryLoader(MINUTES_DIR, glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")
    
    # Extract and add date metadata to documents
    print("Extracting date metadata...")
    documents = [extract_date_from_document(doc) for doc in documents]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    
    # Create ChromaDB client and collection
    client = chromadb.PersistentClient(path=INDEX_PATH)
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=INDEX_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print(f"Vector store saved to {INDEX_PATH}")
    
    # Create BM25 index
    create_bm25_index(chunks)

    return vectorstore

def load_vectorstore():
    """Load existing ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = Chroma(
        persist_directory=INDEX_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    return vectorstore

def create_llm():
    """Create OpenAI ChatGPT LLM."""
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.2
    )
    return llm

# Load expensive resources only when a query actually needs them. In particular,
# rebuild_index imports this module before deleting INDEX_PATH, so opening Chroma
# here would make that process lock the SQLite file it is about to remove.
vectorstore = None
llm = None

def get_vectorstore():
    global vectorstore
    if vectorstore is None:
        if os.path.exists(INDEX_PATH):
            print("Loading existing vector store...")
            vectorstore = load_vectorstore()
        else:
            print("Creating new vector store...")
            vectorstore = create_vectorstore()
    return vectorstore

def get_llm():
    global llm
    if llm is None:
        llm = create_llm()
    return llm

def reciprocal_rank_fusion(results_list, k=60):
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.
    
    Args:
        results_list: List of lists containing (document, score) tuples
        k: Constant for RRF formula (default 60)
    
    Returns:
        List of documents sorted by fused score
    """
    fused_scores = {}
    
    for results in results_list:
        for rank, (doc, score) in enumerate(results, start=1):
            # Use document content as key to identify unique documents
            doc_key = doc.page_content
            if doc_key not in fused_scores:
                fused_scores[doc_key] = {'doc': doc, 'score': 0}
            # RRF formula: 1 / (k + rank)
            fused_scores[doc_key]['score'] += 1 / (k + rank)
    
    # Sort by fused score
    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )
    
    return [item['doc'] for item in sorted_results]

def extract_date_filter_from_query(question):
    """Use LLM to extract date or date range from the user's question."""
    llm = get_llm()
    
    prompt = f"""Analyze the following question and extract any date or date range mentioned.

Question: {question}

If the question mentions:
- A specific date, return it in YYYY-MM-DD format
- A date range, return both start and end dates in YYYY-MM-DD format
- A month/year, return the start and end of that month
- A year, return the start and end of that year
- Relative dates ("last month", "this year"), convert to actual dates (today is {datetime.now().strftime('%Y-%m-%d')})
- "Recent" or "recently", return the date 4 months ago and today
- "Most recent", return the date 2 months ago and today.
- No date mentioned, return "NONE"
- "In a particular month" return the first date of that month and the last date of that month.
- If the user does not mention a year and only mentions a month, assume that the year is this year 
  (this year is {datetime.now().strftime('%Y')}), unless the month has not occured yet.
- If the month has not occured yet this year (ex. December, 2026-12 and todays date is 2025-01), assume the year is last year 
  (ex. 2025-12)

Respond ONLY with a JSON object in one of these formats:
1. For a specific date: {{"type": "single", "date": "YYYY-MM-DD"}}
2. For a date range: {{"type": "range", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}}
3. For no date: {{"type": "none"}}

Examples:
- "What happened in December 2025?" → {{"type": "range", "start_date": "2025-12-01", "end_date": "2025-12-31"}}
- "Summarize the meeting on December 15, 2025" → {{"type": "single", "date": "2025-12-15"}}
- "What decisions were made?" → {{"type": "none"}}
- "Tell me about 2024" → {{"type": "range", "start_date": "2024-01-01", "end_date": "2024-12-31"}}

"""
    
    try:
        response = llm.invoke(prompt)
        result = json.loads(response.content.strip())
        return result
    except Exception as e:
        print(f"Error extracting date filter: {e}")
        return {"type": "none"}

def query_documents(question, verbose=False):
    """Query the RAG system with manual reranking and date filtering.
    
    Args:
        question: The user's question
        verbose: If True, print debug information
    
    Returns:
        tuple: (answer_text, list_of_source_documents)
    """
    
    # Extract date filter from question
    if verbose:
        print("Extracting date filter from query...")
    t0 = time.perf_counter()
    date_filter = extract_date_filter_from_query(question)
    t1 = time.perf_counter()
    if verbose:
        print(f"Date filter: {date_filter}")
        print(f"  ⏱️  Date extraction time: {t1-t0:.4f}s")

    # Load BM25 retriever if not already loaded
    global bm25_retriever, bm25_documents
    if bm25_retriever is None:
        if verbose:
            print("Loading BM25 index...")
        t0 = time.perf_counter()
        bm25_retriever, bm25_documents = load_bm25_index()
        t1 = time.perf_counter()
        if verbose:
            print(f"  ⏱️  BM25 loading time: {t1-t0:.4f}s")
    
    # Build ChromaDB filter based on date requirements
    # ChromaDB uses exact string matching for equality, so we can use date strings directly
    chroma_filter = None
    if date_filter['type'] == 'single':
        chroma_filter = {"date": date_filter['date']}
    elif date_filter['type'] == 'range':
        # For range queries, we need to filter manually after retrieval
        # because ChromaDB doesn't support string comparison operators
        pass
    
    if verbose:
        print(f"ChromaDB filter: {chroma_filter}")
    
    # HYBRID SEARCH: Combine BM25 and semantic search
    
    # 1. Semantic search with ChromaDB (no filter - ChromaDB has bug with query+filter)
    active_vectorstore = get_vectorstore()
    t0 = time.perf_counter()
    try:
        semantic_docs = active_vectorstore.similarity_search_with_score(question, k=100)
        t1 = time.perf_counter()
        
        if verbose:
            print(f"Semantic search: Retrieved {len(semantic_docs)} documents")
            print(f"  ⏱️  Semantic search time: {t1-t0:.4f}s")
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        semantic_docs = []
        t1 = time.perf_counter()
    
    # 2. BM25 keyword search
    t0 = time.perf_counter()
    tokenized_query = question.lower().split()
    bm25_scores = bm25_retriever.get_scores(tokenized_query)
    
    # Get top BM25 results
    bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:100]
    bm25_docs = [(bm25_documents[i], bm25_scores[i]) for i in bm25_top_indices]
    t1 = time.perf_counter()
    
    if verbose:
        print(f"BM25 search: Retrieved {len(bm25_docs)} documents")
        print(f"  ⏱️  BM25 search time: {t1-t0:.4f}s")
    
    # 3. Combine results using Reciprocal Rank Fusion
    t0 = time.perf_counter()
    candidate_docs = reciprocal_rank_fusion([semantic_docs, bm25_docs])
    t1 = time.perf_counter()
    
    if verbose:
        print(f"Hybrid search: Combined to {len(candidate_docs)} unique documents")
        print(f"  ⏱️  RRF fusion time: {t1-t0:.4f}s")
    
    # Apply date filtering manually after hybrid search
    # (ChromaDB has a bug with query+filter, so we filter post-retrieval)
    if date_filter['type'] in ['single', 'range'] and len(candidate_docs) > 0:
        if verbose:
            print("Filtering documents by date...")
        t0 = time.perf_counter()
        filtered_docs = []
        
        for doc in candidate_docs:
            doc_date_str = doc.metadata.get('date')
            if not doc_date_str:
                continue
                
            try:
                if date_filter['type'] == 'single':
                    if doc_date_str == date_filter['date']:
                        filtered_docs.append(doc)
                elif date_filter['type'] == 'range':
                    # Simple string comparison works for YYYY-MM-DD format
                    if date_filter['start_date'] <= doc_date_str <= date_filter['end_date']:
                        filtered_docs.append(doc)
            except Exception as e:
                print(f"Error filtering date for document: {e}")
                continue
        
        candidate_docs = filtered_docs
        t1 = time.perf_counter()
        if verbose:
            print(f"After date filtering: {len(candidate_docs)} documents")
            print(f"  ⏱️  Date filtering time: {t1-t0:.4f}s")
    
    # If no documents match the date filter, inform the user
    if len(candidate_docs) == 0:
        if date_filter['type'] != 'none':
            date_info = date_filter.get('date') or f"{date_filter.get('start_date')} to {date_filter.get('end_date')}"
            return f"No documents found for the specified date: {date_info}. The available document dates range from 2012 to 2025.", []
        return "No relevant documents found for your query.", []

    # Rerank
    t0 = time.perf_counter()
    pairs = [(question, doc.page_content) for doc in candidate_docs]
    scores = reranker.predict(pairs)

    reranked_docs = sorted(
        zip(candidate_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    docs = [doc for doc, score in reranked_docs[:20]]
    t1 = time.perf_counter()
    
    if verbose:
        print(f"  ⏱️  Reranking time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    context = "\n\n".join(
        [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
    )

    prompt = f"""You are an AI assistant helping analyze board meeting minutes. Use the following context from board meeting documents to answer the question accurately and comprehensively.


Context from board meetings:
{context}

Question: {question}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information to fully answer the question, say so and provide what information is available.
"""
    t1 = time.perf_counter()
    if verbose:
        print(f"  ⏱️  Context preparation time: {t1-t0:.4f}s")

    t0 = time.perf_counter()
    llm = get_llm()
    response = llm.invoke(prompt)
    t1 = time.perf_counter()
    if verbose:
        print(f"  ⏱️  LLM generation time: {t1-t0:.4f}s")
    
    return response.content, docs

if __name__ == "__main__":
    # Test query
    question = "Summarize the board docs for December."
    start_time = time.perf_counter()
    response, docs = query_documents(question, verbose=True)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\nElapsed time: {elapsed_time:.4f} seconds")
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response}")
    print(f"\nSources: {[doc.metadata['source'] for doc in docs]}")
