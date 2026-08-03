"""
Rebuild the ChromaDB index with date metadata.
Run this script to rebuild your vector store with date information.
"""

import os
import shutil
import sys
import time
from rag_system import create_vectorstore, INDEX_PATH, BM25_INDEX_PATH

def rebuild_index():
    """Delete existing index and create a new one with date metadata."""
    
    # Remove BM25 index if exists
    if os.path.exists(BM25_INDEX_PATH):
        print(f"Removing existing BM25 index at {BM25_INDEX_PATH}...")
        os.remove(BM25_INDEX_PATH)
        print("BM25 index removed.")
    
    # Try to remove ChromaDB directory
    if os.path.exists(INDEX_PATH):
        print(f"Removing existing ChromaDB index at {INDEX_PATH}...")
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(INDEX_PATH)
                print("ChromaDB index removed.")
                break
            except PermissionError as e:
                if attempt < max_attempts - 1:
                    print(
                        "Could not remove ChromaDB "
                        f"({e!r}); waiting 2 seconds... "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
                    time.sleep(2)
                else:
                    print(f"\\nError: Cannot delete {INDEX_PATH}: {e!r}")
                    print(f"Failing path: {e.filename or 'unknown'}")
                    print(f"Windows error code: {getattr(e, 'winerror', 'unknown')}")
                    print("Another process may have the file open, or Windows may be denying deletion for another reason.")
                    sys.exit(1)
    
    print("\\nCreating new vector store with date metadata and BM25 index...")
    vectorstore = create_vectorstore()
    print("\\nIndex rebuilt successfully!")
    print(f"ChromaDB vector store saved to {INDEX_PATH}")
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

if __name__ == "__main__":
    rebuild_index()
