# chunking.py
import json
import os
from typing import List, Dict
from urllib.parse import urlparse

def load_documents(filename='AgenticRag/data/processed/scraped_documents.json'):
    """Load documents from JSON file"""
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
    
    print(f"Loading: {filename}")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Get documents list
    if isinstance(data, dict) and 'documents' in data:
        docs = data['documents']
    elif isinstance(data, list):
        docs = data
    else:
        print(f"Unexpected format: {type(data)}")
        return None
    
    print(f"Loaded {len(docs)} documents")
    return docs


def chunk_text(text, size=1000, overlap=200):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        chunk = ' '.join(words[i:i + size])
        if len(chunk) > 50:
            chunks.append(chunk)
        i += size - overlap
    
    return chunks


def chunk_documents(documents, chunk_size=1000, overlap=200):
    """Chunk all documents"""
    
    print(f"Chunking {len(documents)} documents...")
    
    chunks = []
    
    for i, doc in enumerate(documents):
        content = doc['page_content']
        meta = doc['metadata']
        
        if len(content) < 50:
            continue
        
        # Split into chunks
        text_chunks = chunk_text(content, chunk_size, overlap)
        
        # Create chunk objects
        for j, text in enumerate(text_chunks):
            chunks.append({
                'page_content': text,
                'metadata': {
                    **meta,
                    'chunk_id': f"doc_{i}_chunk_{j}",
                    'chunk_index': j,
                    'total_chunks': len(text_chunks)
                }
            })
    
    print(f"Created {len(chunks)} chunks")
    return chunks


def save_chunks(chunks, filename='AgenticRag/data/processed/chunked_documents.json'):
    """Save chunks to file"""
    
    # Create directory
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save just the array of chunks (no wrapper)
    with open(filename, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    size = os.path.getsize(filename) / (1024 * 1024)
    print(f"Saved {len(chunks)} chunks to {filename} ({size:.1f}MB)")


def main():
    """Run chunking pipeline"""
    
    print("\nDocument Chunking")
    print("="*60)
    
    # Load
    docs = load_documents()
    if not docs:
        return
    
    # Chunk
    chunks = chunk_documents(docs, chunk_size=1000, overlap=200)
    
    # Save
    save_chunks(chunks)
    
    print("Done!")
    return chunks


if __name__ == "__main__":
    main()