# chunking.py
import json
import os
from typing import List, Dict
from urllib.parse import urlparse

def load_documents(filename='scraped_documents.json'):
    """Load documents from JSON file."""
    
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return None
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        html_count = sum(1 for d in docs if d['metadata'].get('type') == 'html')
        pdf_count = sum(1 for d in docs if d['metadata'].get('type') == 'pdf')
        
        print(f"Loaded {len(docs)} documents ({html_count} HTML, {pdf_count} PDF)")
        return docs
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        return None

def chunk_documents(documents, chunk_size=1000, overlap=200):
    """Split documents into chunks for vector storage."""
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        print("langchain-text-splitters not installed. Run: pip install langchain-text-splitters")
        return None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    
    for i, doc in enumerate(documents):
        content = doc['page_content']
        metadata = doc['metadata']
        
        if len(content.strip()) < 50:
            continue
        
        # Split content
        texts = splitter.split_text(content)
        
        # Create chunk objects
        for j, text in enumerate(texts):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                'chunk_id': f"doc_{i}_chunk_{j}",
                'chunk_index': j,
                'total_chunks': len(texts),
                'source_doc_index': i
            })
            
            chunks.append({
                'page_content': text,
                'metadata': chunk_meta
            })
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def analyze_chunks(chunks):
    """Analyze chunk statistics."""
    
    if not chunks:
        return {}
    
    sizes = [len(c['page_content']) for c in chunks]
    
    # Basic stats
    stats = {
        'count': len(chunks),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'avg_size': int(sum(sizes) / len(sizes)),
        'total_chars': sum(sizes)
    }
    
    # Size distribution
    size_buckets = {
        'small': sum(1 for s in sizes if s < 300),
        'medium': sum(1 for s in sizes if 300 <= s < 800),
        'large': sum(1 for s in sizes if 800 <= s < 1200),
        'xlarge': sum(1 for s in sizes if s >= 1200)
    }
    
    # Type breakdown
    type_counts = {}
    domain_counts = {}
    
    for chunk in chunks:
        # Document type
        doc_type = chunk['metadata'].get('type', 'unknown')
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        # Source domain
        source = chunk['metadata'].get('source', '')
        if source:
            domain = urlparse(source).netloc
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"Chunk Analysis:")
    print(f"  Total: {stats['count']} chunks")
    print(f"  Size range: {stats['min_size']}-{stats['max_size']} chars (avg: {stats['avg_size']})")
    print(f"  Size distribution: small:{size_buckets['small']}, medium:{size_buckets['medium']}, large:{size_buckets['large']}, xl:{size_buckets['xlarge']}")
    print(f"  By type: {dict(type_counts)}")
    print(f"  By domain: {dict(domain_counts)}")
    
    return stats

def save_chunks(chunks, filename='chunked_documents.json'):
    """Save chunks to JSON."""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"Saved {len(chunks)} chunks to {filename} ({size_mb:.1f}MB)")
        
    except Exception as e:
        print(f"Error saving chunks: {e}")

def preview_chunks(chunks, count=3):
    """Show sample chunks."""
    
    print(f"\nSample chunks:")
    for i, chunk in enumerate(chunks[:count]):
        meta = chunk['metadata']
        content = chunk['page_content']
        
        print(f"\nChunk {i+1}:")
        print(f"  ID: {meta.get('chunk_id')}")
        print(f"  Type: {meta.get('type', 'unknown')}")
        print(f"  Source: {meta.get('source', 'unknown')[:60]}...")
        print(f"  Size: {len(content)} chars")
        print(f"  Content: {content[:150]}...")

def main():
    """Main chunking pipeline."""
    
    print("Starting document chunking...")
    
    # Load documents
    docs = load_documents()
    if not docs:
        return None
    
    # Chunk documents
    chunks = chunk_documents(docs, chunk_size=1000, overlap=200)
    if not chunks:
        return None
    
    # Analyze results
    analyze_chunks(chunks)
    
    # Preview some chunks
    preview_chunks(chunks)
    
    # Save results
    save_chunks(chunks)
    
    print(f"\nChunking complete. {len(chunks)} chunks ready for vector store.")
    return chunks

if __name__ == "__main__":
    chunks = main()