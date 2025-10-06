import json
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Dict
from pathlib import Path

def load_chunked_documents(filename='AgenticRag/data/processed/chunked_documents.json') -> List[Dict]:
    """Load chunked documents from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {filename}")
        return chunks
    except FileNotFoundError:
        print(f"File {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading chunks: {e}")
        return []

def map_chunk_to_schema(chunk: Dict) -> Dict:
    """Map chunked document to Weaviate schema."""
    metadata = chunk.get('metadata', {})
    content = chunk.get('page_content', '')
    source = metadata.get('source', '')
    doc_type = metadata.get('type', 'unknown')

    domain = 'unknown'
    
    # Smart domain extraction
    if source.startswith('http://') or source.startswith('https://'):
        domain = urlparse(source).netloc or 'unknown'
    elif doc_type == 'pdf':
        # For PDFs, use actual file_name if available
        file_name = metadata.get('file_name', '')
        if file_name:
            # Use the PDF filename without extension (e.g., "84327" from "84327.pdf")
            domain = Path(file_name).stem
    else:
        domain = 'local_file'
    
    return {
        'content': content,
        'source': source,
        'domain': domain,
        'doc_type': doc_type,
        'chunk_id': metadata.get('chunk_id', f"chunk_{hash(content)}"),
        'chunk_index': metadata.get('chunk_index', 0),
        'section_title': metadata.get('title', ''),
        'url': source if source.startswith('http') else '',
        'created_at': metadata.get('scraped_at', ''),
    }

def validate_document(doc: Dict) -> bool:
    """Validate document has required fields."""
    required_fields = ['content', 'source', 'domain', 'doc_type', 'chunk_id']
    
    for field in required_fields:
        if field not in doc or not doc[field]:
            #print(f"Warning: Missing or empty field '{field}' in document")
            return False
    
    # Check content length
    if len(doc['content'].strip()) < 10:
        #print(f"Warning: Content too short in chunk {doc['chunk_id']}")
        return False
    
    return True

def prepare_documents_for_weaviate(filename='AgenticRag/data/processed/chunked_documents.json'):
    """Transform all chunks for Weaviate ingestion."""
    chunks = load_chunked_documents(filename)
    
    if not chunks:
        return []
    
    weaviate_docs = []
    valid_count = 0
    failed_count = 0  # Add counter
    
    for chunk in chunks:
        try:
            weaviate_doc = map_chunk_to_schema(chunk)
            
            if validate_document(weaviate_doc):
                weaviate_docs.append(weaviate_doc)
                valid_count += 1
            else:
                failed_count += 1  # Count failures
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            failed_count += 1
            continue
    
    print(f"Prepared {valid_count} valid documents for Weaviate")
    print(f"Failed validation: {failed_count}")  # Show failures
    print(f"Total processed: {valid_count + failed_count}")  # Should equal 1756
    return weaviate_docs


def save_prepared_documents(docs: List[Dict], filename='data/processed/weaviate_ready_docs.json'):
    """Save prepared documents for inspection."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(docs)} prepared documents to {filename}")
    except Exception as e:
        print(f"Error saving prepared documents: {e}")

def preview_documents(docs: List[Dict], count: int = 3):
    """Preview sample prepared documents."""
    print(f"\nPreview of {min(count, len(docs))} documents:")
    print("=" * 60)
    
    for i, doc in enumerate(docs[:count]):
        print(f"\nDocument {i+1}:")
        print(f"  Domain: {doc['domain']}")
        print(f"  Type: {doc['doc_type']}")
        print(f"  Chunk ID: {doc['chunk_id']}")
        print(f"  Content length: {len(doc['content'])} chars")
        print(f"  Source: {doc['source'][:60]}...")
        print(f"  Content preview: {doc['content'][:150]}...")
        print("-" * 60)

def main():
    """Main function to prepare documents."""
    print("Starting data preparation for Weaviate...")
    
    # Prepare documents
    docs = prepare_documents_for_weaviate()
    
    if not docs:
        print("No documents prepared. Check your input file.")
        return
    
    # Preview results
    preview_documents(docs)
    
    # Save prepared documents
    save_prepared_documents(docs)
    
    print(f"\nData preparation complete! {len(docs)} documents ready for ingestion.")
    return docs

if __name__ == "__main__":
    prepared_docs = main()