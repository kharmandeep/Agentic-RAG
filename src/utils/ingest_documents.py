# ingest_documents.py
import json
import weaviate
from datetime import datetime

def load_documents(filename='data/processed/weaviate_ready_docs.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Handle wrapper format
    if isinstance(data, dict) and 'documents' in data:
        return data['documents']
    return data

def ingest_documents():
    docs = load_documents()
    print(f"Loading {len(docs)} documents...")
    
    start_time = datetime.now()
    
    with weaviate.connect_to_local() as client:
        collection = client.collections.get("Documents")
        
        with collection.batch.rate_limit(requests_per_minute=600) as batch:
            for i, doc in enumerate(docs):
                # Map to your schema - NOW WITH IMAGES AND PDFS
                properties = {
                    'content': doc['content'],
                    'source': doc['source'],
                    'domain': doc['domain'],
                    'doc_type': doc['doc_type'],
                    'chunk_id': doc['chunk_id'],
                    'chunk_index': doc['chunk_index'],
                    'section_title': doc.get('section_title', ''),
                    'images': doc.get('images', []),  # ADD THIS
                    'pdfs': doc.get('pdfs', []),      # ADD THIS
                }
                
                batch.add_object(properties=properties)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(docs)}")
        
        failed_objects = collection.batch.failed_objects
        failed_refs = collection.batch.failed_references
        
        # DEBUG: Check if any objects with PDFs failed
        if failed_objects:
            print(f"\n🔍 DEBUG: Checking failed objects for PDFs...")
            for fail in failed_objects[:10]:
                print(f"  Failed: {fail}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\nIngestion complete in {duration:.2f}s")
        print(f"Failed objects: {len(failed_objects)}")
        print(f"Failed references: {len(failed_refs)}")
        
        total = collection.aggregate.over_all(total_count=True)
        print(f"Total in collection: {total.total_count}")
        
        if failed_objects:
            print(f"\nFirst few errors:")
            for obj in failed_objects[:5]:
                print(f"  {obj}")

if __name__ == "__main__":
    ingest_documents()