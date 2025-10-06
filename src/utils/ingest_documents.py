# ingest_documents.py - Optimized batch version
import json
import weaviate
from datetime import datetime

def load_documents(filename='data/processed/weaviate_ready_docs.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def ingest_documents():
    docs = load_documents()
    print(f"Loading {len(docs)} documents...")
    
    start_time = datetime.now()
    
    with weaviate.connect_to_local() as client:
        collection = client.collections.get("Documents")
        
        # Use rate_limit batch for better control
        with collection.batch.rate_limit(requests_per_minute=600) as batch:
            for i, doc in enumerate(docs):
                batch.add_object(properties=doc)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(docs)}")
        
        # Check results after batch completes
        failed_objects = collection.batch.failed_objects
        failed_refs = collection.batch.failed_references
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\nIngestion complete in {duration:.2f}s")
        print(f"Failed objects: {len(failed_objects)}")
        print(f"Failed references: {len(failed_refs)}")
        
        # Verify count
        total = collection.aggregate.over_all(total_count=True)
        print(f"Total in collection: {total.total_count}")
        
        # Show any errors
        if failed_objects:
            print(f"\nFirst few errors:")
            for obj in failed_objects[:5]:
                print(f"  {obj}")

if __name__ == "__main__":
    ingest_documents()