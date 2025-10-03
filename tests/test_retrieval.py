# test_retrieval.py
import weaviate
from weaviate.classes.query import MetadataQuery, Filter

def safe_truncate(text, max_len):
    """Safely truncate text to max length"""
    if not text:
        return 'N/A'
    return text[:max_len] + '...' if len(text) > max_len else text


def test_basic_search():
    """Test basic vector search"""
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get('Documents')
        
        test_query = "renewable energy"
        
        print(f"Searching for: '{test_query}'")
        print("="*60)
        
        response = collection.query.near_text(
            query=test_query,
            limit=3,
            return_metadata=MetadataQuery(distance=True)
        )
        
        print(f"Found {len(response.objects)} results\n")
        
        for i, obj in enumerate(response.objects, 1):
            props = obj.properties
            meta = obj.metadata
            
            print(f"Result {i}:")
            print(f"  Type: {props.get('doc_type', 'N/A')}")
            print(f"  Domain: {props.get('domain', 'N/A')}")
            print(f"  Source: {safe_truncate(props.get('source'), 80)}")
            print(f"  Distance: {meta.distance:.4f}")
            print(f"  Content: {safe_truncate(props.get('content'), 200)}")
            print("-"*60)
        
        return True
        
    except Exception as e:
        print(f"Error in basic search: {e}")
        return False
    finally:
        if client is not None:
            client.close()


def test_filter_by_type():
    """Test filtering by document type"""
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get('Documents')
        
        print("\nTesting filters...")
        print("="*60)
        
        # Test HTML documents
        html_response = collection.query.near_text(
            query="energy efficiency",
            filters=Filter.by_property('doc_type').equal('html'),
            limit=2
        )
        html_count = len(html_response.objects)
        print(f"HTML documents found: {html_count}")
        
        # Test PDF documents
        pdf_response = collection.query.near_text(
            query="renewable energy",
            filters=Filter.by_property('doc_type').equal('pdf'),
            limit=2
        )
        pdf_count = len(pdf_response.objects)
        print(f"PDF documents found: {pdf_count}")
        
        return html_count > 0 and pdf_count > 0
        
    except Exception as e:
        print(f"Error in filter test: {e}")
        return False
    finally:
        if client is not None:
            client.close()


def test_hybrid_search():
    """Test hybrid search (vector + keyword)"""
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get('Documents')
        
        print("\nTesting hybrid search...")
        print("="*60)
        
        # Specify the target vector explicitly
        response = collection.query.hybrid(
            query="solar panels",
            alpha=0.75,
            target_vector="embedding",  # Add this line!
            limit=3
        )
        
        result_count = len(response.objects)
        print(f"Hybrid search found {result_count} results\n")
        
        for i, obj in enumerate(response.objects, 1):
            props = obj.properties
            domain = props.get('domain', 'N/A')
            doc_type = props.get('doc_type', 'N/A')
            print(f"Result {i}: {domain} ({doc_type})")
        
        return result_count > 0
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        return False
    finally:
        if client is not None:
            client.close()


def main():
    """Run all retrieval tests"""
    print("Testing Weaviate Retrieval\n")
    
    results = {
        'basic_search': test_basic_search(),
        'filter_test': test_filter_by_type(),
        'hybrid_search': test_hybrid_search()
    }
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)