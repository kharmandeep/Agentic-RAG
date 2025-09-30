# test_data_quality.py
import weaviate

def check_data_quality():
    """Check the quality of ingested data"""
    client = None
    try:
        client = weaviate.connect_to_local()
        collection = client.collections.get('Documents')
        
        print("Data Quality Check")
        print("="*60)
        
        # Get total count
        result = collection.aggregate.over_all(total_count=True)
        total_count = result.total_count
        print(f"\nTotal documents: {total_count}")
        
        # Sample documents
        response = collection.query.fetch_objects(limit=20)
        
        print(f"\nSample Analysis ({len(response.objects)} documents):")
        
        content_lengths = []
        doc_types = {}
        domains = set()
        has_chunk_id = 0
        has_section_title = 0
        
        for obj in response.objects:
            props = obj.properties
            
            # Safe extraction with type conversion
            content = str(props.get('content', ''))
            content_lengths.append(len(content))
            
            doc_type = str(props.get('doc_type', 'unknown'))
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            domain = str(props.get('domain', 'unknown'))
            domains.add(domain)
            
            if props.get('chunk_id'):
                has_chunk_id += 1
            if props.get('section_title'):
                has_section_title += 1
        
        # Display stats
        if content_lengths:
            print(f"\nContent length stats:")
            print(f"  Min: {min(content_lengths)} chars")
            print(f"  Max: {max(content_lengths)} chars")
            print(f"  Avg: {sum(content_lengths)//len(content_lengths)} chars")
        
        print(f"\nDocument types in sample:")
        for dtype, count in doc_types.items():
            print(f"  {dtype}: {count}")
        
        print(f"\nUnique domains in sample: {len(domains)}")
        for domain in sorted(list(domains))[:5]:
            print(f"  - {domain}")
        
        print(f"\nField completeness:")
        print(f"  Has chunk_id: {has_chunk_id}/{len(response.objects)}")
        print(f"  Has section_title: {has_section_title}/{len(response.objects)}")
        
        # Check for issues
        print(f"\nChecking for issues...")
        
        if content_lengths:
            short_content = [l for l in content_lengths if l < 50]
            if short_content:
                print(f"  ⚠️  {len(short_content)} documents with <50 chars")
            else:
                print(f"  ✅ All sample documents have adequate content")
        
        # Check required fields
        missing_fields = 0
        for obj in response.objects:
            props = obj.properties
            content = props.get('content')
            source = props.get('source')
            domain = props.get('domain')
            chunk_id = props.get('chunk_id')
            
            if not all([content, source, domain, chunk_id]):
                missing_fields += 1
        
        if missing_fields:
            print(f"  ⚠️  {missing_fields} documents with missing required fields")
        else:
            print(f"  ✅ All sample documents have required fields")
        
        print("\n" + "="*60)
        print("Data quality check completed!")
        return True
        
    except Exception as e:
        print(f"Error in data quality check: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    success = check_data_quality()
    exit(0 if success else 1)