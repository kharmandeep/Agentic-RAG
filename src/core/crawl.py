from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
import time
from urllib.parse import urlparse

def crawling_with_filtering():
    """Enhanced crawling with better error handling and filtering."""
    
    sites_to_crawl = [
        {"url": "https://lucidmotors.com/knowledge", "max_depth": 2},
        {"url": "https://lucidmotors.com/air", "max_depth": 2},
        {"url": "https://www.wellsfargo.com/help/", "max_depth": 2}
    ]
    
    all_documents = []
    
    for site in sites_to_crawl:
        print(f"Crawling {site['url']} (max depth: {site['max_depth']})")
        
        try:
            loader = RecursiveUrlLoader(
                url=site['url'],
                max_depth=site['max_depth'],
                prevent_outside=True,
                use_async=True,
                # Enhanced filtering to avoid problematic URLs
                exclude_dirs=[
                    "admin", "login", "api", "search", "account",
                    "fonts", "assets", "static", "css", "js",
                    "images", "img", "media", "downloads",
                    "ar-", "en-", "fr-", "de-", "nl-", "zh-"  # Skip language variants
                ],
                timeout=15,  # Shorter timeout
                check_response_status=True,
                # Add custom headers to appear more like a regular browser
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                }
            )
            
            print("Starting crawl (errors are normal)...")
            docs = loader.load()
            
            print(f"Raw documents loaded: {len(docs)}")
            
            # Transform and filter
            html2text = Html2TextTransformer()
            docs_clean = html2text.transform_documents(docs)
            
            # Enhanced filtering
            valid_docs = []
            for doc in docs_clean:
                content = doc.page_content.strip()
                source = doc.metadata.get('source', '')
                
                # Skip if:
                if (len(content) < 100 or  # Too short
                    'error' in source.lower() or  # Error pages
                    any(lang in source for lang in ['/ar-', '/en-', '/fr-', '/de-', '/nl-']) or  # Language variants
                    content.lower().startswith('error') or  # Error content
                    'page not found' in content.lower()):  # 404 pages
                    continue
                
                valid_docs.append(doc)
            
            all_documents.extend(valid_docs)
            
            print(f"Valid documents: {len(valid_docs)}")
            print(f"Running total: {len(all_documents)} documents")
            
            # Show sample URLs that worked
            if valid_docs:
                print(f"   🎯 Sample successful URLs:")
                for doc in valid_docs[:3]:
                    url = doc.metadata.get('source', 'Unknown')
                    length = len(doc.page_content)
                    print(f"      • {url} ({length} chars)")
            
        except Exception as e:
            print(f"Critical error crawling {site['url']}: {e}")
            continue
        
        time.sleep(3)  # Be more respectful between sites
    
    return all_documents

# Execute improved crawling
documents = crawling_with_filtering()

print(f"CRAWLING COMPLETE!")
print(f"Total valid documents: {len(documents)}")

# Analysis of results
if documents:
    domains = {}
    total_chars = 0
    
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown')
        domain = urlparse(source).netloc if source != 'Unknown' else 'Unknown'
        domains[domain] = domains.get(domain, 0) + 1
        total_chars += len(doc.page_content)
    
    print(f"RESULTS BREAKDOWN:")
    for domain, count in domains.items():
        print(f"   {domain}: {count} pages")
    
    print(f"CONTENT STATS:")
    print(f"Total characters: {total_chars:,}")
    print(f"Average per document: {total_chars // len(documents):,} chars")
    
    # Show sample content
    print(f"SAMPLE CONTENT:")
    sample = documents[0]
    print(f"Source: {sample.metadata.get('source')}")
    print(f"Length: {len(sample.page_content)} characters")
    print(f"Preview: {sample.page_content[:400]}...")