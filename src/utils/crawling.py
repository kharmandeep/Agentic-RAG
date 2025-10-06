import os
import re
import requests
import tempfile
import time
import json
from typing import List, Dict
from urllib.parse import urljoin, urlparse

def extract_pdf_urls(documents) -> List[str]:
    """Extract PDF URLs from HTML documents."""
    
    pdf_urls = set()
    pdf_patterns = [
        r'https?://[^\s<>"]+\.pdf',
        r'href=["\']([^"\']*\.pdf[^"\']*)["\'"]',
    ]
    
    for doc in documents:
        content = doc.page_content if hasattr(doc, 'page_content') else doc['page_content']
        source_url = doc.metadata.get('source', '') if hasattr(doc, 'metadata') else doc['metadata'].get('source', '')
        
        for pattern in pdf_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                # Handle tuple from regex groups properly
                if isinstance(match, tuple):
                    pdf_url = match[0] if match[0] else ''
                else:
                    pdf_url = match
                
                # Skip empty URLs
                if not pdf_url:
                    continue
                
                if pdf_url.startswith('/'):
                    base_url = f"https://{urlparse(source_url).netloc}"
                    pdf_url = urljoin(base_url, pdf_url)
                elif not pdf_url.startswith('http'):
                    pdf_url = urljoin(source_url, pdf_url)
                
                pdf_url = pdf_url.split('"')[0].split("'")[0].split(')')[0]
                
                if pdf_url.lower().endswith('.pdf'):
                    pdf_urls.add(pdf_url)
    
    pdf_list = list(pdf_urls)
    print(f"Found {len(pdf_list)} PDF URLs")
    return pdf_list

def discover_pdfs(base_urls: List[str]) -> List[str]:
    """Check common paths for additional PDFs."""
    
    additional_pdfs = set()
    common_paths = [
        '/resources/', '/documents/', '/forms/', '/downloads/',
        '/investor-relations/', '/legal/', '/privacy/', '/terms/'
    ]
    
    for base_url in base_urls:
        domain = urlparse(base_url).netloc
        base_domain = f"https://{domain}"
        
        for path in common_paths:
            try:
                url = base_domain + path
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    pdf_links = re.findall(r'href=["\']([^"\']*\.pdf[^"\']*)["\'"]', 
                                         response.text, re.IGNORECASE)
                    
                    for link in pdf_links:
                        if link.startswith('/'):
                            full_url = base_domain + link
                        else:
                            full_url = urljoin(url, link)
                        additional_pdfs.add(full_url)
                        
            except:
                continue
    
    return list(additional_pdfs)

def process_pdfs(pdf_urls: List[str]) -> List[Dict]:
    """Download and process PDFs using LangChain loaders."""
    
    if not pdf_urls:
        return []
    
    try:
        from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
    except ImportError:
        print("PDF processing libraries not available")
        return []
    
    documents = []
    success_count = 0
    
    for i, url in enumerate(pdf_urls, 1):
        print(f"Processing PDF {i}/{len(pdf_urls)}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(response.content)
                temp_path = tmp.name
            
            try:
                try:
                    loader = UnstructuredPDFLoader(temp_path)
                    docs = loader.load()
                    loader_name = "UnstructuredPDFLoader"
                except ImportError:
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()
                    loader_name = "PyPDFLoader"
                
                for page_num, doc in enumerate(docs):
                    pdf_doc = {
                        'page_content': doc.page_content,
                        'metadata': {
                            'source': url,
                            'type': 'pdf',
                            'loader': loader_name,
                            'filename': url.split('/')[-1],
                            'page': page_num + 1,
                            **doc.metadata
                        }
                    }
                    documents.append(pdf_doc)
                
                success_count += 1
                print(f"  Extracted {len(docs)} pages")
                
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print(f"Successfully processed {success_count}/{len(pdf_urls)} PDFs")
    return documents

def combine_documents(html_docs, pdf_docs):
    """Combine HTML and PDF documents with consistent format."""
    
    combined = []
    
    # Normalize HTML documents
    for doc in html_docs:
        if hasattr(doc, 'page_content'):
            doc_dict = {
                'page_content': doc.page_content,
                'metadata': dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            }
        else:
            doc_dict = doc
        
        doc_dict['metadata']['type'] = 'html'
        combined.append(doc_dict)
    
    # Add PDF documents
    combined.extend(pdf_docs)
    
    html_count = sum(1 for d in combined if d['metadata'].get('type') == 'html')
    pdf_count = sum(1 for d in combined if d['metadata'].get('type') == 'pdf')
    
    print(f"Combined {html_count} HTML and {pdf_count} PDF documents")
    return combined

def scrape_sites():
    """Scrape HTML content from target sites."""
    
    try:
        from langchain_community.document_loaders import RecursiveUrlLoader
        from langchain_community.document_transformers import Html2TextTransformer
    except ImportError:
        print("LangChain libraries not available")
        return []
    
    sites = [
        {"url": "https://lucidmotors.com/knowledge", "depth": 2},
        {"url": "https://lucidmotors.com/air", "depth": 2},
        {"url": "https://www.wellsfargo.com/help/", "depth": 2}
    ]
    
    all_docs = []
    
    for site in sites:
        print(f"Crawling {site['url']} (depth: {site['depth']})")
        
        try:
            loader = RecursiveUrlLoader(
                url=site['url'],
                max_depth=site['depth'],
                prevent_outside=True,
                use_async=True,
                exclude_dirs=[
                    "admin", "login", "api", "search", "fonts", "assets",
                    "css", "js", "images", "downloads", "ar-", "en-", "fr-"
                ],
                timeout=15,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            docs = loader.load()
            print(f"  Loaded {len(docs)} raw documents")
            
            transformer = Html2TextTransformer()
            clean_docs = transformer.transform_documents(docs)
            
            # Filter valid documents
            valid_docs = []
            for doc in clean_docs:
                content = doc.page_content.strip()
                source = doc.metadata.get('source', '')
                
                if (len(content) >= 100 and 
                    'error' not in source.lower() and
                    not content.lower().startswith('error') and
                    'page not found' not in content.lower()):
                    valid_docs.append(doc)
            
            all_docs.extend(valid_docs)
            print(f"  Kept {len(valid_docs)} valid documents")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        time.sleep(2)
    
    return all_docs

def integrate_pdfs(html_docs):
    """Complete PDF integration pipeline."""
    
    # Extract PDF URLs from scraped HTML
    pdf_urls = extract_pdf_urls(html_docs)
    
    # Discover additional PDFs
    base_urls = [
        "https://lucidmotors.com/knowledge",
        "https://lucidmotors.com/air",
        "https://www.wellsfargo.com/help/"
    ]
    additional_pdfs = discover_pdfs(base_urls)
    
    # Combine and deduplicate
    all_pdf_urls = list(set(pdf_urls + additional_pdfs))
    
    print(f"PDF discovery: {len(pdf_urls)} from HTML, {len(additional_pdfs)} additional")
    print(f"Total unique PDFs: {len(all_pdf_urls)}")
    
    # Process PDFs
    pdf_docs = process_pdfs(all_pdf_urls) if all_pdf_urls else []
    
    # Combine everything
    return combine_documents(html_docs, pdf_docs)

def save_documents(docs, filename='scraped_documents.json'):
    """Save documents to JSON file."""
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)
        
        size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"Saved {len(docs)} documents to {filename} ({size_mb:.1f}MB)")
        
    except Exception as e:
        print(f"Save failed: {e}")

def main():
    """Main scraping pipeline."""
    
    print("Starting document scraping pipeline")
    
    # Scrape HTML content
    html_docs = scrape_sites()
    if not html_docs:
        print("No HTML documents scraped")
        return None
    
    print(f"HTML scraping complete: {len(html_docs)} documents")
    
    # Integrate PDFs
    all_docs = integrate_pdfs(html_docs)
    
    # Statistics
    html_count = sum(1 for d in all_docs if d['metadata'].get('type') == 'html')
    pdf_count = sum(1 for d in all_docs if d['metadata'].get('type') == 'pdf')
    total_chars = sum(len(d['page_content']) for d in all_docs)
    
    print(f"\nPipeline complete:")
    print(f"  Total documents: {len(all_docs)}")
    print(f"  HTML: {html_count}, PDF: {pdf_count}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg length: {total_chars // len(all_docs):,} chars")
    
    # Save results
    save_documents(all_docs)
    
    return all_docs

if __name__ == "__main__":
    documents = main()