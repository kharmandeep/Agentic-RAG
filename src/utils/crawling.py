"""
Simple but Complete Web Crawler for Agentic RAG
Clean code with all necessary logic
"""

import os
import re
import requests
import tempfile
import time
import json
from typing import List, Dict
from urllib.parse import urljoin, urlparse
from datetime import datetime


def scrape_websites():
    """Scrape HTML from target websites"""
    
    try:
        from langchain_community.document_loaders import RecursiveUrlLoader
        from langchain_community.document_transformers import Html2TextTransformer
    except ImportError:
        print("ERROR: Install required packages:")
        print("pip install langchain-community beautifulsoup4 lxml")
        return []
    
    # Define sites to crawl
    sites = [
        {
            "url": "https://lucidmotors.com/knowledge",
            "depth": 4,
            "name": "Lucid Knowledge"
        },
        {
            "url": "https://lucidmotors.com/knowledge/vehicles/air/the-basics/lucid-air-essentials",
            "depth": 2,
            "name": "Lucid Essentials"
        },
        {
            "url": "https://www.wellsfargo.com/help/",
            "depth": 3,
            "name": "Wells Fargo Help"
        }
    ]
    
    all_docs = []
    
    for site in sites:
        print(f"\n{'='*60}")
        print(f"Crawling: {site['name']}")
        print(f"URL: {site['url']}")
        print(f"{'='*60}")
        
        try:
            # Create loader
            loader = RecursiveUrlLoader(
                url=site['url'],
                max_depth=site['depth'],
                prevent_outside=True,
                timeout=20,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            # Load pages
            print(f"Loading pages (max depth: {site['depth']})...")
            docs = loader.load()
            print(f"  ✓ Loaded {len(docs)} pages")
            
            # Convert HTML to text
            print("Converting HTML to text...")
            transformer = Html2TextTransformer()
            docs = transformer.transform_documents(docs)
            print(f"  ✓ Converted {len(docs)} documents")
            
            # Filter valid documents
            print("Filtering documents...")
            valid = []
            for doc in docs:
                if is_good_document(doc.page_content, doc.metadata.get('source', '')):
                    doc.metadata['source_name'] = site['name']
                    doc.metadata['crawl_date'] = datetime.now().isoformat()
                    valid.append(doc)
            
            print(f"  ✓ Kept {len(valid)} valid documents")
            all_docs.extend(valid)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Wait between sites
        time.sleep(3)
    
    print(f"\nTotal HTML documents: {len(all_docs)}")
    return all_docs


def is_good_document(content: str, url: str) -> bool:
    """Check if document is worth keeping"""
    
    # Must have enough content
    if len(content) < 150:
        return False
    
    # Skip error pages
    bad_words = ['page not found', '404', 'error', 'access denied']
    if any(word in content.lower() for word in bad_words):
        return False
    
    # Skip pages that are mostly links
    if content.count('http') / len(content) > 0.05:
        return False
    
    return True


def extract_pdf_urls(html_docs) -> List[str]:
    """Find all PDF URLs in HTML documents"""
    
    print(f"\n{'='*60}")
    print("Extracting PDF URLs")
    print(f"{'='*60}")
    
    pdfs = set()
    
    # Regex patterns for finding PDFs
    patterns = [
        r'href=["\']([^"\']*\.pdf[^"\']*)["\'"]',
        r'(https?://[^\s<>"]+\.pdf)'
    ]
    
    for doc in html_docs:
        content = doc.page_content if hasattr(doc, 'page_content') else doc['page_content']
        source = doc.metadata.get('source', '') if hasattr(doc, 'metadata') else doc['metadata'].get('source', '')
        
        # Find all PDF links
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            
            for match in matches:
                pdf_url = match[0] if isinstance(match, tuple) else match
                
                # Clean up URL
                pdf_url = pdf_url.split('"')[0].split("'")[0].split('?')[0]
                
                # Make absolute URL
                if pdf_url.startswith('/'):
                    base = urlparse(source).netloc
                    pdf_url = f"https://{base}{pdf_url}"
                elif not pdf_url.startswith('http'):
                    pdf_url = urljoin(source, pdf_url)
                
                if pdf_url.endswith('.pdf'):
                    pdfs.add(pdf_url)
    
    pdf_list = list(pdfs)
    print(f"Found {len(pdf_list)} PDFs")
    
    # Show first few
    for i, url in enumerate(pdf_list[:3], 1):
        print(f"  {i}. {url.split('/')[-1]}")
    if len(pdf_list) > 3:
        print(f"  ... and {len(pdf_list) - 3} more")
    
    return pdf_list


def download_and_process_pdfs(pdf_urls: List[str]) -> List[Dict]:
    """Download PDFs and extract text"""
    
    if not pdf_urls:
        return []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(pdf_urls)} PDFs")
    print(f"{'='*60}")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print("ERROR: Install pypdf: pip install pypdf")
        return []
    
    pdf_docs = []
    success = 0
    
    for i, url in enumerate(pdf_urls, 1):
        filename = url.split('/')[-1]
        print(f"[{i}/{len(pdf_urls)}] {filename}")
        
        try:
            # Download PDF
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"  ✗ Failed to download (status {response.status_code})")
                continue
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            
            # Extract text from PDF
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            
            # Create document for each page
            for page_num, page in enumerate(pages, 1):
                if len(page.page_content.strip()) > 50:  # Skip empty pages
                    pdf_docs.append({
                        'page_content': page.page_content,
                        'metadata': {
                            'source': url,
                            'type': 'pdf',
                            'filename': filename,
                            'page': page_num,
                            'crawl_date': datetime.now().isoformat()
                        }
                    })
            
            # Clean up temp file
            os.unlink(temp_path)
            
            print(f"  ✓ Extracted {len(pages)} pages")
            success += 1
            
            # Rate limit
            time.sleep(2)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nProcessed {success}/{len(pdf_urls)} PDFs successfully")
    print(f"Total pages extracted: {len(pdf_docs)}")
    
    return pdf_docs


def combine_all_documents(html_docs, pdf_docs) -> List[Dict]:
    """Combine HTML and PDF documents into single list"""
    
    print(f"\n{'='*60}")
    print("Combining documents")
    print(f"{'='*60}")
    
    combined = []
    
    # Convert HTML docs to dict format
    for doc in html_docs:
        combined.append({
            'page_content': doc.page_content if hasattr(doc, 'page_content') else doc['page_content'],
            'metadata': {
                **(doc.metadata if hasattr(doc, 'metadata') else doc['metadata']),
                'type': 'html'
            }
        })
    
    # Add PDF docs
    combined.extend(pdf_docs)
    
    html_count = sum(1 for d in combined if d['metadata']['type'] == 'html')
    pdf_count = sum(1 for d in combined if d['metadata']['type'] == 'pdf')
    
    print(f"Total: {len(combined)} documents")
    print(f"  • HTML: {html_count}")
    print(f"  • PDF: {pdf_count}")
    
    return combined


def save_to_file(documents: List[Dict], output_dir: str = 'AgenticRag/data/processed'):
    """Save documents to your project directory structure"""
    
    print(f"\n{'='*60}")
    print("Saving to files")
    print(f"{'='*60}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate stats
        total_chars = sum(len(d['page_content']) for d in documents)
        
        output = {
            'metadata': {
                'crawl_date': datetime.now().isoformat(),
                'total_documents': len(documents),
                'html_docs': sum(1 for d in documents if d['metadata']['type'] == 'html'),
                'pdf_docs': sum(1 for d in documents if d['metadata']['type'] == 'pdf'),
                'total_characters': total_chars
            },
            'documents': documents
        }
        
        # 1. Save to scraped_documents.json (raw scraped data)
        scraped_path = os.path.join(output_dir, 'scraped_documents.json')
        with open(scraped_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        size_mb = os.path.getsize(scraped_path) / (1024 * 1024)
        print(f"✓ Saved scraped data to: {scraped_path}")
        print(f"  • File size: {size_mb:.2f} MB")
        print(f"  • Documents: {len(documents)}")
        print(f"  • Characters: {total_chars:,}")
        
        # 2. Prepare for chunking (chunked_documents.json will be created by chunking process)
        print(f"\n✓ Next steps:")
        print(f"  1. Run chunking script to create: chunked_documents.json")
        print(f"  2. Run embedding script to create: weaviate_ready_docs.json")
        
    except Exception as e:
        print(f"✗ Error saving: {e}")


def main():
    """Main crawler - run everything"""
    
    start_time = time.time()
    
    print("\n" + "="*60)
    print("WEB CRAWLER FOR AGENTIC RAG")
    print("="*60)
    
    # Step 1: Crawl websites for HTML
    print("\n[1/4] Crawling websites...")
    html_docs = scrape_websites()
    
    if not html_docs:
        print("\n✗ No documents found. Exiting.")
        return
    
    # Step 2: Extract PDF URLs from HTML
    print("\n[2/4] Finding PDFs...")
    pdf_urls = extract_pdf_urls(html_docs)
    
    # Step 3: Download and process PDFs
    print("\n[3/4] Processing PDFs...")
    pdf_docs = download_and_process_pdfs(pdf_urls)
    
    # Step 4: Combine and save
    print("\n[4/4] Finalizing...")
    all_docs = combine_all_documents(html_docs, pdf_docs)
    save_to_file(all_docs)
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Time taken: {elapsed:.1f} seconds")
    print(f"Documents scraped: {len(all_docs)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()