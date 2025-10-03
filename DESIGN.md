# Agentic RAG System - Design Document

## Overview
Building an Agentic RAG (Retrieval Augmented Generation) system using web-scraped data from Lucid Motors and Wells Fargo websites, with hybrid retrieval capabilities (semantic + keyword search).

## Architecture

### 1. Data Collection
Raw Documents(total:240(html 146 + pdf 94) )
    ↓
Chunked Documents (1756 chunks)
    ↓
Validated & Prepared (1756 docs)
    ↓
Weaviate Database (1756 docs)

**Web Scraping**
- Target sites: Lucid Motors (knowledge base, Air models), Wells Fargo (help center)
- Tools: LangChain RecursiveUrlLoader, Html2TextTransformer
- Output: HTML pages + PDFs
- Total documents: ~240 documents (HTML + PDF combined)

**PDF Processing**
- Extracted PDF URLs from HTML content
- Processed using LangChain PDF loaders (UnstructuredPDFLoader/PyPDFLoader)
- Combined with HTML content for comprehensive coverage

### 2. Document Processing
**Chunking**
- Tool: RecursiveCharacterTextSplitter
- Chunk size: 1000 characters
- Overlap: 200 characters
- Output: 1,756 searchable chunks

**Metadata Preserved**
- source (URL)
- domain (lucidmotors.com, wellsfargo.com)
- doc_type (html, pdf)
- chunk_id, chunk_index
- section_title

### 3. Vector Store
**Database**: Weaviate (local deployment)

**Schema**
```python
- content (TEXT, searchable)
- source (TEXT, filterable)
- domain (TEXT, filterable)
- doc_type (TEXT, filterable)
- chunk_id (TEXT)
- chunk_index (INT)
- section_title (TEXT)


User Query
    ↓
Embedding Model (all-mpnet-base-v2)
    ↓
Vector Search in Weaviate
    ↓ (target_vector="embedding")
Retrieved Documents (top 3)
    ↓
Context Formatting
    ↓
Prompt Template
    ↓
Groq LLM (openai/gpt-oss-120b)
    ↓
Final Answer