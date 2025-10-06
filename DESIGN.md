# Agentic RAG System - Design Document

## Overview
Building an Agentic RAG (Retrieval Augmented Generation) system using web-scraped data from Lucid Motors and Wells Fargo websites, with hybrid retrieval capabilities (semantic + keyword search).

## Architecture

### 1. Data Collection
**Web Crawling**
Scrapes Lucid Motors and Wells Fargo websites.
- Crawls 3 starting URLs with depth 2-4
- Extracts HTML content and PDFs
- Filters invalid/error pages
- Crawled 277 documents (204 HTML + 73 PDF pages)
- Chunked into 554 searchable pieces
- Indexed all 554 chunks in Weaviate
- Ready for hybrid search (semantic + BM25)

Time taken: ~95 seconds for ingestion

### 2. Document Processing
**Chunking**
- Tool: RecursiveCharacterTextSplitter
- Loads 277 documents
- Splits into ~1000 word chunks with 200 word overlap
- Preserves metadata (source, type, chunk_index)
- Creates ~554 chunks

Runtime: 1-2 minutes

**Metadata Preserved**
- source (URL)
- domain (lucidmotors.com, wellsfargo.com)
- doc_type (html, pdf)
- chunk_id, chunk_index
- section_title

### 3. Prepare Weaviate
- Maps chunks to Weaviate schema.
- Validates all chunks
- Maps fields to schema (content, source, domain, doc_type, etc.)
- Extracts domain from URLs
- Prepares 554 documents for ingestion

Runtime: < 1 minute
**Database**: Weaviate (local deployment)(Why Weaviate:Native hybrid search support (vector + BM25))

**Schema**
Properties:
- content (TEXT, searchable)
- source (TEXT, filterable)
- domain (TEXT, filterable)
- doc_type (TEXT, filterable)
- chunk_id (TEXT)
- chunk_index (INT)
- section_title (TEXT)

Vector Configuration:
- Vectorizer: text2vec-transformers
- Source property: content
- Named vector: "embedding"

Inverted Index (BM25):
- bm25_b: 0.75
- bm25_k1: 1.2
- cleanup_interval: 60 seconds

Ingest Documents
- Loads 554 prepared documents
- Batch inserts into Weaviate
- Shows progress every 100 documents
- Verifies final count

### 3.1 Embedding and Vectorization

**Embedding Model**: `sentence-transformers/all-mpnet-base-v2`
- 768 dimensions
- General-purpose semantic understanding

**Weaviate Vectorizer**: `text2vec-transformers`
- Automatically vectorizes content on insert
- Vectorizes queries during search
- No external API calls needed

**Why This Choice**:
- Local inference (fast, no API latency)
- Free (no API costs)
- Native hybrid search support in Weaviate
- Good quality-to-speed ratio

**Alternative Rejected**: OpenAI embeddings (slower, costs money per call)

### 3.2 Language Model

**Model**: Groq API - `openai/gpt-oss-120b`
- 120B parameters
- 8,000 token context limit
- Fast inference via Groq's LPU

**Usage**:
- Intent classification
- Query decomposition  
- Query expansion
- Answer generation
- Response validation

**Why Groq**:
- Fast inference (low latency)
- Free tier available
- Good reasoning for complex tasks
- Supports structured output (Pydantic)

**Token Limit Impact**:
- Typical doc: ~500 tokens
- Safe limit: 10 documents = 5,000 tokens
- Remaining: 3,000 tokens for query + answer

### 4. Agent Implementation
```mermaid
┌──────────────────┐
│   USER QUERY     │
└───────┬──────────┘
        │
        ▼
┌────────────────────┐
│  INTENT ROUTER     │
│                    │
│ • LLM classification│
│ • Output: faq/     │
│   troubleshooting/ │
│   procedural       │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│     PLANNER        │
│                    │
│ Complexity Check   │
│ (rule-based)       │
└────────┬───────────┘
│
┌────────┴────────┐
│                 │
Simple Query      Complex Query
│                 │
│                 ▼
│         ┌──────────────┐
│         │ Decompose    │
│         │ (LLM)        │
│         │ → 2-4 sub-Q  │
│         └──────┬───────┘
│                │
└────────┬───────┘
         │
         ▼
┌────────────────────┐
│ RETRIEVAL PLAN     │
│                    │
│ For each query:    │
│ 1. Classify type   │
│    (rule-based)    │
│                    │
│ 2. Set alpha       │
│    factual: 0.3    │
│    conceptual: 0.8 │
│                    │
│ 3. Set top_k       │
│    simple: 5       │
│    complex: 7      │
│                    │
│ 4. Expand flag     │
│    short/conceptual│
└────────┬───────────┘
 
┌────────────────────┐
│ HYBRID RETRIEVAL   │
│                    │
│ For each query:    │
│ ┌────────────────┐ │
│ │ Expand? Yes    │ │
│ │ ↓              │ │
│ │ Generate 2     │ │
│ │ variations (LLM)│ │
│ └────────────────┘ │
│                    │
│ Execute hybrid     │
│ search with:       │
│ • alpha from plan  │
│ • top_k from plan  │
│                    │
│ Retrieved: ~14-28  │
│ documents          │
└───────┬───────────-┘
        │
        ▼
┌────────────────────┐
│  DEDUPLICATION     │
│                    │
│ Hash first 200     │
│ chars of each doc  │
│                    │
└───────┬────────---─┘ 
        │
        ▼
┌────────────────────┐
│  LIMIT TO 10       │
│                    │
│ Prevent token      │
│ overflow (8k limit)│
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   GENERATOR        │
│                    │
│ LLM generates      │
│ answer from        │
│ context + query    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│   VALIDATOR        │
│                    │
│ LLM checks:        │
│ • Grounded?        │
│ • Safe?            │
│ • Complete?        │
└────────┬───────────┘
         │
┌────────┴────────┐
│                 │
Grounded = True   Grounded = False
│                 │
▼                 ▼
Return Answer    Return Fallback
"Not enough info"
```
**Key Design Decisions in Flow:**

1. **Intent Router**: LLM-based for flexibility
2. **Planner Complexity**: Rule-based for speed
3. **Query Decomposition**: LLM for accuracy
4. **Type Classification**: Rule-based (no LLM call)
5. **Alpha Assignment**: Hardcoded map based on type
6. **Query Expansion**: LLM only when needed
7. **Deduplication**: Hash-based for efficiency
8. **Validation**: LLM-as-judge for nuanced checking

### 5. KNOWN ISSUES
**Data Quality:**

Data quality improved.Enhanced website crawling implemented.

**Validation Failures:**

LLM hallucinates when documents lack detail-fixed(more data added)
"Not grounded" even with correct retrieval- fixed (more data added)

### 6.  Future Enhancements

- Agent: Add Orchestrator Agent
- Response Format as per the requirements
- Re-ranking: Add cross-encoder after retrieval
- Better Data: Curate database with substantive content- Fixed
- Evaluation: Implement RAGAS for automated testing
- API Deployment: FastAPI server for production use

