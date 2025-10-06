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

Many documents are navigation/headers, not content
Works well only with NREL renewable energy queries
Lucid Motors/Wells Fargo content too shallow

**Validation Failures:**

LLM hallucinates when documents lack detail
"Not grounded" even with correct retrieval

### 6.  Future Enhancements

- Agent: Add Orchestrator Agent
- Response Format as per the requirements
- Re-ranking: Add cross-encoder after retrieval
- Better Data: Curate database with substantive content
- Evaluation: Implement RAGAS for automated testing
- API Deployment: FastAPI server for production use

