# Agentic RAG System

Multi-agent RAG system with intelligent retrieval planning and hybrid search.

## Overview

This system uses multiple specialized agents to handle complex queries:
- Intent Router: Classifies query intent (faq, troubleshooting, procedural)
- Planner: Decomposes complex queries and creates retrieval strategies
- Hybrid Retrieval: Executes vector + keyword search with dynamic parameters
- Validator: Ensures responses are grounded, safe, and complete

## Tech Stack

- **Vector Database**: Weaviate (local)
- **LLM**: Groq (openai/gpt-oss-120b)
- **Framework**: LangGraph
- **Search**: Hybrid (vector embeddings + BM25)

## Prerequisites
- python 3.10+
- weaviate (running locally on port 8080)
- groq api key

## Setup
### Clone the repository
```bash
git clone https://github.com/kharmandeep/Agentic-RAG.git
cd Agentic-RAG
```

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set environment variables
- Add your GROQ_API_KEY to .env
- GROQ_API_KEY="YOUR_GROQ_API_KEY"

### Start Weaviate
```bash
docker-compose up -d
```

### Create schema and ingest documents
```bash
python src/core/utils/weaviate_schema.py
python src/core/utils/ingest_documents.py
```

### Usage
- Run the agent
```bash
python src/core/simple_agent.py
```

**Or modify the query in the file:**
- pythonquery = "What are the electricity sector decarbonization scenarios?"
- result = agent.invoke({"query": query, "context": ""})

**Example Queries**
- What are the electricity sector decarbonization scenarios?
- What is the Mid-case assumption for renewable energy modeling?
- Compare the 95% by 2050 and 100% by 2035 decarbonization scenarios
- What electricity decarbonization pathways are being considered?

## System Architecture

### Flow Diagram
```mermaid
User Query
    ↓
[Orchestrator - Input Guardrails] # TODO
    ↓   
[Intent Router](classify: FAQ/troubleshooting/procedural)
    ↓
[Planner](Creates sub-questions + retrieval strategy)
    ↓
[Hybrid Retrieval](Executes the plan)
    ↓
[Validator](check relevance & grounding)
    ↓
[Synthesizer](generate answer with citations) # TODO : citations
    ↓
[Orchestrator - Output Guardrails] # TODO       
    ↓
Final Answer

### Folder Structure
Agentic-RAG/
├── src/
│   ├── agents/
│   │   └── rag_agent.py                # Main agent workflow
│   └── utils/
│       ├── retrieval_planning.py       # Query classification, planning logic
│       ├── chunking.py                 # Document chunking
│       ├── crawling.py                 # Web scraping
│       ├── data_preparation.py         # Data preprocessing
│       ├── ingest_documents.py         # Weaviate ingestion
│       └── weaviate_schema.py          # Database schema
├── scripts/
│   └── crawl.py                        # Crawling script
├── data/
│   └── processed/                      # Processed documents
├── tests/
│   ├── check_data_quality.py          # Data validation
│   ├── test_rag_chain.py              # RAG tests
│   └── test_retrieval.py              # Retrieval tests
├── docs/
│   └── DESIGN.md                      # Detailed architecture
├── compose.yml                        # Docker Compose for Weaviate
├── requirements.txt
└── README.md                         # Project overview