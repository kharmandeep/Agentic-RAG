from typing import TypedDict,Any,Literal,List,Dict

def classify_query_type(query: str) -> str:
    """
    Classify query into one of four types:
    - factual: Specific facts (what is, who is, when, where)
    - conceptual: Understanding concepts (how, why, explain)
    - comparative: Comparing things (vs, compare, difference)
    - temporal: Time-based (recent, latest, trends)
    """
    query_lower = query.lower()
    
    # Check for comparative
    if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better']):
        return "comparative"
    
    # Check for temporal
    elif any(word in query_lower for word in ['recent', 'latest', 'new', 'trend', '2024', '2025']):
        return "temporal"
    
    # Check for factual
    elif any(word in query_lower for word in ['what is', 'who is', 'when', 'where', 'define']):
        return "factual"
    
    # Default to conceptual
    else:
        return "conceptual"
    
def determine_alpha(query: str) -> float:
    """
    Determine vector vs keyword weight (0-1) based on query type
    Alpha closer to 0 = More keyword search (BM25)
    Alpha closer to 1 = More vector search (semantic)
    """
    query_type = classify_query_type(query)
    
    alpha_map = {
        "factual": 0.3,      # More keyword - specific facts need exact matches
        "conceptual": 0.8,   # More vector - needs semantic understanding
        "comparative": 0.6,  # Balanced - needs both context and specifics
        "temporal": 0.7      # More vector - recent context matters
    }
    
    return alpha_map.get(query_type, 0.5)  # Default to balanced

def determine_top_k(query: str, is_complex: bool) -> int:
    """
    Determine how many documents to retrieve based on query complexity and type
    """
    if is_complex:
        return 7  # Need more docs for complex queries
    else:
        query_type = classify_query_type(query)
        if query_type == "comparative":
            return 10  # Need docs from both sides of comparison
        else:
            return 5  # Standard retrieval   
        
def should_expand_query(query: str) -> bool:
    """
    Decide if query should be expanded with variations
    Expand if:
    - Query is short (less than 5 words)
    - Query is conceptual (needs semantic variations)
    """
    query_type = classify_query_type(query)
    word_count = len(query.split())
    
    # Expand if query is short OR conceptual
    return word_count < 5 or query_type == "conceptual"

def create_retrieval_plan(queries: List[str], original_query: str, is_complex: bool) -> Dict:
    """
    Create detailed retrieval plan for all queries
    Returns a dictionary with retrieval strategy for each query
    """
    plan = {
        "original_query": original_query,
        "is_complex": is_complex,
        "queries": []
    }
    
    # Create plan for each query
    for query in queries:
        query_plan = {
            "query": query,
            "query_type": classify_query_type(query),
            "alpha": determine_alpha(query),
            "top_k": determine_top_k(query, is_complex),
            "expand_query": should_expand_query(query)
        }
        plan["queries"].append(query_plan)
    
    return plan

def deduplicate_documents(docs):
    """Remove duplicate documents based on content hash"""
    seen = set()
    unique = []
    
    for doc in docs:
        content_hash = hash(doc['content'][:200])
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    
    return unique

def expand_query_with_llm(query: str, llm) -> List[str]:
    """
    Generate query variations for better search coverage.
    Returns original query + 2 variations.
    """
    prompt = f"""Generate 2 alternative phrasings of this query for better search coverage.
Keep them concise and focused on the same topic.

Original: {query}

Return only the 2 alternative queries, one per line, without numbering."""
    
    response = llm.invoke(prompt)
    
    # Parse variations
    variations = [line.strip() for line in response.content.split('\n') 
                  if line.strip() and len(line.strip()) > 5]
    
    # Return original + up to 2 variations
    return [query] + variations[:2]