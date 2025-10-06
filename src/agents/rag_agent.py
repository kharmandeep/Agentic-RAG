# type: ignore
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# User Query →Intent Router → Generate (LLM) → Answer

from typing import TypedDict,Any,Literal,List,Dict
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END
import weaviate
from weaviate.classes.query import MetadataQuery
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
from src.utils.retrieval_planning import create_retrieval_plan,deduplicate_documents,expand_query_with_llm

load_dotenv()

llm = ChatGroq(temperature=0, model="openai/gpt-oss-120b")

# Schema for Intent Classification
class IntentClassification(BaseModel):
    intent : Literal['faq', 'troubleshooting', 'procedural'] = Field(description="Classified intent of the user query")

# Schema for Planner
class QueryPlan(BaseModel):
    is_complex: bool
    sub_questions: List[str]

#Structured Output Schema for Validator
class ValidationResult(BaseModel):
    is_grounded: bool = Field(
        description="Whether the answer is supported by the provided context"
    )
    is_safe: bool = Field(
        description="Whether the answer contains safe, appropriate content"
    )
    is_complete: bool = Field(
        description="Whether the answer fully addresses the original question"
    )
    explanation: str = Field(
        description="Brief explanation of the validation decision"
    )

#State Definition
class AgentState(TypedDict):
    query: str
    intent: NotRequired[Any] 
    is_complex: NotRequired[bool]
    sub_questions: NotRequired[List[str]] 
    retrieval_plan: NotRequired[Dict]
    context: str
    answer: NotRequired[Any]
    is_grounded: NotRequired[bool]
    is_safe: NotRequired[bool]
    is_complete: NotRequired[bool]
    validation_explanation: NotRequired[str]


# Node 1: Route Intent
# Flow: Route → Intent → Retrieve → Generate
def route_intent(state: AgentState) -> AgentState:
    #print(f"Routing intent for query: {state['query']}")
    
    llm_with_structured_ouptut = llm.with_structured_output(IntentClassification)

    prompt = ChatPromptTemplate.from_template(
        """Classify the intent of the following query into one of these categories: 
           - faq: Simple factual questions(What is X? Define Y)
           - troubleshooting : Problem-solving questions (How to fix X? Why is Y happening?)
           - procedural: Step-by-step instructions (How to set up X? Steps to do Y)
    Query : {query}
        """
    )
    messages = prompt.invoke({"query": state["query"]})
    result = llm_with_structured_ouptut.invoke(messages)

    state["intent"] = result.intent 
    
    return state

#Node 2: Plan(Decompose) Complex Queries + make a retrieval plan

#START → route_intent → plan → retrieve → generate → END
def plan(state: AgentState) -> AgentState:
    print(f"\n{'='*60}")
    print(f"PLANNER: Analyzing query")
    print(f"{'='*60}")
    print(f"Query: {state['query']}")
    
    query_lower = state['query'].lower()
    
    # Rule-based check for obvious complexity
    complexity_keywords = [' and ', ' vs ', ' versus ', 'compare', 'difference between']
    
    if any(keyword in query_lower for keyword in complexity_keywords):
        state["is_complex"] = True
        
        # Use LLM to break it down
        prompt = f"""Break this query into 2-4 clear, separate sub-questions.
                    Each sub-question should be standalone and answerable independently.

                    Original query: {state['query']}

                    List each sub-question on a new line:"""
        
        response = llm.invoke(prompt)
        
        # Parse sub-questions from response
        lines = str(response.content).split('\n')
        sub_questions = []
        for line in lines:
            cleaned = line.strip()
            # Remove numbering like "1.", "2.", etc.
            if cleaned and len(cleaned) > 10:
                # Remove leading numbers and dots
                cleaned = cleaned.lstrip('0123456789.- ')
                if cleaned:
                    sub_questions.append(cleaned)
        
        state["sub_questions"] = sub_questions
        
        print(f"Complex query detected")
        print(f"Decomposed into {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            print(f"   {i}. {sq}")
        
        # 🆕 CREATE RETRIEVAL PLAN for sub-questions
        state["retrieval_plan"] = create_retrieval_plan(
            sub_questions, 
            state['query'], 
            is_complex=True
        )
        
    else:
        state["is_complex"] = False
        state["sub_questions"] = []
        
        print(f"Simple query detected")
        
        # 🆕 CREATE RETRIEVAL PLAN for single query
        state["retrieval_plan"] = create_retrieval_plan(
            [state['query']], 
            state['query'], 
            is_complex=False
        )
    
    # 🆕 DISPLAY RETRIEVAL PLAN
    print(f"RETRIEVAL PLAN:")
    print(f"{'─'*60}")
    for i, qp in enumerate(state["retrieval_plan"]["queries"], 1):
        print(f"\nQuery {i}: {qp['query'][:50]}{'...' if len(qp['query']) > 50 else ''}")
        print(f"  • Type: {qp['query_type']}")
        print(f"  • Alpha: {qp['alpha']} (Vector: {qp['alpha']*100:.0f}%, Keyword: {(1-qp['alpha'])*100:.0f}%)")
        print(f"  • Top-K: {qp['top_k']} documents")
        print(f"  • Expand: {'Yes' if qp['expand_query'] else 'No'}")
    print(f"{'─'*60}\n")
    
    return state


#Node 3: Retrieve(Updated to Use Hybrid Search)

def retrieve(state: AgentState) -> AgentState:
    """
    Retrieve documents using hybrid search based on the retrieval plan.
    """
    print("Retrieving documents using hybrid search...")
    
    client = weaviate.connect_to_local()
    collection = client.collections.get("Documents")
    
    retrieval_plan = state.get("retrieval_plan", {})
    all_contexts = []
    
    if retrieval_plan and retrieval_plan.get("queries"):
        for query_plan in retrieval_plan["queries"]:
            query = query_plan["query"]
            alpha = query_plan["alpha"]
            top_k = query_plan["top_k"]
            should_expand = query_plan["expand_query"]
            
            print(f"  Query: {query[:60]}...")
            print(f"  Alpha: {alpha}, Top-K: {top_k}, Expand: {should_expand}")
            
            # Check if we should expand the query
            if should_expand:
                queries_to_search = expand_query_with_llm(query, llm)
                print(f"  Expanded to {len(queries_to_search)} variations:")
                for i, q in enumerate(queries_to_search, 1):
                    print(f"    {i}. {q[:50]}...")
            else:
                queries_to_search = [query]

            # Hybrid search - query param will auto-vectorize
            for search_query in queries_to_search:
                response = collection.query.hybrid(
                    query=search_query,
                    alpha=alpha,
                    limit=top_k // len(queries_to_search) + 1,  # Distribute top_k across variations
                    target_vector="embedding"
                )
            
                for obj in response.objects:
                    all_contexts.append({
                            'content': str(obj.properties.get("content", ""))
                    })
            print(f"Retrieved {len(response.objects)} documents\n")

        print(f"Total documents before deduplication: {len(all_contexts)}")   


        # Deduplicate contexts
        unique_contexts = deduplicate_documents(all_contexts)
        print(f"After deduplication: {len(unique_contexts)} documents")
        
        # Limit to prevent token overflow
        unique_contexts = unique_contexts[:10]
        print(f"Limited to top {len(unique_contexts)} documents\n")
        
        # Convert back to strings for context
        state["context"] = "\n\n".join([doc['content'] for doc in unique_contexts])
        
    else:
        print("Warning: No retrieval plan found")
        response = collection.query.near_text(
            query=state["query"],
            limit=5,
            target_vector="embedding"
        )
        contexts = [str(obj.properties.get("content", "")) for obj in response.objects]
        state["context"] = "\n\n".join(contexts)
        print(f"Retrieved {len(contexts)} documents\n")
    
    client.close()
    return state

# Node 4: Generate
def generate(state: AgentState) -> AgentState:
    print("Generating answer...")

    prompt = ChatPromptTemplate.from_template(
        "Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    )

    messages = prompt.invoke({"context": state["context"], "query": state["query"]})
    response = llm.invoke(messages)

    # Assign only the actual text content to state['answer']
    if hasattr(response, "content"):
        state["answer"] = response.content
    else:
        state["answer"] = str(response)

    #print("Answer generated")
    #print(state["answer"])

    return state

# Node 5: Validate
def validate(state: AgentState) -> AgentState:
    print("Validating answer...")
    structured_llm = llm.with_structured_output(ValidationResult)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a validator checking if an AI-generated answer is good quality.

            Original Question: {query}

            Retrieved Context: {context}

            Generated Answer: {answer}

            Validate the answer based on these criteria:

            1. GROUNDED: Is the answer supported by facts in the context? 
            - Check if claims in the answer can be traced back to the context
            - Mark FALSE if the answer includes information not in the context

            2. SAFE: Is the answer safe and appropriate?
            - Check for harmful, dangerous, or inappropriate content
            - Mark FALSE if answer suggests unsafe actions

            3. COMPLETE: Does the answer fully address the question?
            - Check if all parts of the question are answered
            - Mark FALSE if the answer is too brief or misses key aspects

            Provide your validation assessment."""
                )
    
    messages = prompt.invoke({
        "query": state["query"],
        "context": state["context"],
        "answer": state.get("answer", "")
    })
    
    result = structured_llm.invoke(messages)
    
    state["is_grounded"] = result.is_grounded
    state["is_safe"] = result.is_safe
    state["is_complete"] = result.is_complete
    state["validation_explanation"] = result.explanation
    
    return state

# --- Node 6: Handle Ungrounded Answers ---
def handle_validation(state: AgentState) -> AgentState:
    """Handle cases where answer is not grounded in context"""
    
    if not state.get("is_grounded", True):
        print("Answer not grounded - providing fallback response")
        state["answer"] = (
            "I don't have enough information in my knowledge base to answer this question accurately. "
            f"The question was: {state['query']}\n\n"
            "I can only provide answers based on the documents available to me. "
            "Please try rephrasing your question or ask about a different topic."
        )
    
    return state

#Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("route_intent", route_intent)
workflow.add_node("plan", plan)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("validate", validate)
workflow.add_node("handle_validation", handle_validation)


workflow.set_entry_point("route_intent")
workflow.add_edge("route_intent", "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", "handle_validation")
workflow.add_edge("handle_validation", END)

agent = workflow.compile()

# Run a test query
query = "What are the electricity sector decarbonization scenarios and what is the Mid-case assumption?"
'''
1. What are the electricity sector decarbonization scenarios?
2. What is the Mid-case assumption for renewable energy?
3. Compare the 95% by 2050 and 100% by 2035 decarbonization scenarios
4. What vehicles does Lucid Motors offer?
5. What is Lucid Gravity?
6. Tell me about Lucid Air pricing and features
7. What loan services does Wells Fargo provide?

8. What topics does the Lucid Knowledge Center cover?
9. What learning resources are available for Lucid Air owners?
10. What drive modes does Lucid Air have?
11.What are the electricity sector decarbonization scenarios?
12.What is the Mid-case assumption for renewable energy modeling?
13. Compare the 95% by 2050 and 100% by 2035 scenarios
'''
result = agent.invoke({"query": query, "context": ""})

print("\n" + "=" * 60)
print(f"Question: {query}")
print(f"Intent: {result.get('intent', 'unknown')}")
print(f"Complex: {result.get('is_complex', False)}")
if result.get('sub_questions'):
    print(f"Sub-questions:")
    for i, sq in enumerate(result['sub_questions'], 1):
        print(f"  {i}. {sq}")

if result.get('retrieval_plan'):
    print(f"\nRetrieval Plan:")
    for i, qp in enumerate(result['retrieval_plan']['queries'], 1):
        print(f"  Query {i}: {qp['query'][:40]}...")
        print(f"    - Type: {qp['query_type']}, Alpha: {qp['alpha']}, Top-K: {qp['top_k']}, Expand: {qp['expand_query']}")

print(f"Answer: {result.get('answer', 'No answer generated')}")
print(f"\nValidation:")
print(f"  - Grounded: {result.get('is_grounded', 'N/A')}")
print(f"  - Safe: {result.get('is_safe', 'N/A')}")
print(f"  - Complete: {result.get('is_complete', 'N/A')}")
print(f"  - Explanation: {result.get('validation_explanation', 'N/A')}")
print("=" * 60)



