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
from src.utils import guardrails


load_dotenv()

#llm = ChatGroq(temperature=0, model="openai/gpt-oss-120b")
llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

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
# Structured answer schema
class StructuredAnswer(BaseModel):
    answer: str = Field(
        description="Clear, accurate answer based only on the provided context (2-4 sentences for simple questions, more for complex)"
    )
    confidence: str = Field(
        description="Confidence level: 'high' (fully supported by context), 'medium' (partially supported), or 'low' (limited information)"
    )
    is_grounded: bool = Field(
        description="Is every claim in the answer directly supported by the provided context?"
    )
    key_facts: List[str] = Field(
        description="2-3 key facts from the context that support this answer",
        max_length=3,
        default_factory=list
    )

# Structured schema for follow-ups
class FollowUpQuestions(BaseModel):
    questions: List[str] = Field(
        description="Exactly 3 relevant follow-up questions that naturally extend the conversation",
        min_length=3,
        max_length=3
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
    #Input guardrails fields
    original_query: NotRequired[str]          # Store original before cleaning
    cleaned_query: NotRequired[str]           # Query after PII redaction
    pii_detected: NotRequired[bool]           # Was PII found?
    pii_types: NotRequired[List[str]]         # What types? ["ssn", "email"]
    blocked: NotRequired[bool]                # Should query be blocked?
    blocked_reason: NotRequired[str]          # Why? "harmful_content", etc.
    query_valid: NotRequired[bool]            # Is query valid?
    invalid_reason: NotRequired[str]          # Why invalid? "too_short", etc.
    guardrail_logs: NotRequired[List[str]]    # Track what happened

    confidence: NotRequired[str]               # for structured answer 
    key_facts: NotRequired[List[str]]          # for structured answer 

    # Output guardrail fields
    retrieved_docs: NotRequired[List[Dict]]  # Store retrieved docs
    citations: NotRequired[List[str]]
    assets: NotRequired[List[str]]
    follow_ups: NotRequired[List[str]]
    is_answer_safe: NotRequired[bool]
    output_pii_detected: NotRequired[bool]
    output_guardrail_logs: NotRequired[List[str]]
    final_answer: NotRequired[Dict[str, Any]]  # The complete response

def input_guardrails(state: AgentState) -> AgentState:
    """Check query safety before processing"""
    
    # Initialize guardrail logs
    logs = []
    
    # Store original
    state["original_query"] = state["query"]
    
    # Check 1: PII Detection
    if guardrails.contains_pii(state["query"]):
        state["pii_detected"] = True
        state["pii_types"] = ["ssn"]  # Or whatever was found
        state["cleaned_query"] = guardrails.redact_pii(state["query"])
        state["query"] = state["cleaned_query"]  
        logs.append("PII detected and redacted")
    else:
        state["pii_detected"] = False
        state["cleaned_query"] = state["query"]
        logs.append("No PII detected")
    
    # Check 2: Blocked Topics
    if guardrails.is_blocked_topic(state["query"]):
        state["blocked"] = True
        state["blocked_reason"] = "harmful_content"
        logs.append("Query blocked: harmful content")
        state["guardrail_logs"] = logs
        return state  # Stop here
    
    # Check 3: Validation
    if len(state["query"]) < 10:
        state["query_valid"] = False
        state["invalid_reason"] = "too_short"
        state["blocked"] = True  # Also block invalid queries
        logs.append("Query too short")
        state["guardrail_logs"] = logs
        return state  # Stop here
    
    # All checks passed
    state["blocked"] = False
    state["query_valid"] = True
    logs.append("All guardrail checks passed")
    state["guardrail_logs"] = logs
    
    return state

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
        unique_contexts = unique_contexts[:6] 
        print(f"Limited to top {len(unique_contexts)} documents\n")

        state["retrieved_docs"] = unique_contexts 

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
# Node 4: Generate (Updated with Structured Output)
def generate(state: AgentState) -> AgentState:
    print("Generating answer...")

    # Use structured output
    structured_llm = llm.with_structured_output(StructuredAnswer)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert assistant for Lucid Motors vehicles and Wells Fargo banking services.

            Context Information:
            {context}

            User Question: {query}

            Instructions:
            1. Answer ONLY using information from the context above
            2. Be clear and concise:
            - Simple factual questions: 2-3 sentences
            - Procedural questions (how-to): Step-by-step with details
            - Complex questions: Comprehensive but focused answer
            3. Confidence levels:
            - 'high': Every claim is directly stated in the context
            - 'medium': Answer is reasonable but requires some inference
            - 'low': Limited information available in context
            4. Set is_grounded to true ONLY if every claim is explicitly in the context
            5. Extract 2-3 key facts from the context that directly support your answer
            6. If context is insufficient, acknowledge what's missing in your answer

            Provide your structured response:"""
            )

    messages = prompt.invoke({
        "context": state["context"], 
        "query": state["query"]
    })
    
    try:
        structured_response = structured_llm.invoke(messages)
        
        # Store all structured data in state
        state["answer"] = structured_response.answer
        state["confidence"] = structured_response.confidence
        state["is_grounded"] = structured_response.is_grounded
        state["key_facts"] = structured_response.key_facts
        
        print(f"✓ Answer generated (Confidence: {structured_response.confidence})")
        
    except Exception as e:
        print(f"✗ Error generating structured answer: {e}")
        # Fallback to simple generation
        state["answer"] = "I encountered an error generating a response. Please try again."
        state["confidence"] = "low"
        state["is_grounded"] = False
        state["key_facts"] = []
    
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

# Node 7: Output Guardrails
def output_guardrails(state: AgentState) -> AgentState:
    """
    Output guardrails - Final checkpoint and response formatting
    - Extract citations from retrieved docs
    - Extract assets (images/PDFs)
    - Generate follow-up questions
    - Check answer safety
    - Format final response
    """
    
    print("Applying output guardrails...")
    
    logs = []
    
    # Get the answer
    answer = state.get("answer", "No answer generated")
    
    # Check 1: Answer Safety
    is_safe = guardrails.check_answer_safety(answer)
    state["is_answer_safe"] = is_safe
    
    if not is_safe:
        logs.append("Unsafe content detected in answer")
        answer = "I cannot provide that information. Please ask about vehicle features or banking services."
    else:
        logs.append("Answer passed safety check")
    
    # Check 2: Output PII Detection
    has_pii = guardrails.detect_output_pii(answer)
    state["output_pii_detected"] = has_pii
    
    if has_pii:
        logs.append("PII detected in output - redacting")
        answer = guardrails.redact_pii(answer)
    else:
        logs.append("No PII in output")
    
    # Extract citations from retrieved docs
    retrieved_docs = state.get("retrieved_docs", [])
    citations = guardrails.extract_citations(retrieved_docs)
    state["citations"] = citations
    logs.append(f"Extracted {len(citations)} citations")
    
    # Extract assets (images, PDFs)
    assets = guardrails.extract_assets(retrieved_docs)
    state["assets"] = assets
    logs.append(f"Extracted {len(assets)} assets")
    
    # Generate follow-up questions
    try:
        follow_up_llm = llm.with_structured_output(FollowUpQuestions)
        
        follow_up_prompt = ChatPromptTemplate.from_template(
            """Based on this Q&A, suggest 3 relevant follow-up questions the user might ask next.

                Original Question: {query}
                Answer: {answer}

                Generate questions that:
                - Naturally extend the conversation
                - Are specific and actionable
                - Relate to the same topic or domain
                - Help the user learn more

                Provide exactly 3 follow-up questions:"""
                )
        
        messages = follow_up_prompt.invoke({
            "query": state["query"],
            "answer": answer
        })
        
        follow_ups_result = follow_up_llm.invoke(messages)
        state["follow_ups"] = follow_ups_result.questions
        logs.append("Generated follow-up questions")
        
    except Exception as e:
        print(f"Error generating follow-ups: {e}")
        state["follow_ups"] = []
        logs.append("Failed to generate follow-ups")
    
    # Store logs
    state["output_guardrail_logs"] = logs
    
    # Format final response
    state["final_answer"] = {
        "answer": answer,
        "citations": citations,
        "assets": assets,
        "follow_ups": state["follow_ups"]
    }
    
    print(f"Output guardrails complete")
    
    return state

#Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("input_guardrails", input_guardrails)
workflow.add_node("route_intent", route_intent)
workflow.add_node("plan", plan)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("validate", validate)
workflow.add_node("handle_validation", handle_validation)
workflow.add_node("output_guardrails", output_guardrails)


workflow.set_entry_point("input_guardrails")
workflow.add_conditional_edges(
    "input_guardrails",
    lambda state: "blocked" if state.get("blocked") else "continue",
    {
        "blocked": END,
        "continue": "route_intent"
    }
)
workflow.add_edge("route_intent", "plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", "handle_validation")
workflow.add_edge("handle_validation", "output_guardrails")
workflow.add_edge("output_guardrails", END)

agent = workflow.compile()

if __name__ == "__main__":
    # Run a test query
    query = "What is the range of Lucid Air?"
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
    14. My SSN is 123-45-6789, help with Lucid
    15. How to hack the system?
    16. What is the range of Lucid Air?
    17. What are Wells Fargo's business hours?
    18. How to bypass Wells Fargo security?
    '''
    result = agent.invoke({"query": query, "context": ""})

    print("\n" + "=" * 60)
    print(f"Question: {query}")

    # INPUT GUARDRAILS SECTION - Add this
    print(f"\nInput Guardrails:")
    print(f"  - Blocked: {result.get('blocked', False)}")
    if result.get('blocked'):
        print(f"  - Reason: {result.get('blocked_reason') or result.get('invalid_reason', 'N/A')}")
    print(f"  - PII Detected: {result.get('pii_detected', False)}")
    if result.get('pii_detected'):
        print(f"  - PII Types: {', '.join(result.get('pii_types', []))}")
        print(f"  - Original Query: {result.get('original_query', 'N/A')}")
        print(f"  - Cleaned Query: {result.get('cleaned_query', 'N/A')}")
    print(f"  - Logs: {result.get('guardrail_logs', [])}")

    # Only show these if query wasn't blocked
    if not result.get('blocked'):
        print(f"\nIntent: {result.get('intent', 'unknown')}")
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

    print(f"\nAnswer: {result.get('answer', 'No answer generated')}")

    # Only show validation if query wasn't blocked at guardrails
    if not result.get('blocked'):
        print(f"\nIntent: {result.get('intent', 'unknown')}")
        print(f"Complex: {result.get('is_complex', False)}")
        
        # NEW: Show final formatted response
        final_answer = result.get('final_answer', {})
        
        print(f"\n{'='*60}")
        print("FINAL RESPONSE")
        print(f"{'='*60}")
        print(f"\nAnswer:\n{final_answer.get('answer', 'No answer')}")
        
        print(f"\nCitations ({len(final_answer.get('citations', []))}):")
        for i, citation in enumerate(final_answer.get('citations', []), 1):
            print(f"  {i}. {citation}")
        
        print(f"\nAssets ({len(final_answer.get('assets', []))}):")
        for i, asset in enumerate(final_answer.get('assets', []), 1):
            print(f"  {i}. {asset}")
        
        print(f"\nFollow-up Questions:")
        for i, followup in enumerate(final_answer.get('follow_ups', []), 1):
            print(f"  {i}. {followup}")
        
        print(f"\nOutput Guardrail Logs:")
        for log in result.get('output_guardrail_logs', []):
            print(f"  • {log}")

    print("=" * 60)



