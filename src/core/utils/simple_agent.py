# User Query →Intent Router → Generate (LLM) → Answer

from typing import TypedDict,Any,Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, END
import weaviate
from weaviate.classes.query import MetadataQuery
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

# --- Schema for Intent Classification ---
class IntentClassification(BaseModel):
    intent : Literal['faq', 'troubleshooting', 'procedural'] = Field(description="Classified intent of the user query")

# --- State Definition ---
class AgentState(TypedDict):
    query: str
    context: str
    answer: NotRequired[Any]  
    intent: NotRequired[Any] 

# Node 1: Route Intent
def route_intent(state: AgentState) -> AgentState:
    print(f"Routing intent for query: {state['query']}")
    llm = ChatGroq(temperature=0, model="openai/gpt-oss-120b")
    llm_with_structured_ouptut = llm.with_structured_output(IntentClassification)

    prompt = ChatPromptTemplate.from_template(
        """Classify the intent of the following query into one of these categories: 
           - faq: Simple factual questions(What is X? Define Y)
           - troubleshooting : Problem-solving questions (How to fix X? Why is Y happening?)
           - procedurla: Step-by-step instructions (How to set up X? Steps to do Y)
    Query : {query}
        """
    )
    messages = prompt.invoke({"query": state["query"]})
    result = llm_with_structured_ouptut.invoke(messages)

    state["intent"] = result.intent 
    
    return state

# --- Node 2: Retrieve ---
def retrieve(state: AgentState) -> AgentState:
    print(f"Retrieving docs for: {state['query']}")

    client = weaviate.connect_to_local()
    collection = client.collections.get("Documents")

    response = collection.query.near_text(
        query=state["query"],
        limit=3,
        target_vector="embedding"
    )

    contexts = [str(obj.properties.get("content", "")) for obj in response.objects]
    state["context"] = "\n\n".join(contexts)

    client.close()
    print(f"Retrieved {len(contexts)} docs")
    return state


# --- Node 3: Generate ---
def generate(state: AgentState) -> AgentState:
    print("Generating answer...")

    llm = ChatGroq(temperature=0, model="openai/gpt-oss-120b")

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

    print("Answer generated")
    print(state["answer"])

    return state


# --- Build Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("route_intent", route_intent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Flow: Route → Intent → Retrieve → Generate
workflow.set_entry_point("route_intent")
workflow.add_edge("route_intent", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

agent = workflow.compile()

# --- Run ---
query = "My solar panel isn't producing power. What should I check?"
result = agent.invoke({"query": query, "context": ""})

print("\n" + "=" * 60)
print(f"Question: {query}")
print(f"Intent: {result.get('intent', 'unknown')}")
print(f"Answer: {result.get('answer', 'No answer generated')}")
print("=" * 60)
