# test_rag_chain.py
import weaviate
from weaviate.classes.query import MetadataQuery
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Weaviate
client = weaviate.connect_to_local()
collection = client.collections.get("Documents")

# Setup LLM
llm = ChatGroq(
    temperature=0,
    model="openai/gpt-oss-120b"
)

# Create prompt
template = """Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Manual retriever function
def retrieve_docs(query):
    response = collection.query.near_text(
        query=query,
        limit=3,
        target_vector="embedding",
        return_metadata=MetadataQuery(distance=True)
    )
    
    contexts = []
    for obj in response.objects:
        contexts.append(obj.properties.get('content', ''))
    
    return "\n\n".join(contexts)

# Test queries
queries = [
    "What is renewable energy?",
    "Tell me about electric vehicles",
    "What are solar panels?"
]

for query in queries:
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    # Get context
    context = retrieve_docs(query)
    
    # Get response
    messages = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(messages)
    
    print(response.content)
    print("=" * 60)

client.close()