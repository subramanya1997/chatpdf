import os
import logging
import json
import re
from typing import List, Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
import chainlit as cl
from prompt_manager import PromptManager
from examples_manager import ExamplesManager
from pydantic import BaseModel
from pinecone import Pinecone
import tiktoken
from logging_config import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger.info("OpenAI client initialized")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
logger.info("Pinecone client initialized")

idx_data = pc.describe_index("impax-walmart")
index = pc.Index(host=idx_data["host"])
logger.info("Pinecone index initialized")

prompt_manager = PromptManager()
examples_manager = ExamplesManager()
logger.info("Managers initialized")

# Global dictionary to store chat history for each session
chat_sessions: Dict[str, List] = {}

class Step(BaseModel):
    stepNumber: int
    description: str
    clarifyingQuestion: str
    rationale: str

class QueryPlan(BaseModel):
    queryRestatement: str
    steps: List[Step]
    searchPlan: List[str]
    notes: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    logger.info("New chat session started")
    
    # Initialize an empty chat history for this session
    session_id = cl.user_session.get("id") if hasattr(cl, "user_session") and cl.user_session else None
    if not session_id:
        # Fallback to client ID if user_session.id is not available
        session_id = str(cl.client.id) if hasattr(cl, "client") and cl.client else "default"
    
    logger.info(f"Session ID: {session_id}")
    chat_sessions[session_id] = []

# Function to get the current session's chat history
def get_chat_history():
    session_id = cl.user_session.get("id") if hasattr(cl, "user_session") and cl.user_session else None
    if not session_id:
        session_id = str(cl.client.id) if hasattr(cl, "client") and cl.client else "default"
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    return chat_sessions[session_id]

# Function to update the current session's chat history
def update_chat_history(history):
    session_id = cl.user_session.get("id") if hasattr(cl, "user_session") and cl.user_session else None
    if not session_id:
        session_id = str(cl.client.id) if hasattr(cl, "client") and cl.client else "default"
    
    chat_sessions[session_id] = history

async def understand_user_question(user_question: str, documents: str):
    logger.debug(f"Processing user question: {user_question}")
    examples = examples_manager.get_example("query_understanding")
    example_messages = []
    for example in examples:
        example_messages.extend([{
            "role": "user",
            "content": f"<SearchQuery> {example['query']} </SearchQuery>"
        }, {
            "role": "assistant", 
            "content": json.dumps(example['response'])
        }])
    
    logger.debug(f"Generated {len(example_messages)//2} example messages")

    # chat history
    chat_history = get_chat_history()
    if len(chat_history) > 0:
        chat_history_str = "<ChatHistory>\n"
        # get last 10 messages
        for msg in chat_history[-10:]:
            chat_history_str += f"{msg.role}: {msg.content}\n"
        chat_history_str += "</ChatHistory>"
    else:
        chat_history_str = documents
    
    
    response = await cl.make_async(client.beta.chat.completions.parse)(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": prompt_manager.get_prompt("query_understanding")
            },
            *example_messages,
            {
                "role": "user",
                "content": f"{chat_history_str}\n<SearchQuery>{user_question}</SearchQuery>"
            }
        ],
        temperature=0.0,
        response_format=QueryPlan
    )
    
    try:
        query_plan = QueryPlan(**json.loads(response.choices[0].message.content))
        logger.info(f"Query plan generated successfully")
        logger.debug(f"Query plan: {query_plan}")
        return query_plan
    except Exception as e:
        logger.error(f"Error parsing response: {e}", exc_info=True)
        await cl.Message(
            content=f"Error in query understanding: {str(e)}",
            tags=["Error"]
        ).send()
        raise

def hybrid_score_norm(dense, sparse, alpha: float):
    """Hybrid score using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: scale between 0 and 1
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hs = {
        'indices': sparse.sparse_indices,
        'values':  [v * (1 - alpha) for v in sparse.sparse_values]
    }
    return [v * alpha for v in dense], hs

async def analyze_search_results(dense_embeddings: List[float], sparse_embeddings: dict, top_k: int = 15, namespace: str = None):
    logger.debug("Starting search analysis")
    logger.debug(f"Dense embeddings shape: {len(dense_embeddings)}")
    logger.debug(f"Sparse embeddings: {sparse_embeddings}")

    hdense, hsparse = hybrid_score_norm(dense_embeddings, sparse_embeddings, alpha=0.75)
    logger.debug("Hybrid scores computed")

    responses = index.query(
        top_k=top_k,
        include_metadata=True,
        vector=hdense,
        sparse_vector=hsparse,
        namespace=namespace
    )
    logger.info(f"Retrieved {len(responses['matches'])} matches from index")

    documents = {}
    # logger.info(f"Responses: {responses}")
    for response in responses['matches']:
        if response['score'] > 1:
            documents[response['id']] = response['metadata']
    
    logger.debug(f"Processed {len(documents)} unique documents")
    return documents

async def get_ai_response(system_prompt: str, query_plan: QueryPlan, documents: str, user_question: str, chat_history: List[ChatMessage] = None):
    # Create a structured message using the query plan
    structured_query = (
        f"Documents:\n{documents}\n\n"
        f"Steps to consider:\n"
        + "\n".join(f"{step.stepNumber}. {step.description}" for step in query_plan.steps)
        + f"\n\nUser Question: {user_question}\n\n"
        + f"Response Guidelines:\n"
        + f"1. Strictly source-based answers\n"
        + f"2. Do not incorporate external knowledge, assumptions, or interpretations beyond what is explicitly stated in the documents.\n"
        + f"3. Detailed and precise answers based solely on the document's content.\n"
        + f"4. Handling insufficient information: If a question cannot be answered with the provided documents, state this explicitly and nothing else.\n"
    )

    # Create messages array with system prompt and chat history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history if available
    if chat_history and len(chat_history) > 0:
        for msg in chat_history:
            messages.append({"role": msg.role, "content": msg.content})
    
    # Add the user query
    messages.append({"role": "user", "content": structured_query})
    
    response = await cl.make_async(client.chat.completions.create)(
        model="gpt-4o",
        messages=messages,
        # reasoning_effort="low",
        stream=True
    )
    return response

async def get_dense_embeddings(query: List[str]):
    logger.debug(f"Getting dense embeddings for query: {query}")
    dense_embed = await cl.make_async(client.embeddings.create)(
        model="text-embedding-3-small",
        input=query,
        dimensions=512
    )
    return [data.embedding for data in dense_embed.data]

async def get_sparse_embeddings(query: List[str]):
    sparse_embed = await cl.make_async(pc.inference.embed)(
        model="pinecone-sparse-english-v0",
        inputs=query,
        parameters={
            "input_type": "query",
        }
    )
    return sparse_embed.data

def process_documents(pdocs: dict, include_source: bool = True):
    if include_source:
        document_format = """
        <Document>
        <Content>{content}</Content>
        <Source>{source}</Source>
        </Document>
        """
    else:
        document_format = """
        <Document>
        <Content>{content}</Content>
        </Document>
        """
    final_documents = """
    <Documents>
    {documents}
    </Documents>
    """
    documents = ""
    for _, metadata in pdocs.items():
        document = document_format.format(**metadata)
        documents += document
    return final_documents.format(documents=documents)

@cl.on_message
async def main(message: cl.Message):
    try:            
        logger.info(f"Processing new message: {message.content[:100]}...")
        all_documents = {}
        
        # Get chat history
        chat_history = get_chat_history()
        
        async with cl.Step(show_input=False) as step:
            step.name = "Query Understanding"
            await step.update()
            logger.debug("Starting query understanding step")
            dense_embeddings = await get_dense_embeddings([message.content])
            sparse_embeddings = await get_sparse_embeddings([message.content])
            logger.debug(f"Generated embeddings for user query {message.content}")
            search_results = await analyze_search_results(dense_embeddings[0], sparse_embeddings[0], top_k=50, namespace="summary")
            sources = process_documents(search_results, include_source=False)
            num_tokens = tiktoken.encoding_for_model("gpt-4o").encode(sources)
            logger.info(f"Number of tokens in sources: {len(num_tokens)}")
            query_plan = await understand_user_question(message.content, sources)

            logger.info("Generating embeddings")
            dense_embeddings = await get_dense_embeddings(query_plan.searchPlan + [message.content])
            sparse_embeddings = await get_sparse_embeddings(query_plan.searchPlan + [message.content])
            logger.debug(f"Generated embeddings for {len(query_plan.searchPlan) + 1} queries")

            for search_step, dense_embedding, sparse_embedding in zip(query_plan.searchPlan, dense_embeddings, sparse_embeddings):
                step.name = f"Searching for {search_step}"
                await step.update()
                logger.info(f"Processing search step: {search_step}")
                search_results = await analyze_search_results(dense_embedding, sparse_embedding)
                all_documents.update(search_results)
            await step.remove()
        
        sources = process_documents(all_documents)
        num_tokens = tiktoken.encoding_for_model("gpt-4o").encode(sources)
        logger.info(f"Number of tokens in sources: {len(num_tokens)}")

        step.name = "Generating Response"
        await step.update()
        logger.debug("Starting response generation")
        response = await get_ai_response(
            prompt_manager.get_prompt("system"),
            query_plan,
            sources,
            message.content,
            chat_history[-10:]
        )

        msg = cl.Message(content="")
        logger.debug("Created empty message for streaming")

        async with cl.Step(name="Analyzing and generating response", show_input=False, type="llm") as step:
            sources_files = {}
            for _, metadata in all_documents.items():
                if metadata["file_name"] not in sources_files:
                    sources_files[metadata["file_name"]] = []
                sources_files[metadata["file_name"]].append(int(metadata['page_no']))
            
            logger.debug(f"Processing {len(sources_files)} source files")
            
            # Stream the response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    await msg.stream_token(chunk.choices[0].delta.content)
                    full_response += chunk.choices[0].delta.content

            # process and get only sources from full_response
            file_names = list({match[1] for match in re.findall(r'\[(.*?)\]\((.*?):', full_response)})
            logger.info(f"Referenced files in response: {file_names}")

            msg.elements = [
                cl.File(
                    name=f"{file_name}: {', '.join(map(str, sources_files.get(file_name, [])))}",
                    content=f"Page Number: {', '.join(map(str, sources_files.get(file_name, [])))}",
                    type="application/pdf"
                )
                for file_name in file_names
            ]
            await step.update()
        
        # Update chat history
        chat_history.append(ChatMessage(role="user", content=message.content))
        chat_history.append(ChatMessage(role="assistant", content=full_response))
        
        # Keep only the last 20 messages to avoid context length issues
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
            
        # Save updated chat history
        update_chat_history(chat_history)
        
        logger.info(f"Chat history updated, now contains {len(chat_history)} messages")
        logger.info("Sending final response")
        await msg.send()

    except Exception as e:
        logger.error(f"Error in main handler: {e}", exc_info=True)
        await cl.Message(
            content="❌ I apologize, but I encountered an error while processing your request. Please try again."
        ).send()
