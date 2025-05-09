import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import chainlit as cl
from prompt_manager import PromptManager
from document_manager import DocumentManager

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Prompt Manager
prompt_manager = PromptManager()

# Initialize Document Manager for markdown files
document_manager = DocumentManager("mdfiles")

@cl.on_chat_start
async def on_chat_start():
    await cl.send_window_message("Server: Hello from Chainlit")


async def process_query_understanding(
    message_content,
    chat_history
):
    """
    Process the user's query through the query understanding prompt.
    
    Args:
        message_content (str): The content of the user's message
        chat_history (list): The chat history in OpenAI format
        
    Returns:
        dict or str: The analyzed query as a JSON object, or the original message content if an error occurs
    """
    try:
        query_understanding_prompt = prompt_manager.get_prompt("query_understanding")
        product_catalog = prompt_manager.get_prompt("product_catalog")
        user_message = ""

        if len(chat_history) > 1:
            user_message_template = """<PRODUCT_CATALOG>
            {product_catalog}
            </PRODUCT_CATALOG>
            
            <CHAT_HISTORY>
            {chat_history}
            </CHAT_HISTORY>
            
            {user_message}
            """
            chat_history = "\n".join([f"<{msg['role']}> {msg['content']} </{msg['role']}>" for msg in chat_history[:-1]])
            user_message = user_message_template.format(
                product_catalog=product_catalog,
                chat_history=chat_history,
                user_message=message_content
            )
        else:
            user_message_template = """<PRODUCT_CATALOG>
            {product_catalog}
            </PRODUCT_CATALOG>
            
            {user_message}
            """
            user_message = user_message_template.format(
                product_catalog=product_catalog,
                user_message=message_content
            )
        
        # Process the query with the understanding prompt
        query_understanding_response = await cl.make_async(client.chat.completions.create)(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": query_understanding_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            response_format={
                "type": "json_object"
            }
        )
        
        # Extract the analyzed query information
        analyzed_query = query_understanding_response.choices[0].message.content
        return json.loads(analyzed_query)
        
    except Exception as e:
        return message_content

@cl.on_message
async def on_message(message: cl.Message, action: cl.Action = None):
    try:
        msg = cl.Message(content="")
        # First run the query through query understanding
        analyzed_query = await process_query_understanding(
            message_content=message.content,
            chat_history=cl.chat_context.to_openai()
        )
        if type(analyzed_query) != dict:
            user_message = cl.Message(
                content="I'm sorry, but I don't understand your query. Please try again."
            )
            await user_message.send()
            return

        if analyzed_query["intent"] == "MULTIPLE":
            actions = [
                cl.Action(name=f"option_{i}", payload={"value": option}, label=option)
                for i, option in enumerate(analyzed_query["candidates"])
            ]
            
            res = await cl.AskActionMessage(
                content=analyzed_query["clarification"],
                actions=actions
            ).send()
            
            if res:
                await on_message(cl.Message(content=res.get("payload").get("value")), action=res)
            return
        elif analyzed_query["intent"] == "AMBIGIOUS":
            user_message = cl.Message(
                content=analyzed_query["clarification"]
            )
            await user_message.send()
            return
                
        try:
            product_content = document_manager.get_document(analyzed_query["product_source"])
            product_response_prompt = prompt_manager.get_prompt("product_response")

            user_message = """
            <PRODUCT_CONTENT>
            {product_content}
            </PRODUCT_CONTENT>
            
            {user_query}
            """.format(
                product_content=product_content,
                user_query=analyzed_query["user_query"]
            )

            messages = [
                {"role": "system", "content": product_response_prompt},
                *cl.chat_context.to_openai()[:-1],
                {"role": "user", "content": user_message}
            ]

            response = await cl.make_async(client.chat.completions.create)(
                model="gpt-4.1",
                messages=messages,
                stream=True
            )

            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    await msg.stream_token(chunk.choices[0].delta.content)
                    full_response += chunk.choices[0].delta.content

            await msg.send()

        except Exception as e:
            await cl.Message(
                content="❌ I apologize, but I encountered an error while processing your request. Please try again."
            ).send()
    except Exception as e:
        await cl.Message(
            content="❌ I apologize, but I encountered an error while processing your request. Please try again."
        ).send()
        
