from pinecone_client import build_context
from groq import Groq
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv
load_dotenv()   
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def validate_and_format_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Validate and format conversation history for Groq API.
    
    Args:
        history: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Validated list of message dictionaries
        
    Raises:
        ValueError: If history format is invalid
    """
    if not history:
        return []
    
    validated_messages = []
    for msg in history:
        if not isinstance(msg, dict):
            raise ValueError("History messages must be dictionaries")
        
        role = msg.get("role", "").strip()
        content = msg.get("content", "")
        
        if not role:
            raise ValueError("History messages must have a 'role' field")
        
        if not isinstance(content, str):
            raise ValueError("History message 'content' must be a string")
        
        # Validate role is one of the allowed values (only user and assistant allowed in history)
        valid_roles = ["user", "assistant"]
        if role.lower() not in valid_roles:
            raise ValueError(f"History message role must be one of: {valid_roles}")
        
        validated_messages.append({
            "role": role.lower(),
            "content": content
        })
    
    return validated_messages


async def get_answer(user_query: str, assistant: Optional[str] = "company", history: Optional[List[Dict[str, str]]] = None):
    print("**********************")
    context = build_context(user_query, assistant)

    if(context is None):
        context = ""
    
    # Build system message with context
    system_content = "You are a helpful assistant who is restricted to the provided context. Based on the context and user query, you provided a well-defined response back under 500 characters. You are not allowed to provide answer or assume any answer unless context is provided. If you do not find context, excuse to user and move on to next query. After each message, ask only one follow-up question as well. You have to use the same tone and language as provided by the context. If you have user-name you can refer it every now and then to make conversation seem more natural."
        
    # Build messages array
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "assistant",
            "content": context
        }
    ]
    

    # Validate and append history messages if provided (only user/assistant)
    if history:
        validated_history = validate_and_format_history(history)
        messages.extend(validated_history)
    
    # Add current user query
    messages.append({
        "role": "user",
        "content": user_query
    })

    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=messages,
        temperature=0.7,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
        stop=None
    )

    answer = ""
    for chunk in completion:
        answer += (chunk.choices[0].delta.content or "")

    return answer


