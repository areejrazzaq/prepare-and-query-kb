from pinecone_client import build_context
from groq import Groq
from typing import Optional
import os
from dotenv import load_dotenv
load_dotenv()   
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

async def get_answer(user_query: str, assistant: Optional[str] = "company"):
    print("**********************")
    context = build_context(user_query, assistant)

    if(context is None):
        context = ""

    completion = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL"),
        messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant who is restricted to the provided context. Based on the context and user query, you provided a well-defined response back under 500 characters. You are not allowed to provide answer or assume any answer unless context is provided. If you do not find context, excuse to user and move on to next query. After each message, ask only one follow-up question as well. You have to use the same tone and language as provided by the context"
        },
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": user_query
        }
        ],
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


