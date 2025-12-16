import asyncio
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from semantic_search import get_answer as get_answer_from_semantic_search
from recursive_crawl import build_knowledge_base as build_knowledge_base_from_url
from pydantic import BaseModel

# Fix Windows event loop policy for subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

class UserQuery(BaseModel):
    user_query: str
    assistant: str

class KnowledgeBase(BaseModel):
    url_to_scrape: str
    assistant: str

class GetAnswerResponse(BaseModel):
    response: str

@app.get("/")
async def read_root():
    return {"response": "Hello World!"}

@app.post("/get_answer")
async def get_answer(user_query: UserQuery):
    """
    Get an answer to a user query using semantic search.
    
    Args:
        user_query: UserQuery object containing query and assistant name
        
    Returns:
        GetAnswerResponse with the answer
        
    Raises:
        HTTPException: If there's an error processing the query
    """
    try:
        q = user_query.user_query
        assistant = user_query.assistant
        
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="User query cannot be empty")
        
        if not assistant or not assistant.strip():
            raise HTTPException(status_code=400, detail="Assistant name cannot be empty")
        
        answer = await get_answer_from_semantic_search(q, assistant)
        
        if not answer:
            answer = "I couldn't find relevant information to answer your question. Please try rephrasing or ensure the knowledge base has been built."
        
        return GetAnswerResponse(response=answer)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_answer endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/build_knowledge_base")
async def build_knowledge_base(knowledge_base: KnowledgeBase):
    """
    Build a knowledge base by crawling a URL and storing content in Pinecone.
    
    Args:
        knowledge_base: KnowledgeBase object containing URL and assistant name
        
    Returns:
        dict: Status and message about the knowledge base build process
        
    Raises:
        HTTPException: If there's an error building the knowledge base
    """
    try:
        url_to_scrape = knowledge_base.url_to_scrape
        assistant = knowledge_base.assistant
        
        if not url_to_scrape or not url_to_scrape.strip():
            raise HTTPException(status_code=400, detail="URL cannot be empty")
        
        if not assistant or not assistant.strip():
            raise HTTPException(status_code=400, detail="Assistant name cannot be empty")
        
        # Basic URL validation
        if not url_to_scrape.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
        
        logger.info(f"Building knowledge base for assistant '{assistant}' from URL: {url_to_scrape}")
        
        response = await build_knowledge_base_from_url(url_to_scrape, assistant)
        
        # Handle different response statuses
        if response.get("status") == "error":
            raise HTTPException(status_code=500, detail=response.get("message", "Unknown error"))
        
        return {
            "status": response.get("status", "success"),
            "message": response.get("message", "Knowledge base built successfully"),
            "pages_crawled": response.get("pages_crawled", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in build_knowledge_base endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error building knowledge base: {str(e)}")
