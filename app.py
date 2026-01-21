import asyncio
import sys
import logging
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from semantic_search import get_answer as get_answer_from_semantic_search
from recursive_crawl import build_knowledge_base as build_knowledge_base_from_url
from pinecone_client import upsert_data
from file_processor import extract_text, chunk_text, format_records_for_pinecone
from pydantic import BaseModel

# Fix Windows event loop policy for subprocess support
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

# Configure CORS middleware
# Allow requests from frontend development servers and production domains
origins = [
    "http://localhost:3000",  # React default port
    "http://localhost:3001",  # Alternative React port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    # Production domains - add your frontend URL here
    # "https://yourdomain.com",
    # "https://your-app.onrender.com",
]

# For development/testing, you can use allow_origins=["*"] but it's less secure
# In production, specify exact origins above

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

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
    history: Optional[List[Dict[str, str]]] = None

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
        history = user_query.history
        
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="User query cannot be empty")
        
        if not assistant or not assistant.strip():
            raise HTTPException(status_code=400, detail="Assistant name cannot be empty")
        
        answer = await get_answer_from_semantic_search(q, assistant, history)
        
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


@app.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    assistant: str = Form(...)
):
    """
    Upload a file (PDF, TXT, DOCX, PPT) and store its content in Pinecone.
    
    Args:
        file: Uploaded file (PDF, TXT, DOCX, DOCS, PPT, PPTX)
        assistant: Assistant identifier for filtering
        
    Returns:
        dict: Status and message about the file upload process
        
    Raises:
        HTTPException: If there's an error processing the file
    """
    try:
        # Validate assistant parameter
        if not assistant or not assistant.strip():
            raise HTTPException(status_code=400, detail="Assistant name cannot be empty")
        
        assistant = assistant.strip()
        
        # Validate file extension
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else ''
        allowed_extensions = ['pdf', 'txt', 'docx', 'docs', 'ppt', 'pptx']
        
        if file_extension.lower() not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        # Check if file is empty
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # File size limit check (1MB)
        max_size = 1 * 1024 * 1024  # 1MB in bytes
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of 1MB"
            )
        
        logger.info(f"Processing file '{file.filename}' for assistant '{assistant}'")
        
        # Extract text from file
        try:
            extracted_text = extract_text(file_content, file_extension)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error extracting text from file: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to extract text from file: {str(e)}"
            )
        
        # Validate extracted text
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the file"
            )
        
        # Chunk the text
        chunks = chunk_text(extracted_text, chunk_size=2000, overlap=100)
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No valid chunks could be created from the extracted text"
            )
        
        # Format records for Pinecone
        base_id = f"{assistant}_file_{file.filename.replace('.', '_').replace(' ', '_')}"
        records = format_records_for_pinecone(chunks, assistant, base_id)
        
        # Upsert to Pinecone
        try:
            upsert_result = upsert_data(records)
            logger.info(f"Successfully processed file '{file.filename}': {upsert_result}")
            
            return {
                "status": "success",
                "message": "File uploaded and processed successfully",
                "chunks_created": len(chunks),
                "assistant": assistant,
                "filename": file.filename
            }
        except ValueError as validation_error:
            logger.error(f"Validation error upserting data to Pinecone: {str(validation_error)}")
            raise HTTPException(
                status_code=400,
                detail=f"Data validation failed: {str(validation_error)}"
            )
        except Exception as upsert_error:
            logger.error(f"Error upserting data to Pinecone: {str(upsert_error)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store data in Pinecone: {str(upsert_error)}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in upload_file endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
