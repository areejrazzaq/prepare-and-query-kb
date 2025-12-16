# Knowledgebase Builder

A Retrieval-Augmented Generation (RAG) API built with FastAPI, powered by Crawl4AI, Pinecone, and Groq. This system enables you to crawl websites, build knowledge bases, and query them using semantic search with LLM-powered responses.

## Features

- **Intelligent web crawling**: Recursive crawling with depth control, content filtering, and smart page prioritization using Crawl4AI
- **Multi-assistant support**: Build and query separate knowledge bases for different assistants/companies
- **Semantic search**: Vector-based retrieval using Pinecone with integrated embeddings
- **Token-aware context building**: Automatically limits context to 7000 tokens for optimal LLM performance
- **Content cleaning**: Automatic removal of navigation, headers, and other non-content elements
- **RESTful API**: Clean FastAPI endpoints for building knowledge bases and querying them
- **Streaming responses**: Efficient streaming of LLM responses for better user experience
- **Error handling**: Comprehensive error handling and logging throughout the pipeline

## Prerequisites

- Python 3.10+
- Pinecone API key and index configuration
- Groq API key (for LLM responses)
- Crawl4AI/Playwright dependencies

## Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd fastapi
```

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
playwright install
```

4. **Set up environment variables**:

Create a `.env` file in the project root with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_INDEX_HOST=your_index_host
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=groq_model
```

## Usage

### Starting the API Server

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

### API Endpoints

#### 1. Build Knowledge Base

Crawl a website and build a knowledge base stored in Pinecone.

**Endpoint**: `POST /build_knowledge_base`

**Request Body**:
```json
{
  "url_to_scrape": "https://example.com",
  "assistant": "company"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Knowledge base built successfully",
  "pages_crawled": 15
}
```

**Features**:
- Recursively crawls internal links up to 4 levels deep
- Filters out navigation, headers, scripts, and styles
- Processes up to 30 pages per crawl
- Automatically cleans and chunks content
- Stores data in Pinecone with assistant-specific filtering

**Example**:
```bash
curl -X POST "http://localhost:8000/build_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "url_to_scrape": "https://example.com",
    "assistant": "company"
  }'
```

#### 2. Get Answer

Query the knowledge base and get an LLM-powered answer.

**Endpoint**: `POST /get_answer`

**Request Body**:
```json
{
  "user_query": "What are your main products?",
  "assistant": "company"
}
```

**Response**:
```json
{
  "response": "Based on the provided context, our main products include..."
}
```

**Features**:
- Semantic search retrieves top 10 most relevant chunks
- Context is automatically limited to 7000 tokens
- Uses Groq's GPT-OSS-120B model for responses
- Responses are restricted to provided context only
- Includes follow-up questions in responses

**Example**:
```bash
curl -X POST "http://localhost:8000/get_answer" \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "What are your main products?",
    "assistant": "company"
  }'
```

## Project Structure

```
fastapi/
├── app.py                 # FastAPI application and endpoints
├── recursive_crawl.py     # Web crawling logic using Crawl4AI
├── pinecone_client.py     # Pinecone operations (upsert, search)
├── semantic_search.py     # LLM-powered answer generation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in repo)
└── README.md            # This file
```

## How It Works

### 1. Knowledge Base Building

1. **Crawling**: Uses Crawl4AI's `BestFirstCrawlingStrategy` to recursively crawl websites
2. **Content Extraction**: Extracts clean markdown content, filtering out navigation and non-content elements
3. **Chunking**: Content is automatically chunked and prepared for vector storage
4. **Vectorization**: Pinecone handles embedding generation and storage using integrated embeddings
5. **Storage**: Chunks are stored in Pinecone with metadata including assistant name and source URL

### 2. Query Processing

1. **Semantic Search**: User query is embedded and searched against Pinecone index
2. **Context Building**: Top 10 relevant chunks are retrieved and combined, limited to 7000 tokens
3. **LLM Generation**: Context and query are sent to Groq's LLM for answer generation
4. **Response**: Streamed response is returned to the user

### Key Components

- **`recursive_crawl.py`**: Handles intelligent web crawling with content filtering
- **`pinecone_client.py`**: Manages Pinecone operations including data validation and token-limited context building
- **`semantic_search.py`**: Orchestrates semantic search and LLM response generation
- **`app.py`**: FastAPI application with REST endpoints and error handling

## Advanced Configuration

### Crawling Strategies

The system currently uses `BestFirstCrawlingStrategy` with browser-based crawling, but Crawl4AI supports various strategies optimized for different scenarios. Here's a guide to choosing the right strategy:

| Scenario | Strategy |
|----------|----------|
| Static website, fast crawl | HTTP + BFS |
| Dynamic site with JS | Browser + Best-First |
| Deep topic exploration | Browser + DFS |
| Broad content collection | HTTP + BFS |
| Keyword-specific search | Statistical Adaptive |
| Concept-based search | Embedding Adaptive |
| Resource-constrained | HTTP + BFS |
| High relevance required | Best-First with scorer |
| Technical documentation | Statistical Adaptive |

**Current Implementation**: The system uses `Browser + Best-First` strategy, which is ideal for dynamic websites with JavaScript and provides good balance between relevance and coverage.

### Crawling Parameters

Modify `recursive_crawl.py` to adjust:
- `max_depth`: Maximum crawl depth (default: 4)
- `max_pages`: Maximum pages to crawl (default: 30)
- `excluded_tags`: HTML tags to exclude from content
- Content filtering thresholds
- Crawling strategy selection

### Token Limits

The context token limit can be adjusted in `pinecone_client.py`:
```python
def build_context(user_query, assistant: Optional[str] = "company", max_tokens: int = 7000):
```

### LLM Configuration

Modify `semantic_search.py` to change:
- Model selection (currently `openai/gpt-oss-120b`)
- Temperature and other generation parameters
- System prompt and response constraints

## Troubleshooting

### Common Issues

1. **Pinecone Connection Errors**: Verify your API key and index configuration in `.env`
2. **Crawling Failures**: Ensure Playwright browsers are installed (`playwright install`)
3. **Empty Responses**: Check that the knowledge base has been built for the specified assistant
4. **Token Limit Errors**: Adjust `max_tokens` parameter if needed

### Logging

The application uses Python's logging module. Set log level in `app.py`:
```python
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more details
```

### Windows Compatibility

The code includes Windows-specific event loop configuration for subprocess support. This is automatically handled in `app.py` and `recursive_crawl.py`.

## API Documentation

Interactive API documentation is available when the server is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
