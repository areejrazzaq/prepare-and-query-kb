import re
import asyncio
import sys
import logging
from concurrent.futures import ThreadPoolExecutor

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from pinecone_client import upsert_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _run_crawler_in_thread(url_to_scrape: str, assistant: str):
    """Run the crawler in a separate thread with its own event loop"""
    if sys.platform == "win32":
        # Create a new event loop with the correct policy for Windows
        policy = asyncio.WindowsProactorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        return loop.run_until_complete(smart_recursive_scrape(url_to_scrape, assistant))
    except Exception as e:
        logger.error(f"Error in crawler thread for {url_to_scrape}: {str(e)}", exc_info=True)
        raise
    finally:
        try:
            loop.close()
        except Exception as e:
            logger.warning(f"Error closing event loop: {str(e)}")

async def smart_recursive_scrape(url_to_scrape: str, assistant: str):
    # 1. Setup intelligent keyword-based crawling
    keyword_scorer = KeywordRelevanceScorer(
        keywords=["products", "services"],  # URLs with these keywords prioritized
        weight=1.0
    )
    
    # 2. Setup content cleaning
    prune_filter = PruningContentFilter(
        threshold=0.5,
        min_word_threshold=10
    )
    
    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={
            "ignore_links": True,
            "body_width": 100
        }
    )
    
    # 3. Configure crawler with both recursive crawling and cleaning
    # Note: url_scorer may not be fully supported with streaming, so we make it optional
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=4,
            include_external=False,
            url_scorer=None,  # Temporarily disabled to avoid "not implemented" error
            max_pages=30  # Limit crawl size
        ),
        markdown_generator=md_generator,
        excluded_tags=["nav", "header", "aside", "script", "style", ],
        exclude_external_links=True,
        stream=True  # Process results as they complete
    )
    
    collected_data = []
    i = 1
    
    try:
        async with AsyncWebCrawler() as crawler:
            # Stream results as they complete
            try:
                async for result in await crawler.arun(
                    url=url_to_scrape,
                    config=config
                ):
                    try:
                        if result.success:
                            # Extract clean content - check if markdown exists
                            clean_text = result.markdown.fit_markdown if result.markdown and result.markdown.fit_markdown else result.markdown.raw_markdown if result.markdown else ""
                            
                            # Validate text: must be non-empty and have meaningful content
                            data = {
                                '_id': assistant + "_" + str(i),
                                'text': clean_text,
                                'assistant': assistant,
                            }
                            collected_data.append(data)
                            # Append to the file 
                            output_file = f"{assistant}_markdown.txt"
                            try:
                                with open(output_file, "a", encoding="utf-8") as f:
                                    f.write("\n\n")
                                    f.write("******************************************************************")
                                    f.write("\nURL: " + result.url)
                                    f.write("\n")
                                    f.write("Content: " + clean_text[:1000])  # Limit content length
                                    f.write("\n")
                                    f.write("******************************************************************")
                                    f.write("\n\n")
                            except Exception as file_error:
                                logger.warning(f"Error writing to file for {result.url}: {str(file_error)}")
                            
                            i += 1
                        else:
                            logger.warning(f"Failed to crawl {result.url}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
                    except Exception as result_error:
                        # Handle individual result processing errors (e.g., page closed)
                        error_msg = str(result_error)
                        if "closed" in error_msg.lower() or "target page" in error_msg.lower():
                            logger.warning(f"Page closed error for URL (continuing): {error_msg}")
                        else:
                            logger.error(f"Error processing result: {error_msg}")
                        continue
            except Exception as crawl_error:
                # Handle crawl-level errors
                error_msg = str(crawl_error)
                if "closed" in error_msg.lower() or "target page" in error_msg.lower():
                    logger.warning(f"Browser/page closed during crawl (partial results may be available): {error_msg}")
                else:
                    logger.error(f"Crawl error: {error_msg}", exc_info=True)
                    raise
    except Exception as crawler_error:
        logger.error(f"Error initializing crawler: {str(crawler_error)}", exc_info=True)
        raise
    
    if not collected_data:
        logger.warning(f"No data collected from {url_to_scrape}")
    
    return collected_data

async def build_knowledge_base(url_to_scrape: str, assistant: str):
    """
    Build knowledge base by crawling URL and storing in Pinecone.
    
    Args:
        url_to_scrape: URL to crawl
        assistant: Assistant identifier for filtering
        
    Returns:
        dict: Result with status and message
    """
    try:
        # Run the crawler in a separate thread with its own event loop
        # This ensures Windows ProactorEventLoop is used for subprocess support
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor,
                _run_crawler_in_thread,
                url_to_scrape,
                assistant
            )
        
        if not results:
            logger.warning(f"No data collected from {url_to_scrape}")
            return {
                "status": "partial",
                "message": "Knowledge base built with partial or no data",
                "pages_crawled": 0
            }
        
        # Upsert the data into Pinecone
        try:
            upsert_result = upsert_data(results)
            logger.info(f"Successfully upserted records to Pinecone: {upsert_result}")
            return {
                "status": "success",
                "message": "Knowledge base built successfully",
                "pages_crawled": len(results),
                "details": upsert_result
            }
        except ValueError as validation_error:
            # Validation errors (e.g., no valid records after filtering)
            logger.error(f"Validation error upserting data to Pinecone: {str(validation_error)}")
            return {
                "status": "partial",
                "message": f"Data crawled but validation failed: {str(validation_error)}",
                "pages_crawled": len(results),
                "error_type": "validation"
            }
        except Exception as upsert_error:
            logger.error(f"Error upserting data to Pinecone: {str(upsert_error)}", exc_info=True)
            error_msg = str(upsert_error)
            # Check for specific Pinecone API errors
            if "INVALID_ARGUMENT" in error_msg or "empty" in error_msg.lower():
                return {
                    "status": "partial",
                    "message": f"Data crawled but failed to upsert due to invalid data: {error_msg}",
                    "pages_crawled": len(results),
                    "error_type": "invalid_data"
                }
            return {
                "status": "partial",
                "message": f"Data crawled but failed to upsert: {error_msg}",
                "pages_crawled": len(results),
                "error_type": "upsert_error"
            }
    except Exception as e:
        logger.error(f"Error building knowledge base for {url_to_scrape}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to build knowledge base: {str(e)}",
            "pages_crawled": 0
        }
   