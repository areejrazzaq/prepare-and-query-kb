from pinecone import Pinecone
from typing import Optional, List, Dict, Any
import logging
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize a Pinecone client with your API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index_host = os.getenv("PINECONE_INDEX_HOST")

logger = logging.getLogger(__name__)



def upsert_data(records: List[Dict[str, Any]]):
    """
    Upsert records to Pinecone after validating and filtering empty text fields.
    
    Args:
        records: List of records to upsert. Each record should have '_id', 'text', and 'assistant' fields.
        
    Returns:
        str: Success message with count of records upserted
        
    Raises:
        ValueError: If no valid records after filtering
        Exception: If Pinecone API call fails
    """
    if not records:
        raise ValueError("No records provided for upsert")
    
    # Filter out records with empty or invalid text fields
    valid_records = []
    filtered_count = 0
    
    for record in records:
        # Check if text field exists and is not empty
        text = record.get('text', '')
        
        # Validate text: must be non-empty string with actual content (not just whitespace)
        if not text or not isinstance(text, str):
            filtered_count += 1
            logger.warning(f"Skipping record {record.get('_id', 'unknown')}: empty or invalid text field")
            continue
        
        # Strip whitespace and check if there's actual content
        text_stripped = text.strip()
        if not text_stripped or len(text_stripped) < 3:  # Minimum 3 characters
            filtered_count += 1
            logger.warning(f"Skipping record {record.get('_id', 'unknown')}: text too short or empty after stripping")
            continue
        
        # Ensure required fields exist
        if not record.get('_id'):
            filtered_count += 1
            logger.warning(f"Skipping record: missing '_id' field")
            continue
        
        if not record.get('assistant'):
            filtered_count += 1
            logger.warning(f"Skipping record {record.get('_id', 'unknown')}: missing 'assistant' field")
            continue
        
        # Use stripped text
        record['text'] = text_stripped
        valid_records.append(record)
    
    if not valid_records:
        error_msg = f"No valid records after filtering. Filtered out {filtered_count} invalid records."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} invalid records. Upserting {len(valid_records)} valid records.")
    
    # Target the index
    dense_index = pc.Index(index_name)

    # Upsert the valid records into a namespace
    try:
        dense_index.upsert_records("default", valid_records)
        logger.info(f"Successfully upserted {len(valid_records)} records to Pinecone")
        return f"Data upserted into Pinecone: {len(valid_records)} records"
    except Exception as e:
        logger.error(f"Error upserting records to Pinecone: {str(e)}", exc_info=True)
        raise


def build_context(user_query, assistant: Optional[str] = "company", max_tokens: int = 7000):
    index = pc.Index(host=index_host)

    results = index.search(
        namespace="default", 
        query={
            "inputs": {"text": user_query}, 
            "top_k": 10,
            "filter": {"assistant": assistant}
        },
        fields=["assistant", "text"]
    )
    context = ""
    if len(results['result']['hits']) == 0:
        return None
    
    # Initialize tokenizer for counting tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    current_tokens = 0
    
    for hit in results['result']['hits']:
        text_to_add = " " + hit['fields']['text']
        # Count tokens for the text we're about to add
        tokens_to_add = len(encoding.encode(text_to_add))
        
        # Check if adding this text would exceed the limit
        if current_tokens + tokens_to_add > max_tokens:
            # Try to add a partial chunk if we have space
            remaining_tokens = max_tokens - current_tokens
            if remaining_tokens > 0:
                # Encode and decode to get a partial chunk that fits
                encoded_text = encoding.encode(hit['fields']['text'])
                if len(encoded_text) > 0:
                    # Take as many tokens as we can fit
                    partial_encoded = encoded_text[:remaining_tokens]
                    partial_text = encoding.decode(partial_encoded)
                    context += " " + partial_text
            break
        
        context += text_to_add
        current_tokens += tokens_to_add

    return context










# Create a dense index with integrated embedding
# index_name = "aviro"
# if not pc.has_index(index_name):
#     pc.create_index_for_model(
#         name=index_name,
#         cloud="aws",
#         region="us-east-1",
#         embed={
#             "model":"llama-text-embed-v2",
#             "field_map":{"text": "chunk_text"}
#         }
#     )

# records = [
#     { "_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history" },
#     { "_id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science" },
#     { "_id": "rec3", "chunk_text": "Albert Einstein developed the theory of relativity.", "category": "science" },
#     { "_id": "rec4", "chunk_text": "The mitochondrion is often called the powerhouse of the cell.", "category": "biology" },
#     { "_id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature" },
#     { "_id": "rec6", "chunk_text": "Water boils at 100°C under standard atmospheric pressure.", "category": "physics" },
#     { "_id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history" },
#     { "_id": "rec8", "chunk_text": "Honey never spoils due to its low moisture content and acidity.", "category": "food science" },
#     { "_id": "rec9", "chunk_text": "The speed of light in a vacuum is approximately 299,792 km/s.", "category": "physics" },
#     { "_id": "rec10", "chunk_text": "Newton's laws describe the motion of objects.", "category": "physics" },
#     { "_id": "rec11", "chunk_text": "The human brain has approximately 86 billion neurons.", "category": "biology" },
#     { "_id": "rec12", "chunk_text": "The Amazon Rainforest is one of the most biodiverse places on Earth.", "category": "geography" },
#     { "_id": "rec13", "chunk_text": "Black holes have gravitational fields so strong that not even light can escape.", "category": "astronomy" },
#     { "_id": "rec14", "chunk_text": "The periodic table organizes elements based on their atomic number.", "category": "chemistry" },
#     { "_id": "rec15", "chunk_text": "Leonardo da Vinci painted the Mona Lisa.", "category": "art" },
#     { "_id": "rec16", "chunk_text": "The internet revolutionized communication and information sharing.", "category": "technology" },
#     { "_id": "rec17", "chunk_text": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history" },
#     { "_id": "rec18", "chunk_text": "Dogs have an incredible sense of smell, much stronger than humans.", "category": "biology" },
#     { "_id": "rec19", "chunk_text": "The Pacific Ocean is the largest and deepest ocean on Earth.", "category": "geography" },
#     { "_id": "rec20", "chunk_text": "Chess is a strategic game that originated in India.", "category": "games" },
#     { "_id": "rec21", "chunk_text": "The Statue of Liberty was a gift from France to the United States.", "category": "history" },
#     { "_id": "rec22", "chunk_text": "Coffee contains caffeine, a natural stimulant.", "category": "food science" },
#     { "_id": "rec23", "chunk_text": "Thomas Edison invented the practical electric light bulb.", "category": "inventions" },
#     { "_id": "rec24", "chunk_text": "The moon influences ocean tides due to gravitational pull.", "category": "astronomy" },
#     { "_id": "rec25", "chunk_text": "DNA carries genetic information for all living organisms.", "category": "biology" },
#     { "_id": "rec26", "chunk_text": "Rome was once the center of a vast empire.", "category": "history" },
#     { "_id": "rec27", "chunk_text": "The Wright brothers pioneered human flight in 1903.", "category": "inventions" },
#     { "_id": "rec28", "chunk_text": "Bananas are a good source of potassium.", "category": "nutrition" },
#     { "_id": "rec29", "chunk_text": "The stock market fluctuates based on supply and demand.", "category": "economics" },
#     { "_id": "rec30", "chunk_text": "A compass needle points toward the magnetic north pole.", "category": "navigation" },
#     { "_id": "rec31", "chunk_text": "The universe is expanding, according to the Big Bang theory.", "category": "astronomy" },
#     { "_id": "rec32", "chunk_text": "Elephants have excellent memory and strong social bonds.", "category": "biology" },
#     { "_id": "rec33", "chunk_text": "The violin is a string instrument commonly used in orchestras.", "category": "music" },
#     { "_id": "rec34", "chunk_text": "The heart pumps blood throughout the human body.", "category": "biology" },
#     { "_id": "rec35", "chunk_text": "Ice cream melts when exposed to heat.", "category": "food science" },
#     { "_id": "rec36", "chunk_text": "Solar panels convert sunlight into electricity.", "category": "technology" },
#     { "_id": "rec37", "chunk_text": "The French Revolution began in 1789.", "category": "history" },
#     { "_id": "rec38", "chunk_text": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history" },
#     { "_id": "rec39", "chunk_text": "Rainbows are caused by light refracting through water droplets.", "category": "physics" },
#     { "_id": "rec40", "chunk_text": "Mount Everest is the tallest mountain in the world.", "category": "geography" },
#     { "_id": "rec41", "chunk_text": "Octopuses are highly intelligent marine creatures.", "category": "biology" },
#     { "_id": "rec42", "chunk_text": "The speed of sound is around 343 meters per second in air.", "category": "physics" },
#     { "_id": "rec43", "chunk_text": "Gravity keeps planets in orbit around the sun.", "category": "astronomy" },
#     { "_id": "rec44", "chunk_text": "The Mediterranean diet is considered one of the healthiest in the world.", "category": "nutrition" },
#     { "_id": "rec45", "chunk_text": "A haiku is a traditional Japanese poem with a 5-7-5 syllable structure.", "category": "literature" },
#     { "_id": "rec46", "chunk_text": "The human body is made up of about 60% water.", "category": "biology" },
#     { "_id": "rec47", "chunk_text": "The Industrial Revolution transformed manufacturing and transportation.", "category": "history" },
#     { "_id": "rec48", "chunk_text": "Vincent van Gogh painted Starry Night.", "category": "art" },
#     { "_id": "rec49", "chunk_text": "Airplanes fly due to the principles of lift and aerodynamics.", "category": "physics" },
#     { "_id": "rec50", "chunk_text": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy" }
# ]

# # Target the index
# dense_index = pc.Index(index_name)

# # Upsert the records into a namespace
# dense_index.upsert_records("default", records)


# Wait for the upserted vectors to be indexed
# import time
# time.sleep(10)

# View stats for the index
# stats = dense_index.describe_index_stats()
# print(stats)




