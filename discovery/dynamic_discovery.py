#!/usr/bin/env python3
"""
Dynamic Brand Discovery Script
Automatically discovers new retail brands from news headlines
"""

import os
import json
import requests
import spacy
from datetime import datetime, timedelta
from typing import List, Set
import logging
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BRANDS_FILE = "data/brands.json"

# Initialize OpenAI
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Load spaCy model (install with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Indian retail/brand keywords for filtering
RETAIL_KEYWORDS = {
    'fashion', 'clothing', 'apparel', 'retail', 'store', 'shop', 'mall', 
    'brand', 'ecommerce', 'online', 'marketplace', 'beauty', 'cosmetics',
    'jewelry', 'jewellery', 'footwear', 'shoes', 'accessories', 'lifestyle',
    'home', 'furniture', 'electronics', 'mobile', 'smartphone', 'gadget',
    'food', 'restaurant', 'cafe', 'delivery', 'grocery', 'supermarket'
}

# Common words to exclude
EXCLUDE_WORDS = {
    'india', 'indian', 'new', 'delhi', 'mumbai', 'bangalore', 'chennai',
    'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'government', 'ministry',
    'company', 'limited', 'ltd', 'pvt', 'private', 'public', 'inc',
    'corporation', 'corp', 'group', 'holdings', 'enterprises'
}

def load_existing_brands() -> Set[str]:
    """Load existing brands from JSON file"""
    try:
        with open(BRANDS_FILE, 'r') as f:
            brands = json.load(f)
            return set(brand.lower() for brand in brands)
    except FileNotFoundError:
        logger.warning(f"{BRANDS_FILE} not found, starting with empty list")
        return set()

def save_brands(brands: List[str]):
    """Save brands to JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(BRANDS_FILE), exist_ok=True)
        
        with open(BRANDS_FILE, 'w') as f:
            json.dump(sorted(brands), f, indent=2)
        logger.info(f"Saved {len(brands)} brands to {BRANDS_FILE}")
    except Exception as e:
        logger.error(f"Error saving brands: {e}")

def fetch_recent_news(days: int = 7) -> List[str]:
    """Fetch recent news headlines from NewsAPI"""
    if not NEWS_API_KEY:
        logger.error("NewsAPI key not configured")
        return []
    
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        
        # Search for retail/business related news in India
        queries = [
            "retail India",
            "brand launch India", 
            "ecommerce India",
            "fashion India",
            "startup India"
        ]
        
        all_headlines = []
        
        for query in queries:
            params = {
                "q": query,
                "from": from_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": NEWS_API_KEY
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                headlines = [article["title"] for article in articles if article["title"]]
                all_headlines.extend(headlines)
                logger.info(f"Fetched {len(headlines)} headlines for query: {query}")
            else:
                logger.error(f"NewsAPI error for query '{query}': {response.status_code}")
        
        return all_headlines
    
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

def extract_entities_spacy(headlines: List[str]) -> Set[str]:
    """Extract organization entities using spaCy"""
    if not nlp:
        return set()
    
    entities = set()
    
    for headline in headlines:
        try:
            doc = nlp(headline)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"]:
                    entity = ent.text.strip()
                    if is_valid_brand_name(entity):
                        entities.add(entity)
        except Exception as e:
            logger.error(f"Error processing headline '{headline}': {e}")
            continue
    
    return entities

def extract_entities_gpt(headlines: List[str]) -> Set[str]:
    """Extract brand names using GPT"""
    if not OPENAI_API_KEY:
        return set()
    
    try:
        # Process headlines in batches
        batch_size = 10
        all_entities = set()
        
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i+batch_size]
            headlines_text = "\n".join(f"{j+1}. {headline}" for j, headline in enumerate(batch))
            
            prompt = f"""
            Extract Indian retail brand names, company names, and product names from these news headlines.
            Focus on:
            - Fashion/clothing brands
            - E-commerce platforms
            - Retail chains
            - Beauty/cosmetics brands
            - Food/restaurant brands
            - Technology/gadget brands
            
            Headlines:
            {headlines_text}
            
            Return only the brand/company names, one per line, without numbers or explanations.
            Exclude generic terms like "India", "Indian", "Company", "Limited", etc.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            entities = [line.strip() for line in content.split('\n') if line.strip()]
            
            for entity in entities:
                if is_valid_brand_name(entity):
                    all_entities.add(entity)
        
        return all_entities
    
    except Exception as e:
        logger.error(f"Error extracting entities with GPT: {e}")
        return set()

def is_valid_brand_name(name: str) -> bool:
    """Check if a name is a valid brand name"""
    name_lower = name.lower()
    
    # Basic filters
    if len(name) < 2 or len(name) > 50:
        return False
    
    # Exclude common words
    if name_lower in EXCLUDE_WORDS:
        return False
    
    # Exclude if it's just numbers or special characters
    if not any(c.isalpha() for c in name):
        return False
    
    # Exclude if it contains too many common words
    words = name_lower.split()
    common_word_count = sum(1 for word in words if word in EXCLUDE_WORDS)
    if common_word_count > len(words) // 2:
        return False
    
    return True

def filter_retail_brands(entities: Set[str], headlines: List[str]) -> Set[str]:
    """Filter entities to keep only retail-related brands"""
    retail_brands = set()
    headlines_text = " ".join(headlines).lower()
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Check if entity appears in context with retail keywords
        context_score = 0
        for keyword in RETAIL_KEYWORDS:
            if keyword in headlines_text and entity_lower in headlines_text:
                # Check if they appear near each other
                context_score += 1
        
        # If entity has retail context or is a known pattern, include it
        if context_score > 0 or any(keyword in entity_lower for keyword in ['store', 'shop', 'mart', 'bazaar']):
            retail_brands.add(entity)
    
    return retail_brands

def discover_brands(days: int = 7) -> List[str]:
    """Main brand discovery function"""
    logger.info(f"Starting brand discovery for last {days} days")
    
    # Load existing brands
    existing_brands = load_existing_brands()
    logger.info(f"Loaded {len(existing_brands)} existing brands")
    
    # Fetch recent news
    headlines = fetch_recent_news(days)
    if not headlines:
        logger.warning("No headlines fetched")
        return []
    
    logger.info(f"Fetched {len(headlines)} headlines")
    
    # Extract entities using both methods
    spacy_entities = extract_entities_spacy(headlines)
    gpt_entities = extract_entities_gpt(headlines)
    
    logger.info(f"SpaCy extracted {len(spacy_entities)} entities")
    logger.info(f"GPT extracted {len(gpt_entities)} entities")
    
    # Combine entities
    all_entities = spacy_entities.union(gpt_entities)
    
    # Filter for retail brands
    retail_entities = filter_retail_brands(all_entities, headlines)
    logger.info(f"Filtered to {len(retail_entities)} retail entities")
    
    # Find new brands
    new_brands = []
    for entity in retail_entities:
        if entity.lower() not in existing_brands:
            new_brands.append(entity)
    
    logger.info(f"Discovered {len(new_brands)} new brands: {new_brands}")
    
    return new_brands

def main():
    """Main function"""
    try:
        # Discover new brands
        new_brands = discover_brands()
        
        if new_brands:
            # Load existing brands (original case)
            try:
                with open(BRANDS_FILE, 'r') as f:
                    existing_brands = json.load(f)
            except FileNotFoundError:
                existing_brands = []
            
            # Add new brands
            all_brands = existing_brands + new_brands
            
            # Remove duplicates while preserving order
            seen = set()
            unique_brands = []
            for brand in all_brands:
                if brand.lower() not in seen:
                    unique_brands.append(brand)
                    seen.add(brand.lower())
            
            # Save updated list
            save_brands(unique_brands)
            
            logger.info(f"Discovery complete. Added {len(new_brands)} new brands.")
            logger.info(f"Total brands: {len(unique_brands)}")
            
            # Print new brands
            print("New brands discovered:")
            for brand in new_brands:
                print(f"  - {brand}")
        else:
            logger.info("No new brands discovered.")
    
    except Exception as e:
        logger.error(f"Error in main discovery process: {e}")

if __name__ == "__main__":
    main()
