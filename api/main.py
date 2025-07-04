from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
import asyncio
import aiohttp
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import openai
from pytrends.request import TrendReq
import feedparser
import subprocess
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CorestratAI Brand Watcher API",
    description="Track trending retail brands in India",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_WEIGHT = float(os.getenv("NEWS_WEIGHT", 0.3))
TWITTER_WEIGHT = float(os.getenv("TWITTER_WEIGHT", 0.25))
PR_WEIGHT = float(os.getenv("PR_WEIGHT", 0.2))
TRENDS_WEIGHT = float(os.getenv("TRENDS_WEIGHT", 0.25))
TRENDING_MULTIPLIER = float(os.getenv("TRENDING_MULTIPLIER", 2.0))

# Pydantic models
class BrandRequest(BaseModel):
    brands: Optional[List[str]] = None
    days: Optional[int] = 7

class BrandScore(BaseModel):
    brand: str
    total_score: float
    trending_score: float
    absolute_score: float
    news_count: int
    twitter_count: int
    pr_count: int
    trends_score: float
    growth_rate: float
    summary: str

class HotBrandsResponse(BaseModel):
    brands: List[BrandScore]
    generated_at: str
    period_days: int

class DiscoveryResponse(BaseModel):
    new_brands: List[str]
    total_brands: int
    message: str

# Utility functions
def load_brands() -> List[str]:
    """Load brands from JSON file"""
    try:
        with open("data/brands.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("brands.json not found")
        return []

def save_brands(brands: List[str]):
    """Save brands to JSON file"""
    with open("data/brands.json", "w") as f:
        json.dump(brands, f, indent=2)

async def fetch_news_count(brand: str, days: int = 7) -> int:
    """Fetch news count for a brand using NewsAPI"""
    if not NEWS_API_KEY:
        logger.warning("NewsAPI key not configured")
        return 0
    
    try:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        url = f"https://newsapi.org/v2/everything"
        params = {
            "q": f'"{brand}" AND India',
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": NEWS_API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("totalResults", 0)
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return 0
    except Exception as e:
        logger.error(f"Error fetching news for {brand}: {e}")
        return 0

async def fetch_twitter_count(brand: str, days: int = 7) -> int:
    """Fetch Twitter mentions using snscrape"""
    try:
        since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        cmd = [
            "snscrape", "--jsonl", "--max-results", "1000",
            "twitter-search", f'"{brand}" since:{since_date} lang:en'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            return len([line for line in lines if line.strip()])
        else:
            logger.error(f"snscrape error for {brand}: {result.stderr}")
            return 0
    except Exception as e:
        logger.error(f"Error fetching Twitter data for {brand}: {e}")
        return 0

async def fetch_pr_count(brand: str, days: int = 7) -> int:
    """Fetch PR mentions from RSS feeds"""
    try:
        # Sample PR RSS feeds (you can expand this list)
        pr_feeds = [
            "https://www.business-standard.com/rss/latest.rss",
            "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
            "https://www.livemint.com/rss/companies"
        ]
        
        total_count = 0
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for feed_url in pr_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    entry_date = datetime(*entry.published_parsed[:6])
                    if entry_date >= cutoff_date:
                        if brand.lower() in entry.title.lower() or brand.lower() in entry.summary.lower():
                            total_count += 1
            except Exception as e:
                logger.error(f"Error parsing feed {feed_url}: {e}")
                continue
        
        return total_count
    except Exception as e:
        logger.error(f"Error fetching PR data for {brand}: {e}")
        return 0

async def fetch_trends_score(brand: str) -> float:
    """Fetch Google Trends score"""
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([brand], cat=0, timeframe='now 7-d', geo='IN')
        
        interest_over_time = pytrends.interest_over_time()
        if not interest_over_time.empty:
            return float(interest_over_time[brand].mean())
        return 0.0
    except Exception as e:
        logger.error(f"Error fetching trends for {brand}: {e}")
        return 0.0

async def generate_summary(brand: str, news_count: int, twitter_count: int, 
                          pr_count: int, trends_score: float) -> str:
    """Generate AI summary using OpenAI"""
    try:
        prompt = f"""
        Analyze why "{brand}" is trending this week in India based on these metrics:
        - News mentions: {news_count}
        - Twitter buzz: {twitter_count}
        - PR coverage: {pr_count}
        - Google Trends score: {trends_score:.1f}
        
        Provide a concise 2-3 sentence summary explaining why this brand is hot this week.
        Focus on potential reasons like new launches, campaigns, controversies, or market events.
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating summary for {brand}: {e}")
        return f"High activity detected for {brand} across multiple channels this week."

def calculate_scores(current_metrics: Dict, previous_metrics: Dict = None) -> Dict:
    """Calculate trending and absolute scores"""
    news_count = current_metrics.get('news_count', 0)
    twitter_count = current_metrics.get('twitter_count', 0)
    pr_count = current_metrics.get('pr_count', 0)
    trends_score = current_metrics.get('trends_score', 0)
    
    # Calculate absolute score
    absolute_score = (
        news_count * NEWS_WEIGHT +
        twitter_count * TWITTER_WEIGHT +
        pr_count * PR_WEIGHT +
        trends_score * TRENDS_WEIGHT
    )
    
    # Calculate growth rate if previous data available
    growth_rate = 0.0
    if previous_metrics:
        prev_total = (
            previous_metrics.get('news_count', 1) +
            previous_metrics.get('twitter_count', 1) +
            previous_metrics.get('pr_count', 1)
        )
        current_total = news_count + twitter_count + pr_count
        if prev_total > 0:
            growth_rate = ((current_total - prev_total) / prev_total) * 100
    
    # Calculate trending score (emphasizes growth)
    trending_bonus = max(0, growth_rate / 100) * TRENDING_MULTIPLIER
    trending_score = absolute_score * (1 + trending_bonus)
    
    return {
        'absolute_score': absolute_score,
        'trending_score': trending_score,
        'growth_rate': growth_rate
    }

@app.get("/")
async def root():
    return {"message": "CorestratAI Brand Watcher API", "version": "1.0.0"}

@app.post("/hot-brands", response_model=HotBrandsResponse)
async def get_hot_brands(request: BrandRequest):
    """Get hot brands with trending analysis"""
    try:
        brands = request.brands or load_brands()
        if not brands:
            raise HTTPException(status_code=400, detail="No brands provided or found")
        
        days = request.days or 7
        brand_scores = []
        
        # Process brands concurrently
        tasks = []
        for brand in brands:
            task = process_brand(brand, days)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing brand: {result}")
                continue
            if result:
                brand_scores.append(result)
        
        # Sort by trending score (prioritizes momentum over absolute volume)
        brand_scores.sort(key=lambda x: x.trending_score, reverse=True)
        
        return HotBrandsResponse(
            brands=brand_scores,
            generated_at=datetime.now().isoformat(),
            period_days=days
        )
    
    except Exception as e:
        logger.error(f"Error in get_hot_brands: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_brand(brand: str, days: int) -> Optional[BrandScore]:
    """Process a single brand and return its score"""
    try:
        # Fetch all metrics concurrently
        news_task = fetch_news_count(brand, days)
        twitter_task = fetch_twitter_count(brand, days)
        pr_task = fetch_pr_count(brand, days)
        trends_task = fetch_trends_score(brand)
        
        news_count, twitter_count, pr_count, trends_score = await asyncio.gather(
            news_task, twitter_task, pr_task, trends_task
        )
        
        # Calculate scores
        current_metrics = {
            'news_count': news_count,
            'twitter_count': twitter_count,
            'pr_count': pr_count,
            'trends_score': trends_score
        }
        
        scores = calculate_scores(current_metrics)
        
        # Generate AI summary
        summary = await generate_summary(brand, news_count, twitter_count, pr_count, trends_score)
        
        return BrandScore(
            brand=brand,
            total_score=scores['trending_score'],
            trending_score=scores['trending_score'],
            absolute_score=scores['absolute_score'],
            news_count=news_count,
            twitter_count=twitter_count,
            pr_count=pr_count,
            trends_score=trends_score,
            growth_rate=scores['growth_rate'],
            summary=summary
        )
    
    except Exception as e:
        logger.error(f"Error processing brand {brand}: {e}")
        return None

@app.post("/discover-brands", response_model=DiscoveryResponse)
async def discover_brands():
    """Discover new brands from recent news"""
    try:
        # This would typically run the discovery script
        # For now, we'll return a placeholder response
        current_brands = load_brands()
        
        # In a real implementation, this would:
        # 1. Fetch recent news headlines
        # 2. Extract brand/company names using NLP
        # 3. Filter for Indian retail brands
        # 4. Add new brands to the list
        
        return DiscoveryResponse(
            new_brands=[],
            total_brands=len(current_brands),
            message="Brand discovery feature will be implemented in dynamic_discovery.py"
        )
    
    except Exception as e:
        logger.error(f"Error in discover_brands: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/brands")
async def get_brands():
    """Get current list of tracked brands"""
    return {"brands": load_brands()}

@app.post("/brands")
async def add_brand(brand: str):
    """Add a new brand to track"""
    brands = load_brands()
    if brand not in brands:
        brands.append(brand)
        save_brands(brands)
        return {"message": f"Brand '{brand}' added successfully"}
    return {"message": f"Brand '{brand}' already exists"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
