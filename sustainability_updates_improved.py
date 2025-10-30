#!/usr/bin/env python3
"""
sustainability_updates_improved.py

Flow:
1) fetch RSS -> raw_articles.json
2) extract full text -> filtered_articles.json
3) cluster -> clustered_articles.json
4) summarize clusters -> summarized_articles.json

Improvements:
- Robust date parsing for RSS feeds.
- More robust image URL extraction.
- Use of concurrent futures for parallel text extraction to speed up I/O-bound tasks.
- Improved logging/error handling.
- Centralized model loading to avoid re-initialization.
- Cleaned up the main function for better flow.
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import feedparser
import newspaper
from tqdm import tqdm

# NLP imports (heavy)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline

# #############################################################################
# CONFIG
# #############################################################################

RAW_JSON = "raw_articles.json"
FILTERED_JSON = "filtered_articles.json"
CLUSTERED_JSON = "clustered_articles.json"
SUMMARIES_JSON = "summarized_articles.json"

NUM_CLUSTERS = 5
CUTOFF_DAYS = 7
MAX_WORKERS = 8 # For parallel text extraction

RSS_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

# #############################################################################
# UTIL
# #############################################################################

def save_json(path: str, obj):
    """Save an object to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str, default=None):
    """Load an object from a JSON file."""
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_date(date_str: str) -> Optional[datetime]:
    """
    Attempts to parse a date string using common RSS formats.
    """
    # Common RSS date formats (RFC 822, ISO 8601, etc.)
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',  # RFC 822 (e.g., Fri, 30 Oct 2025 19:30:00 +0000)
        '%Y-%m-%dT%H:%M:%S%z',      # ISO 8601 with timezone (e.g., 2025-10-30T19:30:00+00:00)
        '%Y-%m-%dT%H:%M:%SZ',      # ISO 8601 UTC (e.g., 2025-10-30T19:30:00Z)
        '%a, %d %b %Y %H:%M:%S %Z', # RFC 822 without numeric offset
        '%Y-%m-%d %H:%M:%S',        # Simple format
    ]
    for fmt in formats:
        try:
            # feedparser often returns a time.struct_time object which is handled by
            # entry.published_parsed, but if it's a string, we need to parse it.
            dt = datetime.strptime(date_str.split('(')[0].strip(), fmt)
            # Remove timezone info for comparison with cutoff_date
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    return None

def extract_image_url(entry) -> Optional[str]:
    """
    Extracts a suitable image URL from various feedparser fields.
    """
    # 1. media_content (most reliable for images)
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if 'url' in media and media.get('type', '').startswith('image'):
                return media['url']
            if 'url' in media and 'image' in media['url'].lower():
                return media['url']
    
    # 2. enclosure
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get('type', '').startswith('image'):
                return enc['href']

    # 3. summary/description (regex for first image tag)
    for field in ['summary', 'description', 'content']:
        if hasattr(entry, field):
            content = getattr(entry, field)
            if isinstance(content, list):
                content = content[0].get('value', '') if content else ''
            
            match = re.search(r'<img[^>]+src=["\'](.*?)["\']', content, re.IGNORECASE)
            if match:
                return match.group(1)
    
    return None


# #############################################################################
# STEP 1: FETCH RSS
# #############################################################################

def fetch_rss(feeds: List[str]) -> List[Dict]:
    """Fetches and parses articles from a list of RSS feeds."""
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=CUTOFF_DAYS)

    for url in tqdm(feeds, desc="Fetching RSS Feeds"):
        try:
            feed = feedparser.parse(url)
            feed_title = feed.feed.title if hasattr(feed.feed, 'title') else url
            
            for entry in feed.entries:
                try:
                    # Robust date parsing
                    pub_date_str = getattr(entry, 'published', None)
                    if not pub_date_str:
                        # Fallback to parsed date if string is missing
                        pub_date_struct = getattr(entry, 'published_parsed', None)
                        if pub_date_struct:
                            pub_date = datetime(*pub_date_struct[:6])
                        else:
                            # Skip if no date info is found
                            continue
                    else:
                        pub_date = parse_date(pub_date_str)

                    if not pub_date or pub_date < cutoff_date:
                        continue

                    # Fallback to entry.summary if entry.text is not available
                    summary_text = getattr(entry, 'summary', '')
                    if not summary_text and hasattr(entry, 'content') and entry.content:
                        summary_text = entry.content[0].get('value', '')

                    article = {
                        "title": entry.title,
                        "url": entry.link,
                        "date": pub_date.strftime("%Y-%m-%d"),
                        "source": feed_title,
                        "text": summary_text,
                        "image_url": extract_image_url(entry),
                        "cluster_id": -1,
                    }
                    all_articles.append(article)
                except Exception as e:
                    print(f"Warning: Error processing entry from {url}: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching feed {url}: {e}")
            continue

    return all_articles

# #############################################################################
# STEP 2: EXTRACT FULL TEXT (Parallelized)
# #############################################################################

def _process_article_text(article: Dict) -> Optional[Dict]:
    """Helper function to extract full text for a single article."""
    # Only try to extract if the summary is short or non-existent
    if not article["text"] or len(article["text"]) < 100:
        try:
            a = newspaper.Article(article["url"])
            a.download()
            a.parse()
            article["text"] = a.text
        except Exception:
            # Fail silently on text extraction errors
            pass
    
    # Final filter: only keep articles with substantial text
    if article["text"] and len(article["text"]) > 200:
        return article
    return None

def extract_full_text_parallel(articles: List[Dict]) -> List[Dict]:
    """Extracts full text content in parallel using a ThreadPoolExecutor."""
    filtered_articles = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map the processing function to all articles
        results = list(tqdm(
            executor.map(_process_article_text, articles),
            total=len(articles),
            desc="Extracting Full Text (Parallel)"
        ))

    # Filter out None results (articles that failed the text length check)
    filtered_articles = [res for res in results if res is not None]
    return filtered_articles

# #############################################################################
# STEP 3: CLUSTER
# #############################################################################

def cluster_articles(articles: List[Dict], num_clusters: int, model: SentenceTransformer) -> List[Dict]:
    """Clusters articles based on their text content."""
    
    if len(articles) < num_clusters:
        for article in articles:
            article["cluster_id"] = 0
        return articles

    # Use titles and first 500 characters of text for embedding
    texts = [f"{a['title']} {a['text'][:500]}" for a in articles]

    # Perform embedding
    embeddings = model.encode(texts, show_progress_bar=False)

    # Perform KMeans clustering
    # n_init='auto' is the modern default for scikit-learn
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto') 
    kmeans.fit(embeddings)
    
    for i, label in enumerate(kmeans.labels_):
        articles[i]["cluster_id"] = int(label)

    return articles

# #############################################################################
# STEP 4: SUMMARIZE
# #############################################################################

def summarize_clusters(articles: List[Dict], summarizer_pipeline) -> List[Dict]:
    """Summarizes each cluster and prepares the final data structure."""
    
    # Group articles by cluster_id
    clusters = {}
    for article in articles:
        cluster_id = article["cluster_id"]
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "articles": [],
                "source_link": "",
                "summary": "",
                "image_url": None
            }
        clusters[cluster_id]["articles"].append(article)

    # Sort articles within each cluster by date (most recent first)
    for cluster_id in clusters:
        clusters[cluster_id]["articles"].sort(key=lambda x: x["date"], reverse=True)
        
        # Select source and image from the most recent article
        if clusters[cluster_id]["articles"]:
            most_recent_article = clusters[cluster_id]["articles"][0]
            clusters[cluster_id]["source_link"] = most_recent_article["source"]
            clusters[cluster_id]["image_url"] = most_recent_article["image_url"]
            
    # Summarize each cluster
    MAX_SUMMARY_INPUT = 10000 # A safe character limit before tokenization
    
    for cluster_id, cluster_data in tqdm(clusters.items(), desc="Summarizing Clusters"):
        # Concatenate the text of the top 3 articles for summarization
        combined_text = " ".join([a["text"] for a in cluster_data["articles"][:3]])
        
        # Truncate to a safe length for the summarizer model
        if len(combined_text) > MAX_SUMMARY_INPUT:
            combined_text = combined_text[:MAX_SUMMARY_INPUT]

        summary_text = "No summary available for this cluster."
        try:
            # Generate summary
            summary = summarizer_pipeline(
                combined_text, 
                max_length=150, 
                min_length=30, 
                do_sample=False
            )[0]["summary_text"]
            summary_text = summary
        except Exception as e:
            print(f"Warning: Error summarizing cluster {cluster_id}: {e}")
            
        cluster_data["summary"] = summary_text

    # Convert dictionary to a list and return, sorted by cluster_id
    final_data = [clusters[key] for key in sorted(clusters.keys())]
    return final_data

# #############################################################################
# MAIN
# #############################################################################

def main():
    """Main function to run the entire pipeline."""
    
    print("--- Initializing NLP Models ---")
    # Initialize heavy models once
    try:
        # Clustering Model
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Summarization Pipeline
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"FATAL: Could not load NLP models. Please check your environment and dependencies. Error: {e}")
        return

    print("--- 1. Fetching RSS Articles ---")
    articles = fetch_rss(RSS_FEEDS)
    save_json(RAW_JSON, articles)
    print(f"Fetched {len(articles)} raw articles.")
    
    print("--- 2. Extracting Full Text (Parallel) ---")
    # Using the parallelized function
    articles = extract_full_text_parallel(articles)
    save_json(FILTERED_JSON, articles)
    print(f"Filtered to {len(articles)} articles with substantial text.")
    
    if not articles:
        print("No articles to process. Exiting.")
        return

    print(f"--- 3. Clustering Articles into {NUM_CLUSTERS} Clusters ---")
    articles = cluster_articles(articles, NUM_CLUSTERS, sbert_model)
    # Save the clustered articles (optional, for debugging)
    # save_json(CLUSTERED_JSON, articles)
    
    print("--- 4. Summarizing Clusters ---")
    final_data = summarize_clusters(articles, summarizer_pipeline)
    save_json(SUMMARIES_JSON, final_data)
    
    # Write the final data to the file the frontend expects
    save_json(CLUSTERED_JSON, final_data)
    
    print(f"Successfully processed articles into {len(final_data)} clusters.")

if __name__ == "__main__":
    main()
