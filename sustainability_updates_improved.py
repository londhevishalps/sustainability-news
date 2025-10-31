#!/usr/bin/env python3
"""
sustainability_updates_improved.py

Flow:
1) Fetch RSS -> raw_articles.json
2) Extract full text -> filtered_articles.json
3) Cluster -> clustered_articles.json
4) Summarize clusters -> summarized_articles.json

Improvements:
- CPU-friendly summarization for GitHub Actions
- Cluster headlines for frontend UX
- Image fallback from newspaper top image
- Parallel text extraction
- Logging for errors
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

# NLP imports
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline

# ----------------------------
# CONFIG
# ----------------------------
RAW_JSON = "raw_articles.json"
FILTERED_JSON = "filtered_articles.json"
CLUSTERED_JSON = "clustered_articles.json"
SUMMARIES_JSON = "summarized_articles.json"

NUM_CLUSTERS = 5
CUTOFF_DAYS = 7
MAX_WORKERS = 4  # GitHub Actions CPU-friendly

RSS_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

# ----------------------------
# UTIL FUNCTIONS
# ----------------------------
def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_date(date_str: str) -> Optional[datetime]:
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%SZ',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%Y-%m-%d %H:%M:%S',
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.split('(')[0].strip(), fmt)
            return dt.replace(tzinfo=None)
        except ValueError:
            continue
    return None

def extract_image_url(entry, newspaper_article=None) -> Optional[str]:
    """Extract image from RSS entry or fallback to newspaper top_image."""
    # RSS fields
    for field in ['media_content', 'enclosures']:
        if hasattr(entry, field):
            items = getattr(entry, field)
            if items:
                for item in items:
                    if isinstance(item, dict) and 'url' in item and 'image' in item['url'].lower():
                        return item['url']

    # Check summary/content for <img>
    for field in ['summary', 'description', 'content']:
        if hasattr(entry, field):
            content = getattr(entry, field)
            if isinstance(content, list):
                content = content[0].get('value', '') if content else ''
            match = re.search(r'<img[^>]+src=["\'](.*?)["\']', content, re.IGNORECASE)
            if match:
                return match.group(1)

    # Fallback to newspaper top image
    if newspaper_article and newspaper_article.top_image:
        return newspaper_article.top_image

    return None

# ----------------------------
# STEP 1: FETCH RSS
# ----------------------------
def fetch_rss(feeds: List[str]) -> List[Dict]:
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=CUTOFF_DAYS)

    for url in tqdm(feeds, desc="Fetching RSS Feeds"):
        try:
            feed = feedparser.parse(url)
            feed_title = getattr(feed.feed, 'title', url)
            for entry in feed.entries:
                try:
                    pub_date_str = getattr(entry, 'published', None)
                    pub_date = None
                    if pub_date_str:
                        pub_date = parse_date(pub_date_str)
                    elif getattr(entry, 'published_parsed', None):
                        pub_date_struct = entry.published_parsed
                        pub_date = datetime(*pub_date_struct[:6])

                    if not pub_date or pub_date < cutoff_date:
                        continue

                    summary_text = getattr(entry, 'summary', '')
                    if not summary_text and getattr(entry, 'content', None):
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
                    print(f"⚠️ Skipped entry: {e}")
                    continue
        except Exception as e:
            print(f"⚠️ Failed feed {url}: {e}")
            continue
    return all_articles

# ----------------------------
# STEP 2: EXTRACT FULL TEXT (Parallel)
# ----------------------------
def _process_article_text(article: Dict) -> Optional[Dict]:
    try:
        if not article["text"] or len(article["text"]) < 100:
            a = newspaper.Article(article["url"])
            a.download()
            a.parse()
            article["text"] = a.text
            article["image_url"] = extract_image_url(article, a)
    except Exception as e:
        print(f"⚠️ Text extraction failed for {article['url']}: {e}")
    if article["text"] and len(article["text"]) > 100:
        return article
    return None

def extract_full_text_parallel(articles: List[Dict]) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(_process_article_text, articles),
            total=len(articles),
            desc="Extracting Full Text"
        ))
    return [r for r in results if r is not None]

# ----------------------------
# STEP 3: CLUSTER
# ----------------------------
def cluster_articles(articles: List[Dict], num_clusters: int, model: SentenceTransformer) -> List[Dict]:
    if len(articles) < num_clusters:
        for article in articles:
            article["cluster_id"] = 0
        return articles

    texts = [f"{a['title']} {a['text'][:500]}" for a in articles]
    embeddings = model.encode(texts, show_progress_bar=False)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    kmeans.fit(embeddings)
    for i, label in enumerate(kmeans.labels_):
        articles[i]["cluster_id"] = int(label)
    return articles

# ----------------------------
# STEP 4: SUMMARIZE
# ----------------------------
def summarize_clusters(articles: List[Dict], summarizer_pipeline) -> List[Dict]:
    clusters = {}
    for article in articles:
        cid = article["cluster_id"]
        clusters.setdefault(cid, {"articles": [], "headline": "", "summary": "", "image_url": None})
        clusters[cid]["articles"].append(article)

    for cid, cluster in clusters.items():
        cluster["articles"].sort(key=lambda x: x["date"], reverse=True)
        most_recent = cluster["articles"][0]
        cluster["headline"] = most_recent["title"]
        cluster["image_url"] = most_recent["image_url"]

    MAX_SUMMARY_CHARS = 10000
    for cid, cluster in tqdm(clusters.items(), desc="Summarizing Clusters"):
        combined_text = " ".join([a["text"] for a in cluster["articles"][:3]])
        combined_text = combined_text[:MAX_SUMMARY_CHARS]
        try:
            summary = summarizer_pipeline(
                combined_text,
                max_length=150,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
            cluster["summary"] = summary
        except Exception as e:
            print(f"⚠️ Summary failed for cluster {cid}: {e}")
            cluster["summary"] = "No summary available."

    return [clusters[k] for k in sorted(clusters.keys())]

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("--- Loading NLP Models ---")
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer_pipeline = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1
        )
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    print("--- Step 1: Fetch RSS ---")
    articles = fetch_rss(RSS_FEEDS)
    save_json(RAW_JSON, articles)
    print(f"Fetched {len(articles)} raw articles.")

    print("--- Step 2: Extract Full Text ---")
    articles = extract_full_text_parallel(articles)
    save_json(FILTERED_JSON, articles)
    print(f"{len(articles)} articles with substantial text.")

    if not articles:
        save_json(CLUSTERED_JSON, [])
        print("No articles to process. Exiting.")
        return

    print("--- Step 3: Cluster Articles ---")
    articles = cluster_articles(articles, NUM_CLUSTERS, sbert_model)

    print("--- Step 4: Summarize Clusters ---")
    final_data = summarize_clusters(articles, summarizer_pipeline)
    save_json(SUMMARIES_JSON, final_data)
    save_json(CLUSTERED_JSON, final_data)
    print(f"Processed into {len(final_data)} clusters.")

if __name__ == "__main__":
    main()
