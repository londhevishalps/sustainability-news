#!/usr/bin/env python3
"""
sustainability_updates_improved.py

Generates:
- raw_articles.json
- filtered_articles.json
- clustered_articles.json
- summarized_articles.json

Includes images and sources for frontend display.
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

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline

# ---------------- CONFIG ----------------
RAW_JSON = "raw_articles.json"
FILTERED_JSON = "filtered_articles.json"
CLUSTERED_JSON = "clustered_articles.json"
SUMMARIES_JSON = "summarized_articles.json"

NUM_CLUSTERS = 5
CUTOFF_DAYS = 7
MAX_WORKERS = 8

RSS_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

# ---------------- UTIL ----------------
def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

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

def extract_image_url(entry) -> Optional[str]:
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if 'url' in media and media.get('type', '').startswith('image'):
                return media['url']
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get('type', '').startswith('image'):
                return enc['href']
    for field in ['summary', 'description', 'content']:
        if hasattr(entry, field):
            content = getattr(entry, field)
            if isinstance(content, list):
                content = content[0].get('value', '') if content else ''
            match = re.search(r'<img[^>]+src=["\'](.*?)["\']', content, re.IGNORECASE)
            if match:
                return match.group(1)
    return None

# ---------------- STEP 1: FETCH RSS ----------------
def fetch_rss(feeds: List[str]) -> List[Dict]:
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=CUTOFF_DAYS)
    for url in tqdm(feeds, desc="Fetching RSS Feeds"):
        try:
            feed = feedparser.parse(url)
            feed_title = getattr(feed.feed, 'title', url)
            for entry in feed.entries:
                pub_date_str = getattr(entry, 'published', None)
                if pub_date_str:
                    pub_date = parse_date(pub_date_str)
                else:
                    pub_date_struct = getattr(entry, 'published_parsed', None)
                    pub_date = datetime(*pub_date_struct[:6]) if pub_date_struct else None
                if not pub_date or pub_date < cutoff_date:
                    continue
                summary_text = getattr(entry, 'summary', '')
                if not summary_text and hasattr(entry, 'content') and entry.content:
                    summary_text = entry.content[0].get('value', '')
                article = {
                    "title": getattr(entry, 'title', 'No title'),
                    "url": getattr(entry, 'link', ''),
                    "date": pub_date.strftime("%Y-%m-%d"),
                    "source": feed_title,
                    "text": summary_text,
                    "image_url": extract_image_url(entry),
                    "cluster_id": -1,
                }
                all_articles.append(article)
        except Exception:
            continue
    return all_articles

# ---------------- STEP 2: EXTRACT FULL TEXT ----------------
def _process_article_text(article: Dict) -> Optional[Dict]:
    if not article["text"] or len(article["text"]) < 100:
        try:
            a = newspaper.Article(article["url"])
            a.download()
            a.parse()
            article["text"] = a.text
            if not article["image_url"]:
                article["image_url"] = a.top_image or None
        except Exception:
            pass
    if article["text"] and len(article["text"]) > 200:
        return article
    return None

def extract_full_text_parallel(articles: List[Dict]) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(_process_article_text, articles),
            total=len(articles),
            desc="Extracting Full Text"
        ))
    return [res for res in results if res]

# ---------------- STEP 3: CLUSTER ----------------
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

# ---------------- STEP 4: SUMMARIZE ----------------
def summarize_clusters(articles: List[Dict], summarizer_pipeline) -> List[Dict]:
    clusters = {}
    for article in articles:
        cluster_id = article["cluster_id"]
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "articles": [],
                "summary": "",
                "image_url": None,
                "source_link": "",
            }
        clusters[cluster_id]["articles"].append(article)

    for cluster_id in clusters:
        clusters[cluster_id]["articles"].sort(key=lambda x: x["date"], reverse=True)
        most_recent = clusters[cluster_id]["articles"][0]
        clusters[cluster_id]["image_url"] = most_recent["image_url"]
        clusters[cluster_id]["source_link"] = most_recent["source"]

    MAX_SUMMARY_INPUT = 10000
    for cluster_id, data in tqdm(clusters.items(), desc="Summarizing Clusters"):
        combined_text = " ".join([a["text"] for a in data["articles"][:3]])
        combined_text = combined_text[:MAX_SUMMARY_INPUT]
        summary_text = "No summary available."
        try:
            summary_text = summarizer_pipeline(
                combined_text, max_length=150, min_length=30, do_sample=False
            )[0]["summary_text"]
        except Exception:
            pass
        data["summary"] = summary_text

    return [clusters[k] for k in sorted(clusters.keys())]

# ---------------- MAIN ----------------
def main():
    print("Loading NLP models...")
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print("Fetching RSS articles...")
    articles = fetch_rss(RSS_FEEDS)
    save_json(RAW_JSON, articles)

    print("Extracting full text...")
    articles = extract_full_text_parallel(articles)
    save_json(FILTERED_JSON, articles)
    if not articles:
        save_json(CLUSTERED_JSON, [])
        save_json(SUMMARIES_JSON, [])
        return

    print(f"Clustering into {NUM_CLUSTERS} clusters...")
    articles = cluster_articles(articles, NUM_CLUSTERS, sbert_model)

    print("Summarizing clusters...")
    final_data = summarize_clusters(articles, summarizer_pipeline)
    save_json(CLUSTERED_JSON, final_data)
    save_json(SUMMARIES_JSON, final_data)
    print(f"Processed {len(final_data)} clusters successfully.")

if __name__ == "__main__":
    main()
