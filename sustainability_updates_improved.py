#!/usr/bin/env python3
"""
sustainability_updates_improved.py

Flow:
1) fetch RSS -> raw_articles.json
2) extract full text -> filtered_articles.json
3) filter for sustainability keywords & semantic relevance
4) cluster -> clustered_articles.json
5) summarize clusters -> summarized_articles.json

Improvements:
- Robust date parsing for RSS feeds.
- More robust image URL extraction.
- Use of concurrent futures for parallel text extraction.
- Centralized model loading to avoid re-initialization.
- Safer semantic filtering to avoid zero articles.
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
import torch

# #############################################################################
# CONFIG
# #############################################################################

RAW_JSON = "raw_articles.json"
FILTERED_JSON = "filtered_articles.json"
CLUSTERED_JSON = "clustered_articles.json"
SUMMARIES_JSON = "summarized_articles.json"

NUM_CLUSTERS = 5
CUTOFF_DAYS = 7
MAX_WORKERS = 8  # parallel text extraction
SUSTAINABILITY_KEYWORDS = [
    "sustainability", "climate", "carbon", "green", "energy", "environment",
    "recycle", "emission", "pollution", "eco", "net-zero", "biodiversity"
]

RSS_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

# #############################################################################
# UTILITIES
# #############################################################################

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

# #############################################################################
# STEP 1: FETCH RSS
# #############################################################################

def fetch_rss(feeds: List[str]) -> List[Dict]:
    all_articles = []
    cutoff_date = datetime.now() - timedelta(days=CUTOFF_DAYS)
    for url in tqdm(feeds, desc="Fetching RSS Feeds"):
        try:
            feed = feedparser.parse(url)
            feed_title = feed.feed.title if hasattr(feed.feed, 'title') else url
            for entry in feed.entries:
                try:
                    pub_date_str = getattr(entry, 'published', None)
                    if not pub_date_str:
                        pub_date_struct = getattr(entry, 'published_parsed', None)
                        if pub_date_struct:
                            pub_date = datetime(*pub_date_struct[:6])
                        else:
                            continue
                    else:
                        pub_date = parse_date(pub_date_str)
                    if not pub_date or pub_date < cutoff_date:
                        continue
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
                except Exception:
                    continue
        except Exception:
            continue
    return all_articles

# #############################################################################
# STEP 2: FULL TEXT EXTRACTION
# #############################################################################

def _process_article_text(article: Dict) -> Optional[Dict]:
    if not article["text"] or len(article["text"]) < 100:
        try:
            a = newspaper.Article(article["url"])
            a.download()
            a.parse()
            article["text"] = a.text
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
    return [res for res in results if res is not None]

# #############################################################################
# STEP 3: KEYWORD & SEMANTIC FILTERING
# #############################################################################

def filter_for_sustainability(articles: List[Dict], sbert_model) -> List[Dict]:
    # 1) Keyword filter
    filtered_kw = []
    for article in articles:
        text = (article["title"] + " " + article["text"]).lower()
        if any(k.lower() in text for k in SUSTAINABILITY_KEYWORDS):
            filtered_kw.append(article)
    print(f"{len(filtered_kw)} articles remain after keyword filtering.")

    if not filtered_kw:
        return []

    # 2) Semantic filter (safer version)
    MIN_SIMILARITY_THRESHOLD = 0.5
    filtered_semantic = []

    # Precompute sustainability embedding
    sustainability_embedding = sbert_model.encode(
        "sustainability environment climate green energy pollution carbon emission eco net-zero biodiversity",
        convert_to_tensor=True
    )

    for article in filtered_kw:
        try:
            embedding = sbert_model.encode(article['text'], convert_to_tensor=True)
            similarity = float(torch.cosine_similarity(embedding, sustainability_embedding))
            if similarity >= MIN_SIMILARITY_THRESHOLD:
                filtered_semantic.append(article)
            else:
                print(f"Excluded by semantic filter: {article['title']} (sim={similarity:.2f})")
        except Exception as e:
            print(f"Warning computing similarity for '{article['title']}': {e}")
            filtered_semantic.append(article)  # fallback
    print(f"{len(filtered_semantic)} articles remain after semantic filtering.")
    return filtered_semantic

# #############################################################################
# STEP 4: CLUSTERING
# #############################################################################

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

# #############################################################################
# STEP 5: SUMMARIZATION
# #############################################################################

def summarize_clusters(articles: List[Dict], summarizer_pipeline) -> List[Dict]:
    clusters = {}
    for article in articles:
        cid = article["cluster_id"]
        if cid not in clusters:
            clusters[cid] = {"articles": [], "source_link": "", "summary": "", "image_url": None}
        clusters[cid]["articles"].append(article)

    for cid, cluster_data in clusters.items():
        cluster_data["articles"].sort(key=lambda x: x["date"], reverse=True)
        if cluster_data["articles"]:
            most_recent = cluster_data["articles"][0]
            cluster_data["source_link"] = most_recent["source"]
            cluster_data["image_url"] = most_recent["image_url"]

    MAX_SUMMARY_INPUT = 10000
    for cid, cluster_data in tqdm(clusters.items(), desc="Summarizing Clusters"):
        combined_text = " ".join([a["text"] for a in cluster_data["articles"][:3]])
        if len(combined_text) > MAX_SUMMARY_INPUT:
            combined_text = combined_text[:MAX_SUMMARY_INPUT]
        try:
            summary = summarizer_pipeline(
                combined_text, max_length=150, min_length=30, do_sample=False
            )[0]["summary_text"]
            cluster_data["summary"] = summary
        except Exception:
            cluster_data["summary"] = "No summary available for this cluster."

    final_data = [clusters[key] for key in sorted(clusters.keys())]
    return final_data

# #############################################################################
# MAIN
# #############################################################################

def main():
    print("--- Initializing NLP Models ---")
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"FATAL: Could not load NLP models: {e}")
        return

    print("--- 1. Fetching RSS Articles ---")
    articles = fetch_rss(RSS_FEEDS)
    save_json(RAW_JSON, articles)
    print(f"Fetched {len(articles)} raw articles.")

    print("--- 2. Extracting Full Text ---")
    articles = extract_full_text_parallel(articles)
    save_json(FILTERED_JSON, articles)
    print(f"{len(articles)} articles with substantial text after extraction.")

    if not articles:
        print("No articles to process. Exiting.")
        save_json(CLUSTERED_JSON, [])
        return

    print("--- 3. Filtering for Sustainability ---")
    articles = filter_for_sustainability(articles, sbert_model)
    if not articles:
        print("No sustainability articles found. Exiting.")
        save_json(CLUSTERED_JSON, [])
        return

    print(f"--- 4. Clustering Articles into {NUM_CLUSTERS} Clusters ---")
    articles = cluster_articles(articles, NUM_CLUSTERS, sbert_model)

    print("--- 5. Summarizing Clusters ---")
    final_data = summarize_clusters(articles, summarizer_pipeline)
    save_json(SUMMARIES_JSON, final_data)
    save_json(CLUSTERED_JSON, final_data)

    print(f"Successfully processed {len(final_data)} clusters.")

if __name__ == "__main__":
    main()
