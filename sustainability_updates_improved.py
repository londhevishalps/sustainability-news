#!/usr/bin/env python3
"""
sustainability_updates_improved.py

Full pipeline:
1) Fetch RSS -> raw_articles.json
2) Extract full text -> filtered_articles.json
3) Filter for sustainability (keywords + optional semantic similarity)
4) Cluster -> clustered_articles.json
5) Summarize clusters -> summarized_articles.json

Features:
- Robust date parsing
- Image extraction
- Parallel text extraction
- Keyword + semantic filtering
- Clustering via SBERT
- Summarization via transformers
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

# #############################################################################
# CONFIG
# #############################################################################

RAW_JSON = "raw_articles.json"
FILTERED_JSON = "filtered_articles.json"
CLUSTERED_JSON = "clustered_articles.json"
SUMMARIES_JSON = "summarized_articles.json"

NUM_CLUSTERS = 5
CUTOFF_DAYS = 7
MAX_WORKERS = 8  # Parallel text extraction

RSS_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

# Sustainability keywords
SUSTAINABILITY_KEYWORDS = [
    "sustainability", "climate", "renewable", "carbon", "emissions",
    "green energy", "environment", "ecology", "net zero", "solar",
    "wind energy", "biodiversity", "recycling", "waste", "conservation",
    "water", "pollution", "deforestation", "electric vehicle", "clean tech"
]

# Optional semantic filtering reference
REFERENCE_TEXT = "Articles about sustainability, climate change, renewable energy, conservation, and environmental protection."
SEMANTIC_THRESHOLD = 0.5  # cosine similarity threshold

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
    # media_content
    if hasattr(entry, 'media_content') and entry.media_content:
        for media in entry.media_content:
            if 'url' in media and media.get('type', '').startswith('image'):
                return media['url']
    # enclosure
    if hasattr(entry, 'enclosures') and entry.enclosures:
        for enc in entry.enclosures:
            if enc.get('type', '').startswith('image'):
                return enc['href']
    # summary/description
    for field in ['summary', 'description', 'content']:
        if hasattr(entry, field):
            content = getattr(entry, field)
            if isinstance(content, list):
                content = content[0].get('value', '') if content else ''
            match = re.search(r'<img[^>]+src=["\'](.*?)["\']', content, re.IGNORECASE)
            if match:
                return match.group(1)
    return None

def is_sustainability_article(article: Dict) -> bool:
    combined_text = f"{article.get('title', '')} {article.get('text', '')}".lower()
    for keyword in SUSTAINABILITY_KEYWORDS:
        if keyword in combined_text:
            return True
    return False

# #############################################################################
# STEP 1: FETCH RSS
# #############################################################################

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
                    if not pub_date_str:
                        pub_date_struct = getattr(entry, 'published_parsed', None)
                        pub_date = datetime(*pub_date_struct[:6]) if pub_date_struct else None
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
# STEP 2: EXTRACT FULL TEXT (Parallel)
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
    return article if article["text"] and len(article["text"]) > 200 else None

def extract_full_text_parallel(articles: List[Dict]) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(_process_article_text, articles),
            total=len(articles),
            desc="Extracting Full Text (Parallel)"
        ))
    return [res for res in results if res is not None]

# #############################################################################
# STEP 3: CLUSTER
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
# STEP 4: SUMMARIZE
# #############################################################################

def summarize_clusters(articles: List[Dict], summarizer_pipeline) -> List[Dict]:
    clusters = {}
    for article in articles:
        cluster_id = article["cluster_id"]
        if cluster_id not in clusters:
            clusters[cluster_id] = {"articles": [], "source_link": "", "summary": "", "image_url": None}
        clusters[cluster_id]["articles"].append(article)
    for cluster_id in clusters:
        clusters[cluster_id]["articles"].sort(key=lambda x: x["date"], reverse=True)
        most_recent_article = clusters[cluster_id]["articles"][0]
        clusters[cluster_id]["source_link"] = most_recent_article["source"]
        clusters[cluster_id]["image_url"] = most_recent_article["image_url"]
    MAX_SUMMARY_INPUT = 10000
    for cluster_id, cluster_data in tqdm(clusters.items(), desc="Summarizing Clusters"):
        combined_text = " ".join([a["text"] for a in cluster_data["articles"][:3]])
        if len(combined_text) > MAX_SUMMARY_INPUT:
            combined_text = combined_text[:MAX_SUMMARY_INPUT]
        summary_text = "No summary available for this cluster."
        try:
            summary = summarizer_pipeline(combined_text, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
            summary_text = summary
        except Exception:
            pass
        cluster_data["summary"] = summary_text
    return [clusters[key] for key in sorted(clusters.keys())]

# #############################################################################
# STEP 5: SEMANTIC FILTER (OPTIONAL)
# #############################################################################

def semantic_filter_articles(articles: List[Dict], model: SentenceTransformer) -> List[Dict]:
    ref_emb = model.encode([REFERENCE_TEXT], show_progress_bar=False)
    filtered = []
    for a in articles:
        emb = model.encode([a['title'] + " " + a['text']], show_progress_bar=False)
        similarity = (emb @ ref_emb.T).item()  # dot product if normalized
        if similarity > SEMANTIC_THRESHOLD:
            filtered.append(a)
    return filtered

# #############################################################################
# MAIN
# #############################################################################

def main():
    print("--- Initializing NLP Models ---")
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception as e:
        print(f"FATAL: Could not load NLP models. {e}")
        return

    print("--- 1. Fetching RSS Articles ---")
    articles = fetch_rss(RSS_FEEDS)
    save_json(RAW_JSON, articles)
    print(f"Fetched {len(articles)} raw articles.")

    print("--- 2. Extracting Full Text (Parallel) ---")
    articles = extract_full_text_parallel(articles)
    save_json(FILTERED_JSON, articles)
    print(f"{len(articles)} articles with substantial text.")

    print("--- 2.5 Filtering for sustainability keywords ---")
    articles = [a for a in articles if is_sustainability_article(a)]
    print(f"{len(articles)} articles remain after keyword filtering.")

    # Optional: semantic filter
    print("--- 2.6 Optional semantic relevance filtering ---")
    articles = semantic_filter_articles(articles, sbert_model)
    print(f"{len(articles)} articles remain after semantic filtering.")

    if not articles:
        print("No sustainability articles found. Exiting.")
        save_json(CLUSTERED_JSON, [])
        return

    print(f"--- 3. Clustering Articles into {NUM_CLUSTERS} Clusters ---")
    articles = cluster_articles(articles, NUM_CLUSTERS, sbert_model)

    print("--- 4. Summarizing Clusters ---")
    final_data = summarize_clusters(articles, summarizer_pipeline)
    save_json(SUMMARIES_JSON, final_data)
    save_json(CLUSTERED_JSON, final_data)
    print(f"Processed {len(final_data)} clusters.")

if __name__ == "__main__":
    main()
