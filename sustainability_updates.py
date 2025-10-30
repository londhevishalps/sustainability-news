#!/usr/bin/env python3
"""
sustainability_updates.py

Flow:
  1) fetch RSS -> raw_articles.json
  2) extract full text -> filtered_articles.json
  3) cluster -> clustered_articles.json
  4) summarize clusters -> summarized_articles.json

Notes:
 - Designed to run on CPU (GitHub Actions). Avoids launching Gradio when CI env var is set.
 - Be conservative with summarizer input length (truncate/join safely).
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict

import feedparser
import newspaper
import pandas as pd
from tqdm import tqdm

# NLP imports (heavy)
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
RSS_FEEDS = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

# If running in CI (GitHub Actions sets CI=true), skip launching Gradio later.
IN_CI = os.environ.get("CI", "").lower() in ("1", "true", "yes")

# ----------------------------
# UTIL
# ----------------------------
def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# STEP 1: FETCH ARTICLES FROM RSS FEEDS (last 7 days)
# ----------------------------
def fetch_articles(feeds: List[str]) -> List[Dict]:
    articles = []
    now = datetime.utcnow()
    cutoff = now - timedelta(days=7)
    for url in feeds:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"⚠️ Failed to parse feed {url}: {e}")
            continue

        for entry in feed.entries:
            # published may be missing or in different names; try a few fallbacks
            published_parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
            if not published_parsed:
                # skip items without a parseable date
                continue
            pub_date = datetime(*published_parsed[:6])
            if pub_date < cutoff:
                continue

            title = getattr(entry, "title", None) or getattr(entry, "summary", "")[:140]
            link = getattr(entry, "link", None)
            source = None
            if hasattr(entry, "source"):
                # entry.source may be a dict-like
                try:
                    source = entry.source.get("title") if isinstance(entry.source, dict) else entry.source
                except Exception:
                    source = None

            articles.append({
                "title": title,
                "link": link,
                "published": pub_date.strftime("%Y-%m-%d"),
                "source": source or ""
            })
    return articles

# Run Step 1
articles = fetch_articles(RSS_FEEDS)
save_json(RAW_JSON, articles)
print(f"✅ Step 1 complete: {len(articles)} articles saved to {RAW_JSON}")

# ----------------------------
# STEP 2: EXTRACT FULL TEXT FROM ARTICLES
# ----------------------------
def ensure_nltk_for_newspaper():
    # newspaper requires punkt; attempt to download if missing
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except Exception:
        try:
            import nltk
            nltk.download("punkt")
        except Exception as e:
            print(f"⚠️ Could not download nltk punkt: {e} (newspaper may fail on some sites)")

def extract_full_text(article_list: List[Dict]) -> List[Dict]:
    ensure_nltk_for_newspaper()
    extracted = []
    for art in tqdm(article_list, desc="Extracting articles", unit="article"):
        link = art.get("link")
        if not link:
            continue
        try:
            a = newspaper.Article(link)
            a.download()
            a.parse()
            text = a.text.strip()
            if not text:
                # skip empty extraction
                print(f"⚠️ Empty text for {link}")
                continue
            extracted.append({
                "title": art.get("title", "")[:300],
                "link": link,
                "published": art.get("published", ""),
                "source": art.get("source", ""),
                "text": text
            })
        except Exception as e:
            print(f"⚠️ Skipped: {link} ({e})")
    return extracted

raw_articles = load_json(RAW_JSON)
filtered_articles = extract_full_text(raw_articles)
save_json(FILTERED_JSON, filtered_articles)
print(f"✅ Step 2 complete: {len(filtered_articles)} articles saved to {FILTERED_JSON}")

# ----------------------------
# STEP 3: CLUSTER ARTICLES BY SIMILARITY
# ----------------------------
df = pd.DataFrame(filtered_articles)

if not df.empty:
    # Use a compact sentence-transformer (all-MiniLM...) which is CPU-friendly-ish
    print("🧠 Loading sentence-transformers model for embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

    n_clusters = min(NUM_CLUSTERS, len(df))
    if n_clusters < 1:
        n_clusters = 1

    print(f"🔀 Running KMeans with n_clusters={n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)

    # drop heavy fields if necessary when saving clustered json (keep text truncated)
    clustered_for_save = df.copy()
    clustered_for_save["text"] = clustered_for_save["text"].str.slice(0, 6000)  # keep a chunk
    clustered_for_save = clustered_for_save.to_dict(orient="records")
    save_json(CLUSTERED_JSON, clustered_for_save)
    print(f"✅ Step 3 complete: clusters saved to {CLUSTERED_JSON}")
else:
    print("⚠️ No articles to cluster; skipping Step 3.")
    save_json(CLUSTERED_JSON, [])
    df = pd.DataFrame()

# ----------------------------
# STEP 4: SUMMARIZE CLUSTERS (CPU-friendly)
# ----------------------------
clustered_data = load_json(CLUSTERED_JSON)
if not clustered_data:
    print("⚠️ No clustered data found. Skipping summarization.")
    save_json(SUMMARIES_JSON, [])
else:
    print("✂️ Initializing summarizer (CPU)...")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )

    df_clustered = pd.DataFrame(clustered_data)
    summaries = []

    for cluster_id in sorted(df_clustered["cluster"].unique()):
        subset = df_clustered[df_clustered["cluster"] == cluster_id]

        # Build joined input: titles + short text snippets (avoid sending whole articles)
        parts = []
        for _, row in subset.iterrows():
            title = row.get("title", "")[:300]
            src = row.get("source", "") or ""
            snippet = (row.get("text", "") or "")[:1200]  # keep snippet length sane
            parts.append(f"{title} ({src})\n{snippet}")

        joined_text = "\n\n".join(parts).strip()
        if not joined_text:
            joined_text = "No textual content available for these articles."

        # Truncate overall input to avoid token explosion (transformers CPU can choke)
        MAX_CHARS = 8000
        if len(joined_text) > MAX_CHARS:
            joined_text = joined_text[:MAX_CHARS]  # crude truncation; preserves start

        # Set summarizer length conservatively
        input_words = len(joined_text.split())
        max_len = max(60, min(200, input_words // 4))
        min_len = max(30, max_len // 3)

        try:
            result = summarizer(
                joined_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            summary_text = result[0].get("summary_text", "").strip()
        except Exception as e:
            summary_text = f"⚠️ Summary failed: {e}"

        summaries.append({
            "cluster_id": int(cluster_id),
            "summary": summary_text,
            "sources": list(subset["source"].unique()),
            "articles": subset[["title", "link", "published", "source"]].to_dict(orient="records")
        })

    save_json(SUMMARIES_JSON, summaries)
    print(f"✅ Step 4 complete: summaries saved to {SUMMARIES_JSON}")

# ----------------------------
# STEP 5: (optional) - Gradio viewer only when not in CI
# ----------------------------
def maybe_launch_gradio():
    try:
        import gradio as gr
    except Exception:
        print("ℹ️ Gradio not installed - skipping local viewer.")
        return

    if IN_CI:
        print("ℹ️ Running in CI - skipping Gradio server launch.")
        return

    # Only launch if explicitly requested via env var
    if os.environ.get("LAUNCH_GRADIO", "0") != "1":
        print("ℹ️ LAUNCH_GRADIO not set - skipping Gradio.")
        return

    def view_summaries():
        try:
            with open(SUMMARIES_JSON, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        except Exception as e:
            return f"⚠️ Error loading summaries: {e}"

        out = ""
        for s in summaries:
            out += f"### Cluster {s['cluster_id']}\n\n"
            out += f"**Summary:** {s['summary']}\n\n"
            out += "**Articles:**\n"
            for a in s["articles"]:
                out += f"- [{a['title']}]({a['link']}) ({a.get('published','')})\n"
            out += "\n---\n"
        return out

    iface = gr.Interface(fn=view_summaries, inputs=None, outputs="markdown", title="Sustainability News")
    iface.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    maybe_launch_gradio()
