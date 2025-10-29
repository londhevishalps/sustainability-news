import feedparser
import newspaper
import pandas as pd
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline
import torch
import gradio as gr

# ----------------------------
# CONFIGURATION
# ----------------------------
RAW_JSON = "raw_articles.json"
FILTERED_JSON = "filtered_articles.json"
CLUSTERED_JSON = "clustered_articles.json"
SUMMARIES_JSON = "summarized_articles.json"

# Number of clusters for grouping similar articles
NUM_CLUSTERS = 5

# ----------------------------
# STEP 1: FETCH ARTICLES FROM RSS FEEDS
# ----------------------------
rss_feeds = [
    "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml",
    "https://www.reuters.com/feeds/rss/environment",
    "https://www.theguardian.com/environment/rss",
    "https://www.cnbc.com/id/19836768/device/rss/rss.html",
    "https://www.euractiv.com/section/energy-environment/feed/"
]

def fetch_articles(feeds):
    articles = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            published = getattr(entry, 'published', None)
            if published:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                except Exception:
                    continue
                if pub_date >= datetime.now() - timedelta(days=7):  # last 7 days
                    articles.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": pub_date.strftime("%Y-%m-%d")
                    })
    return articles

articles = fetch_articles(rss_feeds)

with open(RAW_JSON, 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"✅ Step 1 complete: {len(articles)} articles saved to {RAW_JSON}")

# ----------------------------
# STEP 2: EXTRACT FULL TEXT FROM ARTICLES
# ----------------------------
def extract_full_text(article_list):
    extracted = []
    for art in article_list:
        try:
            a = newspaper.Article(art["link"])
            a.download()
            a.parse()
            extracted.append({
                "title": art["title"],
                "link": art["link"],
                "published": art["published"],
                "text": a.text
            })
        except Exception as e:
            print(f"⚠️ Skipped: {art['link']} ({e})")
    return extracted

with open(RAW_JSON, "r", encoding="utf-8") as f:
    raw_articles = json.load(f)

filtered_articles = extract_full_text(raw_articles)

with open(FILTERED_JSON, "w", encoding="utf-8") as f:
    json.dump(filtered_articles, f, indent=2, ensure_ascii=False)

print(f"✅ Step 2 complete: {len(filtered_articles)} articles saved to {FILTERED_JSON}")

# ----------------------------
# STEP 3: CLUSTER ARTICLES BY SIMILARITY
# ----------------------------
df = pd.DataFrame(filtered_articles)

if not df.empty:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

    kmeans = KMeans(n_clusters=min(NUM_CLUSTERS, len(df)), random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(embeddings)

    df.to_json(CLUSTERED_JSON, orient="records", force_ascii=False, indent=2)
    print(f"✅ Step 3 complete: clusters saved to {CLUSTERED_JSON}")
else:
    print("⚠️ No data to cluster.")
    df = pd.DataFrame()

# ----------------------------
# STEP 4: SUMMARIZE EACH CLUSTER
# ----------------------------
if not df.empty:
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    summaries = []
    for cluster_id in df["cluster"].unique():
        cluster_texts = df[df["cluster"] == cluster_id]["text"].tolist()
        combined_text = " ".join(cluster_texts)[:4000]  # keep manageable length

        summary = summarizer(combined_text, max_length=180, min_length=60, do_sample=False)[0]["summary_text"]

        summaries.append({
            "cluster_id": int(cluster_id),
            "summary": summary,
            "articles": df[df["cluster"] == cluster_id][["title", "link", "published"]].to_dict(orient="records")
        })

    with open(SUMMARIES_JSON, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"✅ Step 4 complete: summaries saved to {SUMMARIES_JSON}")
else:
    print("⚠️ No clusters to summarize.")

# ----------------------------
# STEP 5: CREATE GRADIO INTERFACE
# ----------------------------
def view_summaries():
    try:
        with open(SUMMARIES_JSON, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        output = ""
        for s in summaries:
            output += f"### 🟩 Cluster {s['cluster_id']}\n\n"
            output += f"**Summary:** {s['summary']}\n\n"
            output += "**Articles:**\n"
            for a in s["articles"]:
                output += f"- [{a['title']}]({a['link']}) ({a['published']})\n"
            output += "\n---\n"
        return output
    except Exception as e:
        return f"⚠️ Error loading summaries: {e}"

iface = gr.Interface(fn=view_summaries, inputs=None, outputs="markdown", title="🌍 Sustainability News Updates")
iface.launch(server_name="0.0.0.0", share=False)
