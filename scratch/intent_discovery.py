import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

train_path = "data/processed/apple_support/splits/train.csv"
print("Loading train.csv for intent discovery...")
df_train = pd.read_csv(train_path, low_memory=False)
print(f"Total training pairs: {len(df_train):,}")

# Preprocessing for TF-IDF and clustering
stopwords = set("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't
she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
applesupport apple support http https t co com please help thanks hi hello new getting got get user url
""".split())

clean_pat = re.compile(r'(@[A-Za-z0-9_]+|https?://\S+|t\.co/\S+|\[URL\])')
space_pat = re.compile(r'\s+')

def clean_for_nlp(text):
    t = clean_pat.sub(' ', str(text).lower())
    t = space_pat.sub(' ', t).strip()
    return t

texts_cleaned = df_train['customer_text_normalized'].apply(clean_for_nlp)

# Query length statistics
word_counts = texts_cleaned.apply(lambda s: len(s.split()))
char_counts = texts_cleaned.apply(len)
print(f"Query length (words): mean={word_counts.mean():.1f}, median={word_counts.median()}, min={word_counts.min()}, max={word_counts.max()}")
print(f"Query length (chars): mean={char_counts.mean():.1f}, median={char_counts.median()}, min={char_counts.min()}, max={char_counts.max()}")

# TF-IDF Vectorization
print("Fitting TF-IDF on training customer messages...")
tfidf = TfidfVectorizer(
    stop_words=list(stopwords),
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.4,
    max_features=10000
)
X_tfidf = tfidf.fit_transform(texts_cleaned)
feature_names = np.array(tfidf.get_feature_names_out())
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# MiniBatchKMeans Clustering
k = 12
print(f"Running MiniBatchKMeans with k={k}, random_state=42...")
kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=5)
cluster_labels = kmeans.fit_predict(X_tfidf)
df_train['cluster'] = cluster_labels

clusters_summary = []
# Find top terms and representative examples for each cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(k):
    cluster_size = (cluster_labels == i).sum()
    pct = cluster_size / len(df_train) * 100
    top_terms = [feature_names[ind] for ind in order_centroids[i, :15]]
    
    # Representative examples: closest to cluster centroid
    cluster_indices = np.where(cluster_labels == i)[0]
    # Sample 4 random or top examples
    sample_examples = df_train.iloc[cluster_indices]['customer_text'].head(4).tolist()
    
    print(f"\n--- Cluster {i}: {cluster_size:,} queries ({pct:.2f}%) ---")
    print(f"Top terms: {', '.join(top_terms[:10])}")
    print("Examples:")
    for ex in sample_examples[:3]:
        print(f"  * {ex[:120]}...")
        
    clusters_summary.append({
        'cluster_id': i,
        'size': int(cluster_size),
        'pct': round(pct, 2),
        'top_terms': top_terms,
        'examples': sample_examples
    })

# Save discovery results
with open("scratch/intent_discovery_clusters.json", "w", encoding="utf-8") as f:
    json.dump(clusters_summary, f, indent=2)

print("\nIntent discovery clustering complete!")
