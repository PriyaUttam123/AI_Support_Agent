import os
import sys
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

sys.stdout.reconfigure(encoding='utf-8')

train_path = "data/processed/apple_support/splits/train.csv"
print("Loading train.csv for NMF topic & intent discovery...")
df_train = pd.read_csv(train_path, low_memory=False)
print(f"Total training pairs: {len(df_train):,}")

# Preprocessing
clean_pat = re.compile(r'(@[A-Za-z0-9_]+|https?://\S+|t\.co/\S+|\[URL\])')
space_pat = re.compile(r'\s+')

def clean_for_nlp(text):
    t = clean_pat.sub(' ', str(text).lower())
    t = space_pat.sub(' ', t).strip()
    return t

texts_cleaned = df_train['customer_text_normalized'].apply(clean_for_nlp)

# Stopwords tailored for Twitter Apple Support
stopwords = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'could',
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has',
    'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
    'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor',
    'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until',
    'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    # Specific noise / boilerplate
    'apple', 'applesupport', 'support', 'http', 'https', 'user', 'url', 'please', 'help', 'thanks', 'hi',
    'hello', 'hey', 'getting', 'got', 'get', 'like', 'know', 'tell', 'want', 'need', 'trying', 'tried',
    'anyone', 'someone', 'day', 'today', 'days', 'time', 'times', 'still', 'since', 'way', 'make', 'see'
]

print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(
    stop_words=stopwords,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.5,
    max_features=12000
)
X_tfidf = tfidf.fit_transform(texts_cleaned)
feature_names = np.array(tfidf.get_feature_names_out())
print(f"TF-IDF shape: {X_tfidf.shape}")

# Fit NMF with 8 components
n_components = 8
print(f"\nFitting NMF with {n_components} components (random_state=42)...")
nmf = NMF(n_components=n_components, random_state=42, max_iter=200, init='nndsvd')
W = nmf.fit_transform(X_tfidf)
H = nmf.components_

topic_results = []
dominant_topics = W.argmax(axis=1)
df_train['topic'] = dominant_topics

for t_idx in range(n_components):
    top_indices = H[t_idx].argsort()[::-1][:15]
    top_words = feature_names[top_indices]
    
    topic_size = (dominant_topics == t_idx).sum()
    topic_pct = topic_size / len(df_train) * 100
    
    # Get top 5 queries with highest weight for this topic
    top_doc_indices = W[:, t_idx].argsort()[::-1][:4]
    sample_queries = df_train.iloc[top_doc_indices]['customer_text'].tolist()
    sample_responses = df_train.iloc[top_doc_indices]['response_text'].tolist()
    
    print(f"\n================ Topic {t_idx + 1} ================")
    print(f"Size: {topic_size:,} queries ({topic_pct:.2f}%)")
    print(f"Top Terms: {', '.join(top_words)}")
    print("Representative Queries & Responses:")
    for q, r in zip(sample_queries[:2], sample_responses[:2]):
        clean_q = q.replace('\n', ' ')
        clean_r = r.replace('\n', ' ')
        print(f"  [Q]: {clean_q[:110]}...")
        print(f"  [R]: {clean_r[:110]}...")
        
    topic_results.append({
        'topic_id': t_idx + 1,
        'size': int(topic_size),
        'pct': round(float(topic_pct), 2),
        'top_terms': top_words.tolist(),
        'representative_queries': sample_queries,
        'representative_responses': sample_responses
    })

with open("scratch/nmf_topic_discovery.json", "w", encoding="utf-8") as f:
    json.dump(topic_results, f, indent=2)

print("\nNMF Topic discovery completed successfully!")
