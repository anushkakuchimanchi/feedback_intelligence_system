# src/agents/clustering_agent.py
import pandas as pd
import numpy as np
from fastembed import TextEmbedding
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
KB = os.path.join(ROOT, 'data', 'knowledge_base')
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(KB, exist_ok=True)
K_OPTIONS = [10, 15, 20, 25, 30]
FORCE_K = 20   # set e.g. 20 to skip the k search

def sqdist(X, C):
    return (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]

def kmeans(X, k, n_init=5, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):                      # k-means++ init
            d2 = np.maximum(sqdist(X, np.array(centers)).min(1), 0)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(centers)
        labels = None
        for _ in range(iters):
            new_labels = sqdist(X, C).argmin(1)
            if labels is not None and (new_labels == labels).all():
                break
            labels = new_labels
            for c in range(k):
                pts = X[labels == c]
                C[c] = pts.mean(0) if len(pts) else X[rng.integers(len(X))]
        inertia = np.maximum(sqdist(X, C).min(1), 0).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels, C)
    return best[1], best[2]

def silhouette(X, labels):
    k = labels.max() + 1
    D = np.sqrt(np.maximum(sqdist(X, X), 0))
    M = np.eye(k)[labels]
    sums = D @ M
    counts = M.sum(0)
    idx = np.arange(len(X))
    own = counts[labels]
    a = np.where(own > 1, sums[idx, labels] / np.maximum(own - 1, 1), 0)
    mean_other = sums / np.maximum(counts, 1)
    mean_other[idx, labels] = np.inf
    b = mean_other.min(1)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return float(s.mean())

def cluster_reviews():
    df = pd.read_csv(os.path.join(PROCESSED, 'classified_reviews.csv'))
    df['category'] = df['category'].replace('security', 'bug')
    print(f"Clustering {len(df)} reviews...")

    texts = (df['area'].fillna('') + ": " + df['issue'].fillna('')).tolist()
    model = TextEmbedding()
    emb = np.array(list(model.embed(texts)))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    if FORCE_K:
        best_k = FORCE_K
    else:
        best_k, best_score = None, -1
        for k in K_OPTIONS:
            labels, _ = kmeans(emb, k)
            score = silhouette(emb, labels)
            print(f"k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_k, best_score = k, score
        print(f"Best k = {best_k}")

    labels, centers = kmeans(emb, best_k, n_init=10)
    df['cluster_id'] = labels

    centroid_idx = {}
    for cid in range(best_k):
        members = np.where(labels == cid)[0]
        dists = np.linalg.norm(emb[members] - centers[cid], axis=1)
        centroid_idx[cid] = members[dists.argmin()]
    df['is_centroid'] = df.index.isin(centroid_idx.values())

    df.to_csv(os.path.join(PROCESSED, 'clustered_reviews.csv'), index=False)
    np.save(os.path.join(KB, 'embeddings.npy'), emb)
    np.save(os.path.join(KB, 'cluster_centers.npy'), centers)

    print("\n--- Stage 2 Metrics ---")
    print(f"Clusters: {best_k}")
    print(f"Silhouette score: {silhouette(emb, labels):.3f}")
    print(f"Smallest cluster: {df['cluster_id'].value_counts().min()} reviews")
    print(f"Largest cluster: {df['cluster_id'].value_counts().max()} reviews")

    print("\n--- Cluster preview (read these!) ---")
    for cid in range(best_k):
        size = (labels == cid).sum()
        print(f"\nCluster {cid} ({size} reviews) | centroid: {df.loc[centroid_idx[cid], 'issue']}")
        for s in df[df['cluster_id'] == cid]['issue'].sample(min(3, size), random_state=1):
            print(f"   - {s}")

if __name__ == "__main__":
    cluster_reviews()