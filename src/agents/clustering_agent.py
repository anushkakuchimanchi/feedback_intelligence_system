# src/agents/clustering_agent.py  (v2)
# Fixes: over-cluster + merge duplicates, category in embedded text, noise removal,
# cohesion score, top-3 evidence reviews, normalized centers, seed stability (ARI),
# optional PCA, embedding cache, faster KMeans update. numpy only.
import os, shutil, hashlib
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
KB = os.path.join(ROOT, 'data', 'knowledge_base')
os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(KB, exist_ok=True)

# ---------------- settings (tune these) ----------------
K_INITIAL = 40          # over-cluster first, then merge duplicates
MERGE_COS = 0.90        # merge two clusters if their centers have cosine > this
MAX_MERGED_SIZE = 600   # never merge into a cluster bigger than this
NOISE_COS = 0.60        # review with cosine-to-center below this -> noise (-1)
DROP_GENERAL = True     # 'general' category = vague, skip clustering them (-1)
PCA_DIMS = None         # e.g. 50 to cluster in PCA space (compare cohesion!)
STABILITY_CHECK = True  # re-run with other seeds and report ARI
SEED = 42


# ---------------- embeddings (cached) ----------------
def embed_texts(texts):
    key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()
    emb_path = os.path.join(KB, 'emb_cache.npy')
    key_path = os.path.join(KB, 'emb_cache.key')
    if os.path.exists(emb_path) and os.path.exists(key_path):
        if open(key_path).read().strip() == key:
            print("Embeddings loaded from cache.")
            return np.load(emb_path)
    from fastembed import TextEmbedding   # SAME model as every later stage
    model = TextEmbedding()
    emb = np.array(list(model.embed(texts)), dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(emb_path, emb)
    open(key_path, 'w').write(key)
    return emb


# ---------------- numpy KMeans ----------------
def sqdist(X, C):
    return (X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :]

def kmeans(X, k, n_init=5, iters=100, seed=42):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):                          # k-means++ init
            d2 = np.maximum(sqdist(X, np.array(centers)).min(1), 0)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(X[rng.choice(len(X), p=p)])
        C = np.array(centers)
        labels = None
        for _ in range(iters):
            new = sqdist(X, C).argmin(1)
            if labels is not None and (new == labels).all():
                break
            labels = new
            M = np.eye(k, dtype=X.dtype)[labels]       # one-hot, no python loop
            counts = M.sum(0)
            newC = (M.T @ X) / np.maximum(counts, 1)[:, None]
            empty = counts == 0
            if empty.any():
                newC[empty] = X[rng.integers(len(X), size=int(empty.sum()))]
            C = newC
        inertia = np.maximum(sqdist(X, C).min(1), 0).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels, C)
    return best[1]


# ---------------- helpers ----------------
def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)

def centers_of(E, labels, ids):
    return unit(np.array([E[labels == i].mean(0) for i in ids]))

def merge_clusters(E, labels, thr, max_size, verbose=True):
    labels = labels.copy()
    while True:
        ids = np.unique(labels)
        C = centers_of(E, labels, ids)
        S = C @ C.T
        np.fill_diagonal(S, -1)
        sizes = np.array([(labels == i).sum() for i in ids])
        S2 = S.copy()
        S2[(sizes[:, None] + sizes[None, :]) > max_size] = -1
        i, j = np.unravel_index(S2.argmax(), S2.shape)
        if S2[i, j] < thr:
            if verbose:
                flat = [(S[a, b], ids[a], ids[b]) for a in range(len(ids)) for b in range(a+1, len(ids))]
                flat.sort(reverse=True)
                print(f"Merging stopped. {len(ids)} clusters left. Closest remaining pairs (tune MERGE_COS):")
                for s, a, b in flat[:5]:
                    print(f"   clusters {a} & {b}: cosine {s:.3f}")
            return labels
        if verbose:
            print(f"Merge {ids[j]} -> {ids[i]} (cosine {S2[i, j]:.3f}, sizes {sizes[i]}+{sizes[j]})")
        labels[labels == ids[j]] = ids[i]

def ari(a, b):
    """Adjusted Rand Index in plain numpy. 1.0 = identical grouping."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(cont, (ai, bi), 1)
    c2 = lambda x: x * (x - 1) / 2
    s_ij, s_a, s_b, n = c2(cont).sum(), c2(cont.sum(1)).sum(), c2(cont.sum(0)).sum(), c2(len(a))
    exp = s_a * s_b / n
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0

def cluster_core(E, seed, verbose=True):
    """Embeddings (unit vectors) -> merged labels."""
    if PCA_DIMS:
        Xc = E - E.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = unit(Xc @ Vt[:PCA_DIMS].T)
    else:
        X = E
    labels = kmeans(X, K_INITIAL, n_init=5, seed=seed)
    return merge_clusters(E, labels, MERGE_COS, MAX_MERGED_SIZE, verbose)


# ---------------- main agent ----------------
def cluster_reviews():
    df = pd.read_csv(os.path.join(PROCESSED, 'classified_reviews.csv'))
    df['category'] = df['category'].replace('security', 'bug')
    print(f"Loaded {len(df)} reviews")

    # keep a backup of the v1 result once
    old = os.path.join(PROCESSED, 'clustered_reviews.csv')
    bak = os.path.join(PROCESSED, 'clustered_reviews_v1.csv')
    if os.path.exists(old) and not os.path.exists(bak):
        shutil.copy(old, bak)

    # category goes INTO the text so bugs and feature requests separate
    texts = (df['category'].fillna('') + " | " + df['area'].fillna('') + ": " + df['issue'].fillna('')).tolist()
    E_all = embed_texts(texts)                         # aligned with df rows

    df['cluster_id'] = -1
    df['noise_reason'] = ''
    mask = np.ones(len(df), dtype=bool)
    if DROP_GENERAL:
        gen = (df['category'] == 'general').values
        df.loc[gen, 'noise_reason'] = 'general'
        mask &= ~gen
        print(f"Skipping {gen.sum()} 'general' reviews (vague).")
    idx = np.where(mask)[0]
    E = E_all[idx]

    # 1) over-cluster + merge duplicates
    labels = cluster_core(E, SEED)

    # 2) stability across seeds
    if STABILITY_CHECK:
        scores = []
        for s in [1, 2, 3, 4]:
            other = cluster_core(E, s, verbose=False)
            scores.append(ari(labels, other))
        print(f"\nSeed stability (ARI vs seed {SEED}): " + ", ".join(f"{x:.2f}" for x in scores)
              + f"  | mean {np.mean(scores):.2f}  (>0.6 decent, >0.8 strong)")

    # 3) noise removal: cosine to own center
    ids = np.unique(labels)
    C = centers_of(E, labels, ids)
    pos = {c: n for n, c in enumerate(ids)}
    cos = np.array([E[n] @ C[pos[labels[n]]] for n in range(len(E))])
    print(f"\nCosine-to-center percentiles (5/25/50/75): {np.percentile(cos, [5, 25, 50, 75]).round(3)}")
    noisy = cos < NOISE_COS
    print(f"Noise (cos < {NOISE_COS}): {noisy.sum()} reviews")
    labels = np.where(noisy, -1, labels)
    df.loc[idx[noisy], 'noise_reason'] = 'low_cosine'

    # 4) relabel 0..m-1 by size (biggest first), recompute centers WITHOUT noise
    keep_ids = [c for c in np.unique(labels) if c != -1]
    keep_ids.sort(key=lambda c: -(labels == c).sum())
    remap = {c: n for n, c in enumerate(keep_ids)}
    final = np.array([remap.get(l, -1) for l in labels])
    m = len(keep_ids)
    C = centers_of(E, final, list(range(m)))          # normalized -> safe for live cosine

    # 5) per-review cosine, top-3 evidence per cluster, cohesion
    df['cos_to_center'] = np.nan
    df['evidence_rank'] = 0
    df['is_centroid'] = False
    rows = []
    for cid in range(m):
        mem = np.where(final == cid)[0]
        sims = E[mem] @ C[cid]
        order = np.argsort(-sims)
        seen, pick = set(), []
        for o in order:
            t = str(df.loc[idx[mem[o]], 'issue']).strip().lower()
            if t not in seen:
                seen.add(t)
                pick.append(o)
            if len(pick) == 3:
                break
        for r, o in enumerate(pick):
            df.loc[idx[mem[o]], 'evidence_rank'] = r + 1
        df.loc[idx[mem[pick[0]]], 'is_centroid'] = True
        df.loc[idx[mem], 'cos_to_center'] = sims
        ev = [df.loc[idx[mem[o]], 'issue'] for o in pick] + ['', '', '']
        rows.append({
            'cluster_id': cid, 'size': len(mem), 'cohesion': float(sims.mean()),
            'top_category': df.loc[idx[mem], 'category'].mode().iloc[0],
            'evidence_1': ev[0], 'evidence_2': ev[1], 'evidence_3': ev[2],
        })
    df.loc[idx, 'cluster_id'] = final
    summ = pd.DataFrame(rows)
    thr = summ['cohesion'].mean() - summ['cohesion'].std()
    summ['weak'] = summ['cohesion'] < thr

    # 6) save
    df.to_csv(os.path.join(PROCESSED, 'clustered_reviews.csv'), index=False)
    summ.to_csv(os.path.join(PROCESSED, 'cluster_summary.csv'), index=False)
    np.save(os.path.join(KB, 'embeddings.npy'), E_all)        # all rows, aligned with df
    np.save(os.path.join(KB, 'cluster_centers.npy'), C)       # unit-normalized

    # 7) report
    clustered = (df['cluster_id'] >= 0).sum()
    print("\n--- Stage 2 Metrics ---")
    print(f"Clusters: {m} (started at {K_INITIAL})")
    print(f"Clustered reviews: {clustered} | noise/general: {len(df) - clustered}")
    print(f"Mean cohesion: {summ['cohesion'].mean():.3f}  (v1 baseline: run once with K_INITIAL=20, MERGE_COS=2 to compare)")
    print(f"Sizes: min {summ['size'].min()}, max {summ['size'].max()}")
    print(f"Weak clusters (cohesion < {thr:.3f}): {summ.loc[summ['weak'], 'cluster_id'].tolist()}")

    print("\n--- Cluster preview (read these!) ---")
    for _, r in summ.iterrows():
        flag = "  [WEAK]" if r['weak'] else ""
        print(f"\nCluster {int(r['cluster_id'])} ({int(r['size'])} reviews, cohesion {r['cohesion']:.3f}, mostly {r['top_category']}){flag}")
        for k in ['evidence_1', 'evidence_2', 'evidence_3']:
            if r[k]:
                print(f"   - {r[k]}")


if __name__ == "__main__":
    cluster_reviews()