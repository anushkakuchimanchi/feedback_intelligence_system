# src/agents/problem_identification_agent.py
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(ROOT, 'src'))
from utils import call_llm_json

PROCESSED = os.path.join(ROOT, 'data', 'processed')
OUT = os.path.join(PROCESSED, 'cluster_problems.csv')
TYPES = ["defect", "missing_feature", "usability", "performance_reliability", "enhancement", "non_actionable"]

def identify_problems():
    df = pd.read_csv(os.path.join(PROCESSED, 'clustered_reviews.csv'))
    stats = pd.read_csv(os.path.join(PROCESSED, 'cluster_stats.csv')).set_index('cluster_id')
    evidence_cols = [c for c in stats.columns if c.startswith('evidence_issue')]

    rows = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
    done_ids = {r['cluster_id'] for r in rows}

    cluster_ids = sorted(c for c in df['cluster_id'].unique() if c >= 0)   # skip -1 (general, set aside)
    for cid in cluster_ids:
        if cid in done_ids:
            continue

        evidence = []
        if cid in stats.index:
            evidence = [str(stats.loc[cid, c]) for c in evidence_cols if pd.notna(stats.loc[cid, c])]
        issues = df[df['cluster_id'] == cid]['issue'].dropna()
        samples = "\n".join("- " + s for s in issues.sample(min(12, len(issues)), random_state=1))
        typical = "\n".join("- " + e for e in evidence) or "(none)"

        prompt = f"""These customer complaints about the Termux Android app were grouped into one cluster by similarity.

Most typical complaints:
{typical}

Other members:
{samples}

Return JSON with:
- title: short name for the problem, max 8 words
- problem_type: one of {TYPES}
- underlying_need: one sentence on what the customer actually needs (do not just repeat their words)
- actionable: true if engineers could act on this, false if it is too vague
- coherence: "high", "medium" or "low", how well the members describe ONE problem

Return ONLY JSON like {{"title": "...", "problem_type": "defect", "underlying_need": "...", "actionable": true, "coherence": "high"}}"""
        try:
            r = call_llm_json(prompt, max_tokens=1000)
        except Exception as e:
            print(f"Stopped at cluster {cid}: {e}\nRerun the script, it resumes here.")
            break

        rows.append({"cluster_id": cid, **r})
        pd.DataFrame(rows).to_csv(OUT, index=False)      # save after every cluster
        print(f"Cluster {cid}: {r.get('title')} [{r.get('problem_type')}, actionable={r.get('actionable')}, coherence={r.get('coherence')}]")

    print(f"\nNamed {len(rows)}/{len(cluster_ids)} clusters -> cluster_problems.csv")

def check_problems():
    import numpy as np
    from fastembed import TextEmbedding
    p=pd.read_csv(OUT)
    s=pd.read_csv(os.path.join(PROCESSED,'cluster_stats.csv'))
    ec=[c for c in s.columns if c.startswith('evidence_issue')]
    d=p.merge(s[['cluster_id']+ec],on='cluster_id')
    m=TextEmbedding()
    e=lambda x:(lambda v:v/np.linalg.norm(v,axis=1,keepdims=True))(np.array(list(m.embed(x))))
    T=e((d['title']+'. '+d['underlying_need']).tolist())
    C=e(d[ec].fillna('').astype(str).agg(' ; '.join,axis=1).tolist())
    S=T@C.T
    rk=np.argsort(-S,axis=1)
    n=len(d)
    top1=np.mean(rk[:,0]==np.arange(n))
    top3=np.mean([i in rk[i,:3] for i in range(n)])
    TT=T@T.T
    np.fill_diagonal(TT,0)
    pairs=[(int(d.cluster_id[i]),int(d.cluster_id[j])) for i in range(n) for j in range(i+1,n) if TT[i,j]>0.88]
    print('\n--- CHECK problems ---')
    print('named',int(d['title'].notna().sum()),'of',n,'| valid types',bool(d['problem_type'].isin(TYPES).all()),'| duplicate titles',int(d['title'].duplicated().sum()))
    print('title matches its own cluster: top-1',round(100*top1),'% | top-3',round(100*top3),'%')
    print('near-duplicate cluster pairs (sim>0.88):',pairs)
    print(d['problem_type'].value_counts().to_string())
    print('actionable:',d['actionable'].value_counts().to_dict(),'| coherence:',d['coherence'].value_counts().to_dict())
    print('PASS' if d['title'].notna().all() and d['problem_type'].isin(TYPES).all() else 'FAIL')
if __name__ == "__main__":
    identify_problems()
    check_problems()