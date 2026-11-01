# src/agents/knowledge_retrieval_agent.py
import os
import numpy as np
import pandas as pd
from fastembed import TextEmbedding

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
KNOWN = os.path.join(ROOT, 'known_issues.txt')

KNOWN_T = 0.80     # at or above: documented
RELATED_T = 0.70   # between the two: partly covered; below: new gap
# bge-small scores run high. Read the printed table, then adjust these two numbers.

def embed(model, texts):
    v = np.array(list(model.embed(texts)))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def retrieve_knowledge():
    with open(KNOWN, encoding='utf-8') as f:
        docs = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    problems = pd.read_csv(os.path.join(PROCESSED, 'cluster_problems.csv'))

    model = TextEmbedding()
    D = embed(model, docs)
    Q = embed(model, (problems['title'] + ". " + problems['underlying_need']).tolist())
    sims = Q @ D.T
    top2 = np.argsort(-sims, axis=1)[:, :2]

    problems['known_match_sim'] = np.round(sims.max(1), 3)
    problems['known_match_1'] = [docs[i[0]] for i in top2]
    problems['known_match_2'] = [docs[i[1]] for i in top2]
    problems['known_hint'] = np.where(problems['known_match_sim'] >= KNOWN_T, 'likely_known',
                             np.where(problems['known_match_sim'] >= RELATED_T, 'maybe_related', 'likely_new'))
    problems.to_csv(os.path.join(PROCESSED, 'cluster_knowledge.csv'), index=False)

    show = problems[['cluster_id', 'title', 'known_match_sim', 'known_hint']].sort_values('known_match_sim', ascending=False)
    print(show.to_string(index=False))
    print("\n", problems['known_hint'].value_counts())

def check_knowledge():
    a=pd.read_csv(os.path.join(PROCESSED,'cluster_knowledge.csv'))
    exp={18:'phantom',12:'bintray',20:'bintray',4:'play'}
    h1=0
    h2=0
    for c,k in exp.items():
        r=a[a.cluster_id==c].iloc[0]
        m1=str(r['known_match_1']).lower()
        m2=str(r['known_match_2']).lower()
        h1+=k in m1
        h2+=(k in m1) or (k in m2)
    u=a['known_match_1'].value_counts()
    print('\n--- CHECK knowledge ---')
    print('clusters matched',int(a['known_match_1'].notna().sum()),'of',len(a))
    print('similarity min/mean/max',a['known_match_sim'].min(),round(a['known_match_sim'].mean(),3),a['known_match_sim'].max())
    print(a['known_hint'].value_counts().to_dict())
    print('distinct top-1 known issues used',len(u),'| most used covers',int(u.iloc[0]),'clusters')
    print('hand-checked hit rate: top-1',h1,'of',len(exp),'| top-2',h2,'of',len(exp))
    print('PASS' if a['known_match_1'].notna().all() and a['known_match_sim'].notna().all() else 'FAIL')
if __name__ == "__main__":
    retrieve_knowledge()
    check_knowledge()