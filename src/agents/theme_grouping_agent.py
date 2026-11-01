# src/agents/theme_grouping_agent.py
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(ROOT, 'src'))
from utils import call_llm_json

PROCESSED = os.path.join(ROOT, 'data', 'processed')

def group_themes():
    probs = pd.read_csv(os.path.join(PROCESSED, 'cluster_problems.csv'))
    valid = {int(c) for c in probs['cluster_id']}
    lines = "\n".join(f"{r.cluster_id}: {r.title} | {r.underlying_need}" for r in probs.itertuples())
    prompt = f"""Below are {len(probs)} clusters of customer complaints about the Termux Android app.
Group the clusters that describe the SAME underlying problem for an engineering team
(for example, several clusters about failing package installs belong together).
Use 6 to 9 themes. Every cluster must be in exactly one theme.

{lines}

Return ONLY JSON: {{"themes": [{{"theme": "short theme name", "cluster_ids": [0, 9]}}]}}"""
    result = call_llm_json(prompt, max_tokens=2000)

    mapping = {}
    for t in result.get('themes', []):
        for cid in t.get('cluster_ids', []):
            if int(cid) in valid:
                mapping.setdefault(int(cid), t.get('theme'))
    for r in probs.itertuples():                    # anything the model forgot becomes its own theme
        mapping.setdefault(int(r.cluster_id), r.title)

    out = pd.DataFrame({'cluster_id': list(mapping.keys()), 'theme': list(mapping.values())}).sort_values('cluster_id')
    out.to_csv(os.path.join(PROCESSED, 'cluster_themes.csv'), index=False)
    for theme, g in out.groupby('theme'):
        print(f"{theme}: clusters {sorted(g['cluster_id'].tolist())}")

if __name__ == "__main__":
    group_themes()