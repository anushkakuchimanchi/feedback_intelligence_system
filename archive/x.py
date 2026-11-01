import pandas as pd
k = pd.read_csv('../../data/processed/cluster_knowledge.csv')
for cid in [18, 20, 6, 12, 16]:
    r = k[k['cluster_id'] == cid].iloc[0]
    print(cid, r['title'])
    print('   1:', r['known_match_1'][:90])
    print('   2:', r['known_match_2'][:90])