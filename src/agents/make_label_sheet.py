import pandas as pd, os
p='../../data/processed/'
s=pd.read_csv(p+'cluster_stats.csv')
t=pd.read_csv(p+'cluster_problems.csv')[['cluster_id','title']]
print(list(s.columns))
want=['cluster_id','review_count','avg_rating','rating_gap_vs_app','pct_low_rating','first_seen','last_seen','evidence_review_1','evidence_review_2','evidence_review_3']
cols=[c for c in want if c in s.columns]+['centroid_review']
d=s[cols].merge(t,on='cluster_id').sample(frac=1,random_state=7)
d['human_create']=''
os.makedirs('../../eval',exist_ok=True)
for n in ['A','B','C']:
    d.to_csv('../../eval/human_labels_'+n+'.csv',index=False)
print(len(d),'rows written for 3 labelers')