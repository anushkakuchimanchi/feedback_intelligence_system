import pandas as pd
p='../../data/processed/'
d=pd.read_csv(p+'cluster_decisions.csv')
s=pd.read_csv(p+'cluster_stats.csv')[['cluster_id','review_count']]
if 'review_count' not in d.columns:
    d=d.merge(s,on='cluster_id',how='left')
d['verdict_llm']=d['verdict']
d['guard_note']=''
hi=d['impact_rank_score'].quantile(2/3)
med=d['impact_rank_score'].median()
for i,r in d.iterrows():
    if r['verdict']=='no_engineering_action' and r['review_count']>=100 and r['impact_rank_score']>=hi:
        d.loc[i,'verdict']='product_investigation_required'
        d.loc[i,'priority']='high'
        d.loc[i,'guard_note']='upgraded: large high-impact cluster'
    if r['verdict']=='create_engineering_task' and r['review_count']<60 and r['impact_rank_score']<med:
        d.loc[i,'verdict']='product_investigation_required'
        d.loc[i,'guard_note']='downgraded: low volume and low impact'
d.to_csv(p+'cluster_decisions_final.csv',index=False)
print('hi_cut',round(hi,2),'median',round(med,2))
print(d[d.guard_note!=''][['cluster_id','review_count','impact_rank_score','verdict_llm','verdict','guard_note']].to_string())
print(d['verdict'].value_counts())