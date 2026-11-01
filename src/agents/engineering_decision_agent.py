import os, json, time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P=ROOT+'/data/processed/'
VER='d1'
V=['create_engineering_task','product_investigation_required','future_enhancement','no_engineering_action']
c=Groq(api_key=os.environ['GROQ_API_KEY'])
s=pd.read_csv(P+'cluster_stats.csv')
t=pd.read_csv(P+'cluster_problems.csv')[['cluster_id','title','problem_type','underlying_need']]
k=pd.read_csv(P+'cluster_knowledge.csv')
kc=[x for x in k.columns if x.startswith('known_match')]
d=s.merge(t,on='cluster_id').merge(k[['cluster_id']+kc],on='cluster_id',how='left')
SYS="""You are an engineering decision agent for the Termux Android app. Pick exactly one verdict for a cluster of customer reviews.
create_engineering_task: the evidence shows a significant problem in Termux's own code or build, so engineering investigation or implementation is justified.
product_investigation_required: the feedback is significant, but more information or product-level analysis is needed before creating a task. Use this when many users are affected but the cause is external, documented, mixed, or unclear.
future_enhancement: a useful feature or improvement, but not enough evidence for immediate engineering work.
no_engineering_action: isolated, low-impact, duplicate, non-actionable, or unrelated to an engineering problem.
Rules:
- Do not decide only from sentiment or review count. Use the rating gap, real example reviews, and the known-issue text.
- Android limits, dead mirrors, third-party package failures and the deprecated Play Store build are not Termux code defects. If many users are still hurt by them, choose product_investigation_required. If few are, choose no_engineering_action.
- Recency is a weak signal because few reviews are recent.
- Vague feedback with no specific failure is future_enhancement or no_engineering_action.
- If another cluster in the list covers the same problem, give the same kind of verdict.
Reply with JSON only: {"verdict":"...","reasoning":"2-3 sentences using the numbers","ai_inference":"1-2 sentences about the underlying customer need and likely cause"}"""
allt='\n'.join(str(r.cluster_id)+': '+str(r.title) for r in d.itertuples())
def facts(r):
    o=['review_count','avg_rating','rating_gap_vs_app','pct_low_rating','first_seen','last_seen','recent_12m_pct','text_pct_blocking']
    return ' | '.join(a+': '+str(r[a]) for a in o)
def prompt(r):
    kn=' ; '.join(str(r[x]) for x in kc if pd.notna(r[x]))
    ex='\n'.join('- '+str(r['evidence_review_'+str(i)])[:300] for i in (1,2,3))
    return ('Cluster '+str(r['cluster_id'])+': '+str(r['title'])+'\nType: '+str(r['problem_type'])+'\nUnderlying need: '+str(r['underlying_need'])+'\nObserved: '+facts(r)+'\nExample reviews:\n'+ex+'\nRetrieved known-issue text: '+kn+'\nAll clusters:\n'+allt)
out=P+'cluster_decisions_partial.csv'
rows=[]
if os.path.exists(out):
    x=pd.read_csv(out)
    if 'ver' in x.columns:
        rows=x[x['ver']==VER].to_dict('records')
ids={r['cluster_id'] for r in rows}
for _,r in d.iterrows():
    if r['cluster_id'] in ids:
        continue
    try:
        z=c.chat.completions.create(model='openai/gpt-oss-120b',messages=[{'role':'system','content':SYS},{'role':'user','content':prompt(r)}],temperature=0,reasoning_effort='low')
        a=z.choices[0].message.content
        j=json.loads(a[a.index('{'):a.rindex('}')+1])
        assert j['verdict'] in V
    except Exception as e:
        print('stopped at cluster',r['cluster_id'],':',e)
        break
    rows.append({'cluster_id':r['cluster_id'],'verdict':j['verdict'],'reasoning':j['reasoning'],'ai_inference':j['ai_inference'],'ver':VER})
    pd.DataFrame(rows).to_csv(out,index=False)
    print(r['cluster_id'],r['title'],'->',j['verdict'])
    time.sleep(12)
if len(rows)==len(d):
    f=pd.DataFrame(rows).merge(d,on='cluster_id')
    f['observed_evidence']=f.apply(facts,axis=1)
    q1=f['impact_rank_score'].quantile(1/3)
    q2=f['impact_rank_score'].quantile(2/3)
    def pr(r):
        if r['verdict']=='no_engineering_action':
            return 'n/a'
        if r['impact_rank_score']>=q2:
            return 'high'
        if r['impact_rank_score']>=q1:
            return 'medium'
        return 'low'
    f['priority']=f.apply(pr,axis=1)
    f['verdict_llm']=f['verdict']
    f['guard_note']=''
    f.to_csv(P+'cluster_decisions_final.csv',index=False)
    print(f['verdict'].value_counts())
else:
    print('not finished:',len(rows),'of',len(d),'- just run it again to resume')