import os, json, time
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
c=Groq(api_key=os.environ['GROQ_API_KEY'])
d=pd.read_csv('../../eval/human_labels_A.csv')
rule="Mark 1 if a Termux developer should open an engineering task to change Termux's own code or build. Mark 0 if it's an Android limit, a dead mirror, a third-party package problem, the deprecated Play build, vague feedback, or not worth engineering time."
out=[]
for i,r in d.iterrows():
    t=r.drop(['human_create']).to_string()
    m=[{"role":"system","content":"You label feedback clusters. "+rule+' Reply only with JSON: {"create":0 or 1,"why":"one short sentence"}'},{"role":"user","content":t}]
    try:
        x=c.chat.completions.create(model='openai/gpt-oss-120b',messages=m,temperature=0,reasoning_effort='low')
        j=json.loads(x.choices[0].message.content.strip().strip('`').replace('json','',1))
    except Exception as e:
        print('stopped:',e)
        break
    out.append({'cluster_id':r['cluster_id'],'llm_create':int(j['create']),'why':j['why']})
    print(r['cluster_id'],j['create'])
    time.sleep(12)
pd.DataFrame(out).to_csv('../../eval/llm_labels.csv',index=False)
print(len(out),'labelled')