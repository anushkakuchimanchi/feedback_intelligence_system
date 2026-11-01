# src/agents/impact_analysis_agent.py  (v2)
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
RAW = os.path.join(ROOT, 'data', 'raw', 'termux_reviews.csv')

WINDOW_DAYS = 365     # recency window (was 90, too few reviews)
DORMANT_DAYS = 180    # no new review in this long = probably fixed / dormant
BLOCKING = (r"\b(?:cannot|can't|can not|unable|fails?|failed|failing|not working|"
            r"does not work|doesn't work|won't|no longer|crash(?:es|ing)?|unusable|stuck|broken)\b")

def analyze_impact():
    df = pd.read_csv(os.path.join(PROCESSED, 'clustered_reviews.csv'))
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['cluster_id'] >= 0].copy()          # drop general/noise (-1)
    total = len(df)

    app_rating = pd.read_csv(RAW)['rating'].mean()  # overall rating, all scraped reviews
    max_date = df['date'].max()
    cutoff = max_date - pd.DateOffset(days=WINDOW_DAYS)
    dormant_cut = max_date - pd.DateOffset(days=DORMANT_DAYS)
    overall_recent = (df['date'] >= cutoff).mean()
    df['blocking'] = df['issue'].fillna('').str.contains(BLOCKING, case=False, regex=True)

    rows = []
    for cid, g in df.groupby('cluster_id'):
        ev = g[g['evidence_rank'] > 0].sort_values('evidence_rank')
        recent = (g['date'] >= cutoff).mean()
        months = g['date'].dt.to_period('M')
        months_span = (months.max() - months.min()).n + 1
        row = {
            'cluster_id': int(cid),
            # ---- OBSERVED (straight from review data) ----
            'review_count': len(g),
            'share_pct': round(len(g) / total * 100, 1),
            'avg_rating': round(g['rating'].mean(), 2),
            'rating_gap_vs_app': round(g['rating'].mean() - app_rating, 2),
            'pct_low_rating': round((g['rating'] <= 2).mean() * 100, 1),
            'first_seen': g['date'].min().date(),
            'last_seen': g['date'].max().date(),
            'recent_12m_pct': round(recent * 100, 1),
            'trend_ratio': round(recent / overall_recent, 2) if overall_recent else None,
            'months_active': int(months.nunique()),
            'months_span': int(months_span),
            # ---- RULE-BASED (plain code, no LLM) ----
            'text_pct_blocking': round(g['blocking'].mean() * 100, 1),
            'status': 'dormant' if g['date'].max() < dormant_cut else 'active',
            # ---- AI-DERIVED (Stage 1 LLM labels) ----
            'pct_high_severity': round((g['severity'] == 'high').mean() * 100, 1),
            'top_category': g['category'].mode()[0],
        }
        for r in range(1, 4):                       # top-3 evidence reviews
            if len(ev) >= r:
                row[f'evidence_issue_{r}'] = ev.iloc[r-1]['issue']
                row[f'evidence_review_{r}'] = ev.iloc[r-1]['review_text']
            else:
                row[f'evidence_issue_{r}'] = ''
                row[f'evidence_review_{r}'] = ''
        row['centroid_issue'] = row['evidence_issue_1']      # kept for old scripts
        row['centroid_review'] = row['evidence_review_1']
        rows.append(row)

    out = pd.DataFrame(rows)

    # rule-based priority from OBSERVED inputs only (no AI severity)
    pr = lambda s: s.rank(pct=True)
    out['impact_rank_score'] = ((pr(out['review_count']) + pr(-out['rating_gap_vs_app'])
                                 + pr(out['pct_low_rating']) + pr(out['recent_12m_pct'])) / 4).round(2)

    summ_path = os.path.join(PROCESSED, 'cluster_summary.csv')
    if os.path.exists(summ_path):
        s = pd.read_csv(summ_path)[['cluster_id', 'cohesion', 'weak']]
        out = out.merge(s, on='cluster_id', how='left')

    out = out.sort_values('impact_rank_score', ascending=False)
    out.to_csv(os.path.join(PROCESSED, 'cluster_stats.csv'), index=False)

    print(f"App overall rating (all scraped reviews): {app_rating:.2f}")
    print(f"Clustered reviews: {total} | window: last {WINDOW_DAYS} days up to {max_date.date()}\n")
    print("OBSERVED + RULE-BASED:")
    print(out[['cluster_id', 'review_count', 'share_pct', 'avg_rating', 'rating_gap_vs_app',
               'pct_low_rating', 'recent_12m_pct', 'trend_ratio', 'months_active', 'months_span',
               'text_pct_blocking', 'status', 'impact_rank_score']].to_string(index=False))
def check_impact():
    a=pd.read_csv(os.path.join(PROCESSED,'cluster_stats.csv'))
    r=lambda s:s.rank()
    c1=r(a['impact_rank_score']).corr(r(a['review_count']))
    c2=r(a['impact_rank_score']).corr(r(-a['rating_gap_vs_app']))
    c3=r(a['pct_high_severity']).corr(r(a['pct_low_rating']))
    c4=r(a['text_pct_blocking']).corr(r(a['pct_low_rating']))
    print('\n--- CHECK impact ---')
    print('clusters',len(a),'| share sum',round(a['share_pct'].sum(),1),'| null scores',int(a['impact_rank_score'].isna().sum()))
    print('score vs review_count',round(c1,2),'| score vs rating gap',round(c2,2))
    print('AI severity vs observed low-rating %',round(c3,2),'| blocking-text vs low-rating %',round(c4,2))
    print('clusters under 3% recent',int((a['recent_12m_pct']<3).sum()),'of',len(a),'| dormant',int((a['status']=='dormant').sum()))
    print(a.nlargest(5,'impact_rank_score')[['cluster_id','review_count','avg_rating','impact_rank_score']].to_string(index=False))
    ok=a['impact_rank_score'].notna().all() and abs(a['share_pct'].sum()-100)<2 and c1>0 and c2>0
    print('PASS' if ok else 'FAIL')
if __name__ == "__main__":
    analyze_impact()
    check_impact()