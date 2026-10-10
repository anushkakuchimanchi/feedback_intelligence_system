# src/agents/impact_analysis_agent.py
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED = os.path.join(ROOT, 'data', 'processed')
RAW = os.path.join(ROOT, 'data', 'raw', 'termux_reviews.csv')

def analyze_impact():
    df = pd.read_csv(os.path.join(PROCESSED, 'clustered_reviews.csv'))
    df['date'] = pd.to_datetime(df['date'])
    total = len(df)

    app_rating = pd.read_csv(RAW)['rating'].mean()      # the app's overall rating, from ALL scraped reviews
    cutoff = df['date'].max() - pd.Timedelta(days=90)
    overall_recent = (df['date'] >= cutoff).mean()

    rows = []
    for cid, g in df.groupby('cluster_id'):
        centroid = g[g['is_centroid']].iloc[0]
        recent = (g['date'] >= cutoff).mean()
        rows.append({
            'cluster_id': cid,
            'review_count': len(g),
            'share_pct': round(len(g) / total * 100, 1),
            'avg_rating': round(g['rating'].mean(), 2),
            'rating_gap_vs_app': round(g['rating'].mean() - app_rating, 2),
            'pct_high_severity': round((g['severity'] == 'high').mean() * 100, 1),
            'top_category': g['category'].mode()[0],
            'first_seen': g['date'].min().date(),
            'last_seen': g['date'].max().date(),
            'recent_90d_pct': round(recent * 100, 1),
            'trend_ratio': round(recent / overall_recent, 2) if overall_recent else None,
            'centroid_issue': centroid['issue'],
            'centroid_review': centroid['review_text'],
        })

    out = pd.DataFrame(rows).sort_values('review_count', ascending=False)
    out.to_csv(os.path.join(PROCESSED, 'cluster_stats.csv'), index=False)

    print(f"App overall rating (all scraped reviews): {app_rating:.2f}\n")
    print(out.drop(columns=['centroid_review', 'centroid_issue']).to_string(index=False))

if __name__ == "__main__":
    analyze_impact()