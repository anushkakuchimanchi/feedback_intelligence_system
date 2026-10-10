# src/agents/feedback_analysis_agent.py
import pandas as pd
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(SRC_DIR))
from utils import call_llm_json

BATCH_SIZE = 15          # smaller than stage0 since each review now returns more fields
SLEEP_BETWEEN_BATCHES = 15
MAX_CHARS = 350

CATEGORIES = ["bug", "feature_request", "usability", "performance", "general"]

def classify_batch(review_texts):
    numbered = "\n".join(f"{i}: {t[:MAX_CHARS]}" for i, t in enumerate(review_texts))
    prompt = f"""Below are {len(review_texts)} app reviews of Termux, numbered 0 to {len(review_texts)-1}.

For EACH review, extract:
- category: one of {CATEGORIES}
- area: short phrase for the affected feature (e.g. "SSH", "package install", "UI", "networking")
- severity: "low", "medium", or "high"
- issue: one short sentence summarizing the actual problem

{numbered}

Return ONLY a JSON object like this, one entry per review index:
{{"0": {{"category": "bug", "area": "SSH", "severity": "high", "issue": "SSH stops working after Android update"}}, "1": {{...}}}}"""
    try:
        result = call_llm_json(prompt, max_tokens=1500)
        return result
    except Exception as e:
        print(f"  Stopped: {e}")
        return None

def classify_reviews():
    df = pd.read_csv(PROCESSED_DIR / "clean_reviews.csv")
    total = len(df)
    print(f"Total negative reviews to classify: {total}")

    partial_path = PROCESSED_DIR / "classified_reviews_partial.csv"
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        start_from = len(partial)
        results = partial.to_dict('records')
        print(f"Resuming from review {start_from}/{total}")
    else:
        start_from = 0
        results = []

    for start in range(start_from, total, BATCH_SIZE):
        batch = df['review_text'].iloc[start:start + BATCH_SIZE].tolist()
        batch_ids = df['review_id'].iloc[start:start + BATCH_SIZE].tolist()
        classified = classify_batch(batch)

        if classified is None:
            print(f"Paused at review {start}. Rerun the script to resume.")
            partial_df = pd.DataFrame(results)
            partial_df.to_csv(partial_path, index=False)
            return results

        for i, rid in enumerate(batch_ids):
            info = classified.get(str(i), {})
            results.append({
                "review_id": rid,
                "review_text": batch[i],
                "category": info.get("category", "general"),
                "area": info.get("area", "unknown"),
                "severity": info.get("severity", "medium"),
                "issue": info.get("issue", batch[i][:100]),
            })

        print(f"Processed {len(results)}/{total}")
        pd.DataFrame(results).to_csv(partial_path, index=False)
        time.sleep(SLEEP_BETWEEN_BATCHES)

    final = df.merge(pd.DataFrame(results), on=["review_id", "review_text"])
    final.to_csv(PROCESSED_DIR / "classified_reviews.csv", index=False)
    print(f"DONE. {final.shape[0]} reviews classified -> classified_reviews.csv")
    return results

if __name__ == "__main__":
    results = classify_reviews()
    total = len(results)
    parse_failures = sum(1 for r in results if r['category'] == 'general' and r['area'] == 'unknown')
    success_rate = (total - parse_failures) / total * 100

    print(f"\n--- Stage 1 Metrics ---")
    print(f"Total classified: {total}")
    print(f"Parse/fallback failures: {parse_failures}")
    print(f"Successful extraction rate: {success_rate:.1f}%")
    print(f"Category distribution:\n{pd.DataFrame(results)['category'].value_counts()}")
    print(f"Severity distribution:\n{pd.DataFrame(results)['severity'].value_counts()}")