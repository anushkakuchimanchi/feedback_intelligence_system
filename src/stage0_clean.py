# src/stage0_clean.py
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import time
import json
import os
from utils import call_llm
import os
if os.path.exists('data/processed/clean_reviews_partial.csv'):
    os.remove('data/processed/clean_reviews_partial.csv')
BATCH_SIZE = 25          # fewer, bigger batches = fewer total requests, stays under 500/day easily
SLEEP_BETWEEN_BATCHES = 5  # 15 req/min limit = 1 every 4s minimum; 5s gives safety margin

def basic_clean(df):
    df = df.dropna(subset=['review_text'])
    df = df[df['review_text'].str.split().str.len() > 3]
    df = df.drop_duplicates(subset=['review_text'])
    df = df.reset_index(drop=True)
    df['review_id'] = df.index
    return df

def llm_filter_batch(review_texts, retries=3):
    numbered_reviews = "\n".join([f"{i}: \"{text}\"" for i, text in enumerate(review_texts)])
    prompt = f"""Here are {len(review_texts)} app reviews, numbered 0 to {len(review_texts)-1}:

{numbered_reviews}

For EACH review, decide if it is a genuine NEGATIVE review — a complaint, problem,
criticism, bug, missing feature, or poor performance. False if positive, neutral, spam, or vague.

Return ONLY a JSON array: [{{"index": 0, "is_negative": true}}, ...]"""

    for attempt in range(retries):
        try:
            text = call_llm(prompt, max_tokens=1500)
            text = text.strip().replace("```json", "").replace("```", "")
            results = json.loads(text)
            flags = [False] * len(review_texts)
            for r in results:
                idx = r.get("index")
                if idx is not None and 0 <= idx < len(review_texts):
                    flags[idx] = r.get("is_negative", False)
            return flags
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                wait = (attempt + 1) * 15
                print(f"  Server overloaded (503) — waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                print(f"  Batch error: {e}")
                return None
    print("  Gave up after retries on this batch.")
    return None

def clean_reviews():
    df = pd.read_csv('../data/raw/termux_reviews.csv')
    df = basic_clean(df)
    total = len(df)
    print(f"Total reviews to process: {total}")

    partial_path = '../data/processed/clean_reviews_partial.csv'
    if os.path.exists(partial_path):
        partial = pd.read_csv(partial_path)
        start_from = len(partial)
        all_flags = partial['is_negative'].tolist()
        print(f"Resuming from review {start_from}/{total}")
    else:
        start_from = 0
        all_flags = []

    for start in range(start_from, total, BATCH_SIZE):
        batch_texts = df['review_text'].iloc[start:start + BATCH_SIZE].tolist()
        flags = llm_filter_batch(batch_texts)

        if flags is None:
            print(f"Stopped at review {start} due to an error. Just rerun this script — it resumes here automatically.")
            return

        all_flags.extend(flags)
        print(f"Processed {min(start + BATCH_SIZE, total)}/{total}")

        # save after EVERY batch — never lose more than one batch of progress
        partial_df = df.iloc[:len(all_flags)].copy()
        partial_df['is_negative'] = all_flags
        partial_df.to_csv(partial_path, index=False)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # all done — build final filtered file
    final_df = df.copy()
    final_df['is_negative'] = all_flags
    final_df = final_df[final_df['is_negative'] == True].drop(columns=['is_negative'])
    final_df = final_df.reset_index(drop=True)
    final_df.to_csv('../data/processed/clean_reviews.csv', index=False)
    print(f"DONE. {final_df.shape[0]} negative reviews saved to clean_reviews.csv")

if __name__ == "__main__":
    clean_reviews()