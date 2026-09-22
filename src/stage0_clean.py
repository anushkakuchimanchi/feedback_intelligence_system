# src/stage0_clean.py
import pandas as pd
from utils import call_llm_json

def basic_clean(df):
    df = df.dropna(subset=['review_text'])
    df = df[df['review_text'].str.split().str.len() > 3]
    df = df.drop_duplicates(subset=['review_text'])
    df = df.reset_index(drop=True)
    df['review_id'] = df.index
    return df

def llm_filter_review(review_text):
    prompt = f"""Review: "{review_text}"

Is this a genuine NEGATIVE review of the app?
Return true if the review contains a complaint, problem, criticism, dissatisfaction,
bug, missing feature, poor performance, or other negative experience.
Return false if it is positive, neutral, spam, gibberish, or too vague.

Return JSON exactly like this: {{"is_negative": true, "reason": "short reason"}}"""
    try:
        result = call_llm_json(prompt, max_tokens=500)
        return result.get("is_negative", False)
    except Exception as e:
        print(f"Skipping review due to error: {e}")
        return False

def clean_reviews():
    df = pd.read_csv('../data/raw/termux_reviews.csv')
    df = basic_clean(df)
    print(f"After basic clean: {df.shape[0]} reviews")

    valid_flags = []
    for i, text in enumerate(df['review_text']):
        valid_flags.append(llm_filter_review(text))
        if i % 50 == 0:
            print(f"Checked {i}/{len(df)}")

    df['is_negative'] = valid_flags
    df = df[df['is_negative'] == True].drop(columns=['is_negative'])
    df = df.reset_index(drop=True)

    df.to_csv('../data/processed/clean_reviews.csv', index=False)
    print(f"Final cleaned dataset: {df.shape[0]} reviews remain")
    return df

if __name__ == "__main__":
    clean_reviews()