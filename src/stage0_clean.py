import pandas as pd

def clean_reviews():
    df = pd.read_csv('data/raw/termux_reviews.csv')

    # drop empty/null reviews
    df = df.dropna(subset=['review_text'])

    # drop very short/junk reviews (less than ~4 words = not useful)
    df = df[df['review_text'].str.split().str.len() > 3]

    # drop exact duplicate reviews
    df = df.drop_duplicates(subset=['review_text'])

    # reset index, add a clean review_id
    df = df.reset_index(drop=True)
    df['review_id'] = df.index

    df.to_csv('data/processed/clean_reviews.csv', index=False)
    print(f"Cleaned dataset: {df.shape[0]} reviews remain")
    return df

if __name__ == "__main__":
    clean_reviews()