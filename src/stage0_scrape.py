from google_play_scraper import reviews, Sort
import pandas as pd

def scrape_reviews():
    result, _ = reviews(
        'com.termux',
        lang='en',
        country='us',
        sort=Sort.NEWEST,
        count=10000
    )
    df = pd.DataFrame(result)
    df = df[['content', 'score', 'at']]
    df.columns = ['review_text', 'rating', 'date']
    df.to_csv('data/raw/termux_reviews.csv', index=False)
    print(f"Saved {df.shape[0]} reviews")
    return df

if __name__ == "__main__":
    scrape_reviews()