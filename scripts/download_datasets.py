import os
import pandas as pd
from datasets import load_dataset

os.makedirs("data/raw", exist_ok=True)

print("Downloading Twitter dataset...")
twitter = load_dataset("osanseviero/twitter-airline-sentiment")
twitter["train"].to_csv("data/raw/twitter_sentiment.csv")

print("Downloading complaints dataset...")
url = "https://raw.githubusercontent.com/danielpreotiuc/complaints-social-media/master/complaints-data.csv"
df = pd.read_csv(url)
df.to_csv("data/raw/complaints.csv", index=False)

print("All datasets downloaded!")