import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_PATH = os.path.join(ROOT_DIR, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- SENTIMENT DATA ----------
def process_twitter():
    print("Processing Twitter Sentiment Dataset...")

    file_path = os.path.join(RAW_PATH, "twitter_sentiment.csv")

    if not os.path.exists(file_path):
        print("Twitter dataset not found in data/raw/")
        return

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        "text": "query",
        "airline_sentiment": "sentiment"
    })

    df["query"] = df["query"].apply(clean_text)

    train, val = train_test_split(
        df[["query", "sentiment"]],
        test_size=0.1,
        random_state=42,
        stratify=df["sentiment"]
    )

    train.to_csv(os.path.join(PROCESSED_PATH, "sentiment_train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_PATH, "sentiment_val.csv"), index=False)

    print("Twitter sentiment dataset processed")


# ---------- BANKING77 DATA ----------
def process_banking77():
    print("Processing Banking77 Dataset...")

    train_file = os.path.join(RAW_PATH, "banking77_train.csv")
    test_file = os.path.join(RAW_PATH, "banking77_test.csv")

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError("Banking77 CSV files not found in data/raw/")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat([train_df, test_df])

    df = df.rename(columns={
        "text": "query",
        "category": "intent"
    })

    df["query"] = df["query"].apply(clean_text)

    return df[["query", "intent"]]


# ---------- COMPLAINT DATA ----------
def process_complaints():
    print("Processing Complaints Dataset...")

    file_path = os.path.join(RAW_PATH, "complaints.csv")

    if not os.path.exists(file_path):
        print("Complaints dataset not found")
        return pd.DataFrame(columns=["query", "intent"])
        
    df = pd.read_csv(file_path, header=None)

    df.columns = ["tweet_id", "query", "label", "category"]

    df["query"] = df["query"].astype(str).apply(clean_text)

    df["intent"] = "general_complaint"

    return df[["query", "intent"]]


# ---------- FINAL INTENT BUILDER ----------
def build_intent_dataset():
    print("Building Final Intent Dataset...")

    banking_df = process_banking77()
    complaints_df = process_complaints()

    final_df = pd.concat([banking_df, complaints_df], ignore_index=True)

    train, val = train_test_split(
        final_df,
        test_size=0.1,
        random_state=42,
        stratify=final_df["intent"]
    )

    train.to_csv(os.path.join(PROCESSED_PATH, "intent_train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_PATH, "intent_val.csv"), index=False)

    print("Intent dataset ready")


if __name__ == "__main__":
    print("Starting data processing pipeline...\n")

    process_twitter()
    build_intent_dataset()

    print("\nALL DATA PROCESSING COMPLETED SUCCESSFULLY")