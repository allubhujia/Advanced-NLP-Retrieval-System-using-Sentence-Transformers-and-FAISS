import os
import pickle 
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import torch
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(ROOT_DIR, "data", "processed")
MODEL_SAVE_DIR  = os.path.join(ROOT_DIR, "models", "intent_model")

TRAIN_FILE      = os.path.join(DATA_DIR, "intent_train.csv")
VAL_FILE        = os.path.join(DATA_DIR, "intent_val.csv")
LABEL_ENCODER_PATH = os.path.join(MODEL_SAVE_DIR, "label_encoder.pkl")

MODEL_CHECKPOINT = "distilbert-base-uncased"
NUM_EPOCHS       = 3
BATCH_SIZE       = 16
MAX_SEQ_LENGTH   = 128
LEARNING_RATE    = 2e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
SEED             = 42


def load_data(train_path: str, val_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("Loading datasets....")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    for name,df in [("train", train_df), ("val", val_df)]:
        assert "query" in df.columns, f"[Error] query column missing in {name} set."
        assert "intent" in df.columns, f"[Error] 'intent' column missing in {name} set."
        df.dropna(subset=["query","intent"], inplace=True)
        df["query"] = df["query"].astype(str).str.strip()
        df["intent"] = df["intent"].astype(str).str.strip()

    print(f" Train samples : {len(train_df)}")
    print(f" Val samples : {len(val_df)}")
    print(f" Unique intents : {train_df['intent'].nunique()}")
    print(f" Intent classes : {sorted(train_df['intent'].unique())}\n")

    return train_df, val_df

def preprocess( train_df: pd.DataFrame, val_df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:

    print("Encoding intent labels ...")
    le = LabelEncoder()
    le.fit(train_df["intent"])

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["label"] = le.transform(train_df["intent"])

    # Validation may contain classes unseen in train (edge-case guard)
    unknown_mask = ~val_df["intent"].isin(le.classes_)
    if unknown_mask.any():
        print(f" [WARN] {unknown_mask.sum()} val samples have unknown intents — dropping them.")
        val_df = val_df[~unknown_mask].copy()
    
    val_df["label"] = le.transform(val_df["intent"])

    num_classes = len(le.classes_)
    print(f" Number of classes : {num_classes}")
    print(f" Label mapping : {{c: i for i, c in enumerate(le.classes_)}}\n")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f" LabelEncoder saved -> {LABEL_ENCODER_PATH}\n")

    return train_df, val_df, le

def to_hf_dataset(df: pd.DataFrame) -> Dataset:

    hf_df = df[["query","label"]].rename(columns={"query": "text"})
    return Dataset.from_pandas(hf_df, preserve_index=False)

def build_tokenize_function(tokenizer: DistilBertTokenizerFast):
    def tokenize_function(batch: dict)->dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
    return tokenize_function

def compute_metrics(eval_pred) -> dict:
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    return{
        "accuracy": round(acc,4),
        "f1_weighted": round(f1_weighted, 4),
    }

def train(train_dataset: Dataset, val_dataset: Dataset, num_labels : int,) -> None:
    print(f"Loading tokenizer : {MODEL_CHECKPOINT}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

    print("Tokenizing datasets...")
    tokenize_fn = build_tokenize_function(tokenizer)

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True)

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_val   = tokenized_val.remove_columns(["text"])

    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    print("  Tokenization complete.\n")

    print(f"Loading model: {MODEL_CHECKPOINT} (num_labels={num_labels})")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels = num_labels,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Total parameters : {total_params: ,}\n")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir = MODEL_SAVE_DIR,
        num_train_epochs = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        learning_rate = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
        warmup_ratio = WARMUP_RATIO,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "f1_weighted",
        greater_is_better = True,
        logging_dir = os.path.join(ROOT_DIR,"logs","intent"),
        logging_steps = 50,
        seed = SEED,
        report_to = "none",
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_val,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training ...\n")
    print("="*60)
    trainer.train()
    print("="*60)

    print("\n Running final evaluation on validation set ....")
    metrics = trainer.evaluate()
    print(f"\n  Validation Accuracy   : {metrics.get('eval_accuracy',   'N/A')}")
    print(f"  Validation F1 (W-avg) : {metrics.get('eval_f1_weighted', 'N/A')}\n")

    print(f"Saving model and tokenizer -> {MODEL_SAVE_DIR}")
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print("Saved successfully\n")

class IntentClassifier:
    """
    Handles inference for the trained intent classification model.
    Loads DistilBERT weights and the label encoder for prediction.
    """
    def __init__(self, model_path: str = MODEL_SAVE_DIR, encoder_path: str = LABEL_ENCODER_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(model_path) or not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Intent model artifacts not found at {model_path}. "
                "Please run 'python nlp/intent_training.py' to train the model first."
            )

        print(f"Loading Intent Classifier from {model_path} on {self.device}...")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        with open(encoder_path, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, text: str) -> str:
        """Predict the intent label for a single input text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
        
        return self.le.inverse_transform([pred_idx])[0]


def main() -> None:
    print("\n"+"="*60)
    print("  Intent Classification Training — DistilBERT")
    print("=" * 60 + "\n")

    #1. Load CSV's
    train_df, val_df = load_data(TRAIN_FILE, VAL_FILE)

    #2. Encode labels
    train_df, val_df, le = preprocess(train_df, val_df)
    num_labels = len(le.classes_)

    #3. Convert to HuggingFace Dataset
    train_dataset = to_hf_dataset(train_df)
    val_dataset = to_hf_dataset(val_df)

    #4. Train(Tokenization, Model init, Evaluation, Saving)

    train(train_dataset, val_dataset, num_labels)
    print("Training pipeline complete")
    print(f"Artifacts saved in: {MODEL_SAVE_DIR}")
    print("="*60+"\n")

if __name__ == "__main__":
    main()
    

