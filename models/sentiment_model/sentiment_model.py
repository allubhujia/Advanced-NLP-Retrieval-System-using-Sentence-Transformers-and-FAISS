from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, List, Dict


class SentimentModel:
    """
    Transformer-based sentiment classifier (Positive / Negative / Neutral).
    Supports HuggingFace pipeline mode and fine-tuned model mode.
    Outputs sentiment label + softmax confidence score.
    """

    LABEL_MAP = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        "POSITIVE": "Positive",
        "NEGATIVE": "Negative",
        "NEUTRAL":  "Neutral",
    }

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str = "cpu",
        use_pipeline: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize the sentiment model.

        Args:
            model_name:    HuggingFace model identifier or local path.
            device:        'cpu' or 'cuda'.
            use_pipeline:  If True, use HuggingFace pipeline (simpler).
                           If False, load tokenizer + model manually (more control).
            max_length:    Max token length for truncation.
        """
        self.model_name = model_name
        self.device = device
        self.use_pipeline = use_pipeline
        self.max_length = max_length
        self._device_index = 0 if device == "cuda" and torch.cuda.is_available() else -1

        if use_pipeline:
            self._pipeline = pipeline(
                task="text-classification",
                model=model_name,
                device=self._device_index,
                top_k=None,          # return all class scores
                truncation=True,
                max_length=max_length,
            )
            self._tokenizer = None
            self._model = None
        else:
            self._pipeline = None
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.to(device)
            self._model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_label(self, raw_label: str) -> str:
        """Map raw HuggingFace label strings to Positive / Negative / Neutral."""
        return self.LABEL_MAP.get(raw_label.upper(), raw_label.capitalize())

    def _parse_pipeline_output(self, scores: List[Dict]) -> Dict:
        """
        Convert pipeline's list-of-dicts into a clean prediction dict.
        scores: [{'label': 'LABEL_0', 'score': 0.02}, ...]
        """
        best = max(scores, key=lambda x: x["score"])
        return {
            "label":      self._normalize_label(best["label"]),
            "confidence": round(float(best["score"]), 4),
            "scores": {
                self._normalize_label(s["label"]): round(float(s["score"]), 4)
                for s in scores
            },
        }

    def _infer_manual(self, texts: List[str]) -> List[Dict]:
        """Run inference using raw tokenizer + model (non-pipeline path)."""
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**encoded).logits          # (B, num_labels)

        probs = F.softmax(logits, dim=-1).cpu().numpy()     # (B, num_labels)
        id2label = self._model.config.id2label              # {0: 'LABEL_0', ...}

        results = []
        for row in probs:
            label_scores = {
                self._normalize_label(id2label[i]): round(float(p), 4)
                for i, p in enumerate(row)
            }
            best_label = max(label_scores, key=label_scores.get)
            results.append({
                "label":      best_label,
                "confidence": label_scores[best_label],
                "scores":     label_scores,
            })
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict:
        """
        Classify sentiment of a single text string.

        Args:
            text: Raw input text.

        Returns:
            {
              "label":      "Positive" | "Negative" | "Neutral",
              "confidence": float,          # softmax probability of top label
              "scores":     {               # full distribution
                  "Positive": float,
                  "Negative": float,
                  "Neutral":  float,
              }
            }
        """
        if self.use_pipeline:
            raw = self._pipeline(text)[0]   # list of score dicts for one input
            return self._parse_pipeline_output(raw)
        else:
            return self._infer_manual([text])[0]

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Classify sentiment for a list of texts.

        Args:
            texts:      List of raw input texts.
            batch_size: Number of texts per forward pass.

        Returns:
            List of prediction dicts (same schema as predict()).
        """
        results = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]

            if self.use_pipeline:
                raw_batch = self._pipeline(chunk)   # list[list[dict]]
                for raw in raw_batch:
                    results.append(self._parse_pipeline_output(raw))
            else:
                results.extend(self._infer_manual(chunk))

        return results

    def predict_label(self, text: str) -> str:
        """Convenience method — returns only the top label string."""
        return self.predict(text)["label"]

    def predict_score(self, text: str) -> float:
        """Convenience method — returns only the confidence score."""
        return self.predict(text)["confidence"]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "pipeline" if self.use_pipeline else "manual"
        return (
            f"SentimentModel(model='{self.model_name}', "
            f"device='{self.device}', mode='{mode}')"
        )