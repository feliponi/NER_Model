
#!/usr/bin/env python
"""
Inference script for the Key Words NER (Token Classification) model.

- Loads a fine-tuned token classification model (e.g., RoBERTa) from --model-dir
- Accepts either a single --text or a --file with one line per example
- Outputs predicted spans labeled via BIO tags by grouping consecutive B-/I- tokens
- Optionally saves results to JSON

Usage examples:

python inference.py --model-dir outputs/kw-roberta --text "Efficient fine-tuning for domain-specific terminology"
python inference.py --model-dir outputs/kw-roberta --file data/test.txt --output predictions.json

Notes:
- Matches training tokenization using add_prefix_space=True for RoBERTa-like models
- Special tokens (offset start==end) are ignored during aggregation
"""

import argparse
import json
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_dir: str):
    """Load model and tokenizer from a directory saved by train.py."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_spans(texts: List[str], model, tokenizer, max_length: int = 512, device: str = None) -> List[Dict[str, Any]]:
    """Run token classification and aggregate B-/I- labels into character-level spans.

    Returns a list of dicts, one per input text, each containing:
      - text: the original text
      - spans: list of {label, start, end, text, confidence}
      - tokens: list of per-token predictions (optional debug info)
    """
    if device is None:
        device = _device()

    enc = tokenizer(
        texts,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = enc.pop("offset_mapping")  # keep offsets on CPU
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits  # (batch, seq_len, num_labels)
        probs = F.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)

    id2label = model.config.id2label

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(texts):
        seq_offsets = offsets[i].tolist()
        seq_pred_ids = pred_ids[i].tolist()
        seq_probs = probs[i].tolist()

        # Collect per-token info (skip special tokens where start==end)
        tokens_info = []
        for (start, end), pid, pvec in zip(seq_offsets, seq_pred_ids, seq_probs):
            if start == end:
                continue  # special token
            label = id2label.get(pid, "O")
            confidence = float(max(pvec))  # probability of predicted label
            tokens_info.append({
                "start": start,
                "end": end,
                "label": label,
                "confidence": confidence,
            })

        # Aggregate into spans: consecutive B-/I- tokens form a span
        spans = []
        current = None  # {label, start, end, confidences: []}

        def close_current():
            if current is not None:
                start = current["start"]
                end = current["end"]
                span_text = text[start:end]
                confs = current.get("confidences", [])
                avg_conf = float(sum(confs) / max(len(confs), 1))
                spans.append({
                    "label": current["label"],
                    "start": start,
                    "end": end,
                    "text": span_text,
                    "confidence": avg_conf,
                })

        for tok in tokens_info:
            label = tok["label"]
            if label.startswith("B-"):
                # Start a new span
                close_current()
                current = {
                    "label": label.split("B-", 1)[1],
                    "start": tok["start"],
                    "end": tok["end"],
                    "confidences": [tok["confidence"]],
                }
            elif label.startswith("I-"):
                if current is None:
                    # Handle I- without a preceding B- as a new span
                    current = {
                        "label": label.split("I-", 1)[1],
                        "start": tok["start"],
                        "end": tok["end"],
                        "confidences": [tok["confidence"]],
                    }
                else:
                    # Extend current span
                    current["end"] = tok["end"]
                    current["confidences"].append(tok["confidence"])
            else:
                # O label: close any active span
                close_current()
                current = None
        # Close trailing span
        close_current()

        results.append({
            "text": text,
            "spans": spans,
            "tokens": tokens_info,
        })

    return results


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def main():
    parser = argparse.ArgumentParser(description="Inference for Key Words NER (Token Classification)")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to saved model directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Single text input for prediction")
    group.add_argument("--file", type=str, help="Path to a text file (one example per line)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save JSON predictions")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max length")

    args = parser.parse_args()

    device = _device()
    print(f"Device: {device}")

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    model.to(device)

    if args.text is not None:
        texts = [args.text]
    else:
        texts = read_lines(args.file)
        if not texts:
            print("No non-empty lines found in the provided file.")
            return

    results = predict_spans(texts, model, tokenizer, max_length=args.max_length, device=device)

    # Print human-readable output
    for i, res in enumerate(results):
        print("\n" + "=" * 80)
        print(f"Example {i+1}:")
        print(res["text"])
        if res["spans"]:
            print("\nPredicted spans:")
            for sp in res["spans"]:
                print(f" - [{sp['label']}] ({sp['start']}, {sp['end']}) | conf={sp['confidence']:.3f} | \"{sp['text']}\"")
        else:
            print("\nNo spans predicted.")

    # Optionally save JSON
    if args.output:
        from pathlib import Path
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved predictions to {out_path}")


if __name__ == "__main__":
    main()
