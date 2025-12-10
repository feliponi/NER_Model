
### Overview

This project trains a **token classification (NER)** model (default: **RoBERTa‑base**) to tag tokens that belong to a single label: `key_words`. The training script reads a plain **text file** where **each line is one example**, and it treats the **entire line** as the entity span labeled `key_words`. The model is trained with a BIO scheme (`B-key_words`, `I-key_words`, `O`).

***

## 1) What’s in this repo

*   **`train.py`** – CLI training script that:
    *   Converts each line of the input text file into a NER example where the entity spans from **char 0 to the end of the line** and is labeled `key_words`.
    *   Builds BIO labels, aligns them to tokens via offset mappings, and trains/evaluates with Hugging Face’s `Trainer`.
    *   Saves the model, tokenizer, and a `test_metrics.json` file in the output directory.

*   **`utils.py`** – Convenience utilities (only `save_json` is used by `train.py` to write metrics). It also contains helpers for CSV/JSON IO and dataset inspection if you later work with JSON‑formatted NER data.

***

## 2) Quick start (5 minutes)

### Prerequisites

*   **Python 3.9+**
*   A machine with **GPU (CUDA)** is recommended but optional; the script will use GPU automatically if available.

### 2.1 Create and activate a virtual environment

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 2.2 Install dependencies

```bash
pip install torch datasets transformers scikit-learn numpy
```

> If you have a CUDA‑capable GPU, install the corresponding `torch` build for your CUDA version.

### 2.3 Prepare your training data

Create a text file (e.g., `data/train.txt`) with **one example per line**:

```text
Graph neural networks for document classification
Efficient fine-tuning for domain-specific terminology
Named entity recognition for key words extraction
```

> In this setup, **each line is treated as the sole entity** (from char index 0 to the end) labeled `key_words`.

### 2.4 Run training

```bash
python train.py \
  --input data/train.txt \
  --output outputs/kw-roberta \
  --model-name roberta-base \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --test-size 0.1 \
  --val-size 0.1
```

*   The script reports whether it’s using **GPU or CPU**, splits the dataset (train/val/test) with defaults `test_size=0.1`, `val_size=0.1`, and starts training.
*   Checkpoints and logs are written under your `--output` directory, with evaluation and saving done **each epoch**; it keeps only the **two most recent checkpoints** and loads the **best model** at the end using `macro_f1`.

***

## 3) What the script does (under the hood)

### 3.1 Data preparation

*   **Input**: a plain text file, read line‑by‑line. Empty lines are skipped. 
*   **Entity creation**: for each line `text`, it sets `start=0`, `end=len(text)`, and `label="key_words"`. The script builds an example:
    ```json
    {"text": "...", "entities": [[0, len(text), "key_words"]]}
    ```
     
*   **PyArrow compatibility**: It converts each entity element to **string** before building HF Datasets (later cast back to `int` during label conversion). 

### 3.2 Labels and BIO scheme

*   Label map: `{"O":0, "B-key_words":1, "I-key_words":2}` (order built programmatically). 
*   For each char position in the text, the script assigns `B` at `start` and `I` for the remaining positions up to `end`. 

### 3.3 Tokenization & alignment

*   Tokenization uses the **RoBERTa tokenizer** with `add_prefix_space=True`, `max_length=512`, and offset mappings. 
*   For each token, it looks up the char label at the **token’s start offset** and assigns that label; special tokens get label `-100` (ignored in loss/metrics). 

### 3.4 Training configuration

*   **Model**: `AutoModelForTokenClassification.from_pretrained(model_name)` with the custom `label2id/id2label` maps. Default `model_name` is **`roberta-base`**.
*   **Arguments** (key ones):
    *   `eval_strategy="epoch"`, `save_strategy="epoch"`, `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="macro_f1"`.
    *   `per_device_train_batch_size`, `per_device_eval_batch_size` = `--batch-size`. 
    *   `num_train_epochs = --epochs`, `learning_rate = --learning-rate`, `weight_decay=0.01`, `logging_steps=50`.
*   **Collation**: `DataCollatorForTokenClassification` with the tokenizer.

### 3.5 Metrics

*   Metrics computed on `B-key_words` and `I-key_words` only (macro precision, recall, F1), ignoring `O` and masked `-100`. Results are saved to `test_metrics.json`.

***

## 4) Command‑line options

| Flag              | Description                                                        | Default                                                                                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--input`         | Path to **text file** (one sample per line).                       | **Required**    |
| `--output`        | Output directory for model/tokenizer/checkpoints/metrics.          | **Required**    |
| `--model-name`    | HF model name/path (e.g., `roberta-base`, `distilroberta-base`).   | `roberta-base`  |
| `--epochs`        | Training epochs.                                                   | `3`             |
| `--batch-size`    | Train/eval batch size per device.                                  | `16`            |
| `--learning-rate` | Optimizer learning rate.                                           | `2e-5`          |
| `--test-size`     | Test split proportion.                                             | `0.1`           |
| `--val-size`      | Validation split proportion (applied on the remaining train data). | `0.1`           |

***

## 5) After training: where to find results

*   **Model & tokenizer**: in `--output` (e.g., `outputs/kw-roberta`). 
*   **Metrics**: `--output/test_metrics.json` (macro P/R/F1 for `key_words`). 
*   **Logs/checkpoints**: in `--output/logs` and under the output directory; only 2 latest checkpoints are kept. 

***

## 6) Simple inference example

Here’s a minimal script to run predictions on raw text using the saved model:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_dir = "outputs/kw-roberta"  # your --output dir
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
id2label = model.config.id2label

text = "Efficient fine-tuning for domain-specific terminology"
enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
with torch.no_grad():
    logits = model(**{k: v for k, v in enc.items() if k != "offset_mapping"}).logits
pred_ids = logits.argmax(-1)[0].tolist()
offsets = enc["offset_mapping"][0].tolist()

tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())

# Print tokens tagged as B-key_words / I-key_words
for tok, (start, end), pid in zip(tokens, offsets, pred_ids):
    if start == end:
        continue  # special token
    label = id2label[pid]
    if label != "O":
        print(f"{tok}\t{label}\t({start}, {end})")
```

> The training saved `id2label/label2id` into the model config; the example above reads them back from `model.config`.

***

## 7) Customizing the label or model

*   **Change the label name**: Update `TARGET_LABEL = "key_words"` in `train.py` if you want a different label. The script automatically builds `B-<LABEL>` and `I-<LABEL>`.
*   **Try other backbones**: Set `--model-name` to any compatible token‑classification backbone (e.g., `distilroberta-base`). The script adapts the classification head to the number of labels. 
*   **Sequence length**: Tokens are truncated to `max_length=512`. If your lines are longer, consider pre‑splitting or a longer backbone with sliding windows.

***

## 8) Tips & troubleshooting

*   **GPU not found**: The script logs `GPU Device: cpu` when CUDA isn’t available; training still works but is slower.
*   **PyArrow / type issues**: If you later move to JSON datasets, ensure entity spans are valid ints; the current script converts them to strings before creating HF Datasets and casts back during label building.
*   **Imbalanced / trivial labels**: Because the entire line is marked as the entity, many tokens will be `B/I` and few `O`. For more realistic NER, consider providing **exact spans** for one or more entities per line and extending the label map accordingly. 
*   **Checkpoints**: Only the two latest checkpoints are kept (`save_total_limit=2`). If you want all epochs, increase this limit.
*   **Logging**: Training logs are saved under `--output/logs`, with `logging_steps=50`.

***

## 9) Optional utilities (`utils.py`)

If you move to JSON/CSV datasets, `utils.py` includes handy functions:

*   `load_json(path)` / `save_json(data, path)` – read/write JSON.
*   `load_csv(path)` / `save_csv(df, path)` – read/write CSV via pandas.
*   `validate_dataset_format(data, required_keys)` – quick sanity check for your dataset’s schema.
*   `analyze_label_distribution(labeled_data)` + `print_label_stats(stats)` – basic stats on NER entities (counts, lengths, averages).

***

## 10) Reproducibility notes

*   Deterministic train/val/test splits use `random_state=42`. 
*   Best model is selected by `macro_f1` at the end of training. 

***

## 11) License & attribution

This project uses the **Hugging Face Transformers & Datasets** libraries and the **RoBERTa** tokenizer/model by default. See their respective licenses if you plan to distribute models.

***

### Example “just works” command

```bash
python train.py \
  --input data/train.txt \
  --output outputs/kw-roberta \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 2e-5
```

# INFERENCE — How to run predictions

## 1) What you need

- **Python 3.9+**
- Dependencies (install in your virtualenv):
  ```bash
  pip install torch transformers
  ```
- A trained model directory produced by `train.py` (e.g., `outputs/kw-roberta`).

> GPU is optional. The script automatically uses CUDA if available; otherwise it runs on CPU.

---

## 2) Script entry points

The `inference.py` script supports:

- **Single text input** via `--text`
- **Batch input from file** via `--file` (one example per line)
- Optional JSON output via `--output`

### 2.1 Single text
```bash
python inference.py \
  --model-dir outputs/kw-roberta \
  --text "Efficient fine-tuning for domain-specific terminology"
```

### 2.2 From a file (batch)
```bash
python inference.py \
  --model-dir outputs/kw-roberta \
  --file data/test.txt \
  --output predictions.json
```

Where `data/test.txt` contains one text per line.

---

## 3) Command-line options

- `--model-dir` **(required)**: Path to the saved model directory created by `train.py`.
- `--text` **(mutually exclusive with `--file`)**: A single input text to analyze.
- `--file` **(mutually exclusive with `--text`)**: A UTF-8 text file with **one example per line**.
- `--output` *(optional)*: If provided, saves predictions to this JSON file.
- `--max-length` *(default: 512)*: Tokenizer max sequence length (must match or be compatible with training).

---

## 4) What the output looks like

The script prints a human-readable summary to the console and, if `--output` is used, writes a JSON array where each item corresponds to an input example:

```json
[
  {
    "text": "Efficient fine-tuning for domain-specific terminology",
    "spans": [
      {
        "label": "key_words",
        "start": 0,
        "end": 56,
        "text": "Efficient fine-tuning for domain-specific terminology",
        "confidence": 0.94
      }
    ],
    "tokens": [
      {"start": 0, "end": 8,  "label": "B-key_words", "confidence": 0.97},
      {"start": 9, "end": 19, "label": "I-key_words", "confidence": 0.95}
      // ... One entry per non-special token
    ]
  }
]
```

- **spans**: Aggregated from consecutive `B-`/`I-` predictions; `start`/`end` are **character** indices relative to the original text.
- **tokens**: Per-token predictions (useful for debugging). Special tokens are omitted.

> Confidence is the average probability over the tokens comprising a span.

---

## 5) Programmatic usage (optional)

You can call the functions directly from Python if `inference.py` is on your `PYTHONPATH`:

```python
from inference import load_model_and_tokenizer, predict_spans

model_dir = "outputs/kw-roberta"
model, tokenizer = load_model_and_tokenizer(model_dir)
texts = [
    "Graph neural networks for document classification",
    "Named entity recognition for key words extraction",
]
results = predict_spans(texts, model, tokenizer)
for r in results:
    print(r["text"], r["spans"])  # list of {label, start, end, text, confidence}
```

---

## 6) Tips & troubleshooting

- **No spans predicted**: This can happen if the model is undertrained or the text is out-of-domain. Try more epochs or domain-specific data.
- **Wrong `--model-dir`**: If the tokenizer/model can’t be loaded, verify that `--model-dir` is the same folder you passed to `--output` during training and that it contains `config.json`, `pytorch_model.bin`, and tokenizer files.
- **Long inputs**: Texts longer than `--max-length` are truncated. Consider pre-splitting long texts or increasing `--max-length` if your backbone supports it.
- **Large files**: The script loads all lines into memory for batch mode. For very large datasets, split the file or adapt the script to stream in chunks.

---

## 7) Reproducing training assumptions

- Tokenizer: RoBERTa (or compatible) with `add_prefix_space=True`.
- Labels: BIO scheme derived from the single target label `key_words`.
- Alignment: Character-level labels mapped to tokens using offset mappings; special tokens are ignored.

That’s it — you’re set to run predictions and inspect results.
