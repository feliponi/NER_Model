## Option A — Local Windows Desktop (Tkinter)

### 1) Prereqs (one‑time)

```powershell
# From your project folder on Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Core deps (same as training/inference)
pip install torch transformers
```

> If you haven’t installed `datasets`/`scikit-learn` before, they’re only needed for training, not for this UI.

### 2) Make sure files are in place

Put **`chat_ui.py`** and **`inference.py`** in the same folder as your saved model directory (e.g., `outputs/kw-roberta` created by `train.py`).

### 3) Run the chat UI

```powershell
python chat_ui.py --model-dir outputs/kw-roberta --threshold 0.30
```

*   **Type** into the message box and press **Enter** or click **Send**.
*   The UI prints your message, then an **“Analyzed:”** line with the same text and highlights the **predicted spans** (yellow).
*   Below that, it shows a short **span summary** (start/end character indices + confidence).
*   Use the **File → Save transcript…** menu to export the whole session to a `.txt` file.

> `--threshold` (default `0.0`) lets you hide low‑confidence spans (e.g., `0.30` means only highlight predictions with confidence ≥ 0.30).  
> `--max-length` matches training truncation (default `512`).

### 4) What the UI code is doing

*   Loads your fine‑tuned model and tokenizer using **`inference.py`** helpers.
*   For each message, calls `predict_spans(... )` which:
    *   Tokenizes with **offset mappings** (same as training).
    *   Gets per‑token labels (`B/I/O`) and confidences.
    *   Groups consecutive `B`/`I` tokens into **character spans** and computes **average confidence**.
*   In Tkinter’s `ScrolledText`, it inserts the text and **adds a tag** over each span to draw the yellow highlight.
*   Special tokens (where offset start==end) are skipped—consistent with training/inference logic.

***

## Option B — Turn it into a click‑to‑run Windows app (optional)

If you want to give your colleague a single `.exe` without Python:

```powershell
pip install pyinstaller
pyinstaller -F -w chat_ui.py
```

*   `-F` creates a single-file executable.
*   `-w` hides the console window (GUI only).
*   Put the **model folder** (e.g., `outputs/kw-roberta`) next to the executable, and the app will load it at runtime via `--model-dir`.

> Note: First run on another machine might take a moment to cache the model files; bundling the model inside the executable is possible but increases size substantially.

***

## Option C — Browser UI (Flask) or Teams/Slack bot (optional)

If you’d prefer a browser‑based UI (Flask) or a lightweight bot that posts highlighted text into a channel, I can scaffold that next. It’s a bit more setup (HTTP server + front‑end for highlighting, or a bot registration for Teams/Slack), but it’s very doable. Just tell me which route you want.

***

## Quick checks / common pitfalls

*   **Model dir mismatch:** Use the **same directory** you passed as `--output` in `train.py` (it must contain `config.json`, `pytorch_model.bin`, tokenizer files).
*   **CPU vs GPU:** The UI auto‑detects CUDA. If no GPU, it logs **Device: cpu**—still fine, just slower.
*   **Very long inputs:** Messages beyond `512` chars will be truncated (default). You can raise `--max-length`, but RoBERTa supports up to 512 max; for longer texts, consider splitting into chunks.

***

## Want the UI to show only the detected key words?

We can add a toggle to display **just the extracted spans** (one per line) or to **copy them to clipboard**. Also, if you plan to share broadly, I can add:

*   A **confidence slider** (0–1).
*   A **light/dark theme** toggle.
*   A **JSON export** of predictions per message.

***

### Fast path: one command to test now

```powershell
python chat_ui.py --model-dir outputs/kw-roberta --threshold 0.25
```
