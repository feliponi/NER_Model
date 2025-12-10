#!/usr/bin/env python
"""
chat_ui.py — Simple Windows chat-style interface for Key Words NER

This GUI lets you type text messages and immediately see predicted spans
(e.g., key_words) highlighted, using the fine-tuned model you trained.

Usage:
  python chat_ui.py --model-dir outputs/kw-roberta --threshold 0.3

Options:
  --model-dir   Path to the saved model directory (same as train.py --output)
  --max-length  Tokenizer max sequence length (default: 512)
  --threshold   Only highlight spans with confidence >= threshold (default: 0.0)

Notes:
- Requires inference.py in the same folder (or on PYTHONPATH)
- Uses Tkinter (bundled with Python) for a lightweight Windows GUI
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import torch

try:
    from inference import load_model_and_tokenizer, predict_spans
except ImportError:
    message = (
        "Could not import 'inference'. Make sure inference.py is in the same folder "
        "or added to PYTHONPATH."
    )
    raise ImportError(message)


def build_ui(model, tokenizer, device: str, max_length: int, threshold: float):
    root = tk.Tk()
    root.title("Key Words NER — Chat Interface")
    root.geometry("900x600")

    # Top frame: menu
    menubar = tk.Menu(root)

    def save_transcript():
        path = filedialog.asksaveasfilename(
            title="Save transcript",
            defaultextension=".txt",
            filetypes=[["Text", "*.txt"], ["All Files", "*.*"]],
        )
        if not path:
            return
        text = chat.get("1.0", "end-1c")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("Saved", f"Transcript saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear_transcript():
        chat.delete("1.0", "end")

    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="Save transcript...", command=save_transcript)
    file_menu.add_command(label="Clear", command=clear_transcript)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.destroy)
    menubar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menubar)

    # Chat transcript
    chat = ScrolledText(root, wrap="word", font=("Segoe UI", 10))
    chat.pack(fill="both", expand=True, padx=8, pady=8)

    # Tag styles
    chat.tag_config("user", foreground="#1f4cff")
    chat.tag_config("sys", foreground="#008000")
    chat.tag_config("kw", background="#fff3a3", foreground="#000000")  # highlight spans

    # Bottom input frame
    bottom = ttk.Frame(root)
    bottom.pack(fill="x", padx=8, pady=8)

    entry = ttk.Entry(bottom)
    entry.pack(side="left", fill="x", expand=True)
    entry.focus_set()

    send_btn = ttk.Button(bottom, text="Send")
    send_btn.pack(side="left", padx=6)

    status = ttk.Label(bottom, text=f"Device: {device} | threshold: {threshold}")
    status.pack(side="right")

    def append(line: str, tag: str = None):
        chat.insert("end", line + "\n", tag)
        chat.see("end")

    def on_send(event=None):
        text = entry.get().strip()
        if not text:
            return
        entry.delete(0, "end")
        append(f"You: {text}", "user")

        try:
            res = predict_spans([text], model, tokenizer, max_length=max_length, device=device)[0]
        except Exception as e:
            append(f"Error during inference: {e}", None)
            return

        spans = res.get("spans", [])

        # Insert analyzed line and highlight spans
        prefix = "Analyzed: "
        start_idx = chat.index("end")
        chat.insert("end", prefix + text + "\n", "sys")

        for sp in spans:
            if float(sp.get("confidence", 0.0)) < threshold:
                continue
            s = f"{start_idx}+{len(prefix)+sp['start']}c"
            e = f"{start_idx}+{len(prefix)+sp['end']}c"
            chat.tag_add("kw", s, e)

        if spans:
            for sp in spans:
                append(f"  -> [{sp['label']}] ({sp['start']}, {sp['end']}) | conf={sp['confidence']:.3f}", "sys")
        else:
            append("  -> No spans predicted.", "sys")

    send_btn.configure(command=on_send)
    entry.bind("<Return>", on_send)

    return root


def main():
    parser = argparse.ArgumentParser(description="Windows Chat UI for Key Words NER")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max length")
    parser.add_argument("--threshold", type=float, default=0.0, help="Highlight spans with confidence >= threshold")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    model.to(device)

    ui = build_ui(model, tokenizer, device, max_length=args.max_length, threshold=args.threshold)
    ui.mainloop()


if __name__ == "__main__":
    main()
