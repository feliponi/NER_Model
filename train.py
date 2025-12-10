import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# import utils functions from utils import load_json, save_json
try:
    from utils import load_json, save_json
except ImportError:
    
    import json
    def load_json(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    def save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- CA key words labels ---
TARGET_LABEL = "key_words"

#-- Function to load text file and create NER formatted data ---
def load_text_and_create_ner_data(file_path: str, target_label: str) -> List[Dict]:
    """
    Read a text file and create NER formatted data.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"Processing {len(lines)} lines...")
        
        for line in lines:
            text = line.strip()
            if not text:
                continue

            # Define the entity as the whole text
            start = 0
            end = len(text)
            
            # format [start_char, end_char, label]
            entity = [start, end, target_label]
            
            data.append({
                "text": text,
                "entities": [entity] # Entities is a list of [start, end, label]
            })
            
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
        
    return data

# --- NER Dataset Processor ---
class NERDatasetProcessor:
    """ Process NER dataset for training and evaluation. """
    
    def __init__(self, model_name: str = "roberta-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )

        # Mapping labels to IDs
        self.label2id = {"O": 0}                                    #O
        self.label2id[f"B-{TARGET_LABEL}"] = len(self.label2id)     #B
        self.label2id[f"I-{TARGET_LABEL}"] = len(self.label2id)     #I

        self.id2label = {v: k for k, v in self.label2id.items()}
        logger.info(f"Labels : {self.label2id}")

    def convert_to_bio_labels(self, text: str, entities: List[List]) -> List[int]:
        """Convert entities to BIO labels for each character in the text."""
        char_labels = [self.label2id["O"]] * len(text)

        for entity in entities:
            if not (isinstance(entity, (list, tuple)) and len(entity) > 2):
                logger.warning(f"Entity is not in the right format: {entity}")
                continue

            try:
                # Convert start and end to int
                start, end, label = int(entity[0]), int(entity[1]), entity[2]
            except ValueError:
                logger.warning("Was not possible to convert start/end to int.")
                continue

            if start >= len(text) or end > len(text) or label != TARGET_LABEL:
                logger.warning(f"Invalid Span Label: {entity}. Ignoring.")
                continue

            b_label_name = f"B-{label}"
            i_label_name = f"I-{label}"

            # Apply BIO labels
            if b_label_name in self.label2id and i_label_name in self.label2id:
                char_labels[start] = self.label2id[b_label_name]
                for i in range(start + 1, end):
                    if i < len(char_labels):
                        char_labels[i] = self.label2id[i_label_name]
                
        return char_labels    
    
    # --- Tokenization and alignment ---
    def tokenize_and_align(self, examples: Dict) -> Dict:
        """Tokenization and alignment of labels with tokens."""
        tokenized_inputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_offsets_mapping=True,
            is_split_into_words=False,
        )

        all_labels = []

        for i, (text, entities) in enumerate(
            zip(examples["text"], examples["entities"])
        ):
            char_labels = self.convert_to_bio_labels(text, entities)
            offset_mapping = tokenized_inputs["offset_mapping"][i]
            token_labels = []

            for start, end in offset_mapping:
                if start == end:
                    token_labels.append(-100)
                else:
                    safe_start = min(start, len(char_labels) - 1)
                    token_labels.append(char_labels[safe_start])

            all_labels.append(token_labels)

        tokenized_inputs["labels"] = all_labels
        tokenized_inputs.pop("offset_mapping")

        return tokenized_inputs

    def prepare_dataset(
        self, data: List[Dict], test_size: float = 0.1, val_size: float = 0.1
    ) -> DatasetDict:
        """Prepare dataset for training and evaluation."""
        
        logger.info(f"Preparing {len(data)} examples...")
        
        train_data, test_data = train_test_split(
            data, test_size=test_size, random_state=42
        )
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, random_state=42
        )

        logger.info(
            f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
        )

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
                "test": Dataset.from_list(test_data),
            }
        )

        tokenized_datasets = dataset_dict.map(
            self.tokenize_and_align,
            batched=True,
            remove_columns=["text", "entities"],
        )

        return tokenized_datasets                

#--- Metrics computation ---
def create_compute_metrics_fn(id2label: Dict[int, str]):
    """Creating compute_metrics."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_predictions = []

        for prediction, label in zip(predictions, labels):
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_labels.append(id2label[lab])
                    true_predictions.append(id2label[pred])

        all_b_i_labels = [f"B-{TARGET_LABEL}", f"I-{TARGET_LABEL}"]

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            average="macro",
            labels=all_b_i_labels,
            zero_division=0,
        )

        results = {
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": f1,
            f"{TARGET_LABEL}_precision": precision,
            f"{TARGET_LABEL}_recall": recall,
            f"{TARGET_LABEL}_f1": f1,
        }
        return results

    return compute_metrics

# Main training function
def train_model(
    tokenized_datasets: DatasetDict,
    processor: NERDatasetProcessor,
    output_dir: str,
    model_name: str = "roberta-base",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """Token classification model training."""
    logger.info(f"Starting with model {model_name}...")

    label2id = processor.label2id
    id2label = processor.id2label

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    compute_metrics_fn = create_compute_metrics_fn(id2label)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", 
        push_to_hub=False,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info(f"Saving model in {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("Evaluating model...")
    test_results = trainer.predict(tokenized_datasets["test"])

    logger.info(f"Train metrics: {train_result.metrics}")
    logger.info(f"Test metrics: {test_results.metrics}")

    return trainer, test_results

def main():
    parser = argparse.ArgumentParser(description="Training NER model for Key Words Extraction")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input text file for training",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Model output directory"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="roberta-base",
        help="Pre-trained model name or path",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.1, help="Test set proportion"
    )
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="Validation set proportion"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"GPU Device: {device}")

    logger.info(f"Loading data from {args.input}...")
    
    # Load text data and create NER formatted data
    data = load_text_and_create_ner_data(args.input, TARGET_LABEL)
    
    if not data:
        logger.error(f"No valid data found {args.input}. Exiting.")
        return
    logger.info(f"Loading {len(data)} samples")

    logger.info("Normalizing for PyArrow...")
    for item in data:
        # Ensure entities are lists of strings for PyArrow compatibility
        item["entities"] = [
            [str(e) for e in entity]
            for entity in item.get("entities", [])
        ]

    processor = NERDatasetProcessor(model_name=args.model_name)

    tokenized_datasets = processor.prepare_dataset(
        data, test_size=args.test_size, val_size=args.val_size
    )

    logger.info("Data set prepared.")

    trainer, test_results = train_model(
        tokenized_datasets,
        processor=processor,
        output_dir=args.output,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    metrics_path = output_path / "test_metrics.json"
    logger.info(f"Saving metrics in {metrics_path}...")
    save_json(test_results.metrics, str(metrics_path))

    logger.info("Training complete.")


if __name__ == "__main__":
    main()    