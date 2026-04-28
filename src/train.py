import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    """Calculates evaluation metrics for the Trainer."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def run_training_pipeline():

    SUBSET_SIZE = 50000 
    MAX_LENGTH = 128
    MODEL_NAME = "bert-base-uncased"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models", "best_model")

    print("Loading and subsetting dataset...")
    dataset = load_dataset("fancyzhx/amazon_polarity")
    train_val_subset = dataset['train'].shuffle(seed=42).select(range(SUBSET_SIZE))
    train_val_split = train_val_subset.train_test_split(test_size=0.2, seed=42)

    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        texts = [f"{t} {c}" for t, c in zip(examples['title'], examples['content'])]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)

    tokenized_train = train_val_split['train'].map(tokenize_function, batched=True, remove_columns=['title', 'content'])
    tokenized_val = train_val_split['test'].map(tokenize_function, batched=True, remove_columns=['title', 'content'])

    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    # 5. Execute Training
    print("Starting training...")
    trainer.train()

    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\nFinal Validation Metrics:\n{eval_results}")

    # 6. Save Artifacts
    print(f"\nSaving best model to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training pipeline complete.")

if __name__ == "__main__":
    run_training_pipeline()