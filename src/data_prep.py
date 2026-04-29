import os
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_data(subset_size=50000, max_length=128, model_name="bert-base-uncased"):
    """
    Downloads, subsets, tokenizes, and splits the amazon_polarity dataset.
    """
    print(f"Loading amazon_polarity dataset...")

    dataset = load_dataset("fancyzhx/amazon_polarity")

    print(f"Subsetting data to handle hardware limits (Train: {subset_size})...")
    train_val_subset = dataset['train'].shuffle(seed=42).select(range(subset_size))
    test_subset = dataset['test'].shuffle(seed=42).select(range(int(subset_size * 0.2))) 
    train_val_split = train_val_subset.train_test_split(test_size=0.2, seed=42)
    final_dataset = {
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
        'test': test_subset
    }
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        texts = [f"{title} {content}" for title, content in zip(examples['title'], examples['content'])]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    print("Tokenizing dataset splits...")
    tokenized_datasets = {}
    for split_name, split_data in final_dataset.items():
        tokenized_datasets[split_name] = split_data.map(
            tokenize_function, 
            batched=True, 
            remove_columns=['title', 'content']
        )
        if 'label' not in tokenized_datasets[split_name].column_names:
            tokenized_datasets[split_name] = tokenized_datasets[split_name].rename_column("labels", "label")
            
        tokenized_datasets[split_name].set_format("torch")
        print(f"{split_name.capitalize()} split: {len(tokenized_datasets[split_name])} samples")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    for split_name in tokenized_datasets:
        save_path = os.path.join(output_dir, split_name)
        tokenized_datasets[split_name].save_to_disk(save_path)
        print(f"Saved {split_name} split to {save_path}")
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print("Data preparation complete!")
if __name__ == "__main__":
    prepare_data()