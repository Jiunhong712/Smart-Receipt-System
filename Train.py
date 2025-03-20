import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess the data with oversampling
def load_and_oversample_data(csv_path, target_size=500):
    # Load the CSV with encoding fallback
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("Warning: UTF-8 decoding failed. Attempting with 'latin1' encoding.")
        df = pd.read_csv(csv_path, encoding='latin1')
    
    df = df.drop_duplicates(subset=['Product Name'])
    
    # Oversample to balance the categories
    oversampled_dfs = []
    for category, group in df.groupby('Category'):
        if len(group) < target_size:
            oversampled_group = group.sample(n=target_size, replace=True, random_state=42)
        else:
            oversampled_group = group.sample(n=target_size, random_state=42)
        oversampled_dfs.append(oversampled_group)
    
    balanced_df = pd.concat(oversampled_dfs).reset_index(drop=True)
    
    print(f"Balanced dataset size: {len(balanced_df)}")
    print("Category distribution after oversampling:")
    print(balanced_df['Category'].value_counts())
    
    return balanced_df

# Step 2: Prepare the dataset for BERT
class ProductDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 3: Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Step 4: Main training function
def train_bert_model(csv_path, output_dir='fine_tuned_bert', target_size=500):
    # Load and oversample data
    df = load_and_oversample_data(csv_path, target_size)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['Category'])
    texts = df['Product Name'].tolist()

    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_encoder.classes_)
    ).to(device)

    # Create datasets
    train_dataset = ProductDataset(train_texts, train_labels, tokenizer)
    val_dataset = ProductDataset(val_texts, val_labels, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Initialize Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )]
    )

    # Train the model
    print("Starting fine-tuning with early stopping on CUDA...")
    trainer.train()

    # Evaluate the model on the validation set
    print("\nEvaluating the model on the validation set...")
    eval_results = trainer.evaluate()
    print("Evaluation metrics:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # Save the label encoder
    import pickle
    with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved.")

    # Save evaluation metrics to a file
    with open(f'{output_dir}/evaluation_metrics.txt', 'w') as f:
        f.write("Evaluation Metrics:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Evaluation metrics saved to {output_dir}/evaluation_metrics.txt")

# Step 5: Run the script
if __name__ == "__main__":
    csv_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP\Data\LLM\Train\product_names (Jaya Grocer).csv"
    output_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP"
    train_bert_model(csv_path, output_dir, target_size=500)