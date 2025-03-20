import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available.")
else:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_and_oversample_data(df_train, target_size=500):
    print("\nCategory distribution (no oversampling):")
    print(df_train['Category'].value_counts())
    return df_train

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

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def predict_product_category(model, tokenizer, label_encoder, product_name, max_len=128):
    encoding = tokenizer.encode_plus(
        product_name,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_label_idx = logits.argmax(-1).item()
        pred_label = label_encoder.inverse_transform([pred_label_idx])[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    print(f"\nProduct: {product_name} | Predicted: {pred_label}")
    print(f"Tokens: {tokenizer.tokenize(product_name)}")
    print(f"Raw logits: {logits.cpu().numpy()[0]}")
    print(f"Softmax probabilities: {probs}")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label}: {probs[i]:.4f}")
    return pred_label, logits

def train_bert_model(csv_path, output_dir='fine_tuned_bert', target_size=500):
    df_full = pd.read_csv(csv_path, encoding='latin1').drop_duplicates(subset=['Product Name'])
    train_df, val_df = train_test_split(df_full, test_size=0.2, stratify=df_full['Category'], random_state=42)
    
    # Add external examples to improve generalization
    external_df = pd.DataFrame({
        'Product Name': ["Carlsberg Beer 330ml", "Farmfresh Yogurt 200g", "Nestle Milo 1L", "Colgate Toothpaste 100g"],
        'Category': ["Alcohol", "Chilled and Frozen", "Beverages", "Beauty and Health"]
    })
    train_df = pd.concat([train_df, external_df], ignore_index=True)
    balanced_df = load_and_oversample_data(train_df, target_size)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(balanced_df['Category'])
    texts = balanced_df['Product Name'].tolist()
    val_labels = label_encoder.transform(val_df['Category'])
    val_texts = val_df['Product Name'].tolist()

    print("\nLabel encoder classes:", list(label_encoder.classes_))

    original_labels = label_encoder.transform(df_full['Category'])
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=original_labels)
    class_weights = np.clip(class_weights, 0.5, 5.0)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ProductDataset(texts, labels, tokenizer)
    val_dataset = ProductDataset(val_texts, val_labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(label_encoder.classes_),
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.3
    ).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.1,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        no_cuda=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.01
        )]
    )

    print("\nStarting fine-tuning with early stopping on CUDA...")
    trainer.train()

    print("\nEvaluating the model on the validation set...")
    eval_results = trainer.evaluate()
    print("Evaluation metrics:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

    print("\nAnalyzing misclassified examples...")
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    logits = predictions.predictions
    misclassified_indices = np.where(preds != val_labels)[0]
    print(f"Number of misclassified examples: {len(misclassified_indices)}")
    
    print("\nSample misclassified examples with tokens and logits:")
    for i in misclassified_indices[:5]:
        true_label = label_encoder.inverse_transform([val_labels[i]])[0]
        pred_label = label_encoder.inverse_transform([preds[i]])[0]
        text = val_texts[i]
        tokens = tokenizer.tokenize(text)
        logit_scores = logits[i]
        print(f"Product: {text} | True: {true_label} | Pred: {pred_label}")
        print(f"Tokens: {tokens}")
        print(f"Logits: {logit_scores}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel and tokenizer saved to {output_dir}")

    with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved.")

    with open(f'{output_dir}/evaluation_metrics.txt', 'w') as f:
        f.write("Evaluation Metrics:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"Evaluation metrics saved to {output_dir}/evaluation_metrics.txt")

    # Manual checking loop
    print("\nEntering manual testing mode. Enter a product name to classify (or 'quit' to stop):")
    while True:
        product_name = input("> ")
        if product_name.lower() == 'quit':
            break
        predict_product_category(model, tokenizer, label_encoder, product_name)

if __name__ == "__main__":
    csv_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP\Data\LLM\Train\product_names (Jaya Grocer).csv"
    output_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP\fine_tuned_bert"
    train_bert_model(csv_path, output_dir, target_size=500)