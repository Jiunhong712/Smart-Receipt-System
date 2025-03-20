import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import pandas as pd
import numpy as np

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the fine-tuned model, tokenizer, and label encoder
model_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP\fine_tuned_bert"
try:
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# Load the label encoder
try:
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit()

# Step 2: Load few-shot examples from CSV and check distribution
def load_few_shot_examples(csv_path, num_examples_per_category=3):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("Warning: UTF-8 decoding failed. Attempting with 'latin1' encoding.")
        df = pd.read_csv(csv_path, encoding='latin1')
    
    df = df.drop_duplicates(subset=['Product Name'])
    print("\nTraining data category distribution:")
    print(df['Category'].value_counts())
    
    few_shot_examples = []
    for category, group in df.groupby('Category'):
        sampled = group.sample(n=min(num_examples_per_category, len(group)), random_state=42)
        for _, row in sampled.iterrows():
            few_shot_examples.append({"product": row['Product Name'], "category": row['Category']})
    
    print(f"\nLoaded {len(few_shot_examples)} few-shot examples:")
    for ex in few_shot_examples:
        print(f"- {ex['product']} → {ex['category']}")
    
    return few_shot_examples

csv_path = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP\Data\LLM\Train\product_names (Jaya Grocer).csv"
few_shot_examples = load_few_shot_examples(csv_path, num_examples_per_category=3)

# Step 3: Prediction function with debugging
def predict_with_few_shot(test_product, examples=few_shot_examples, max_len=128):
    prompt = "Classify the following product into a category based on these examples:\n"
    for ex in examples:
        prompt += f"- {ex['product']} → {ex['category']}\n"
    prompt += f"Now classify: {test_product}"
    print(f"\nFull prompt:\n{prompt}")

    encoding = tokenizer.encode_plus(
        prompt,
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

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        softmax_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Probabilities
        predicted_label = torch.argmax(logits, dim=1).item()

    # Debugging: Print logits and probabilities
    print(f"Raw logits: {logits.cpu().numpy()[0]}")
    print(f"Softmax probabilities: {softmax_probs}")
    print(f"Predicted label index: {predicted_label}")

    predicted_category = label_encoder.inverse_transform([predicted_label])[0]
    print(f"Class probabilities (label: probability):")
    for idx, prob in enumerate(softmax_probs):
        category = label_encoder.inverse_transform([idx])[0]
        print(f"{category}: {prob:.4f}")

    return predicted_category

# Step 4: Interactive testing with debugging
print("\nEnter a product name to classify (or 'quit' to stop):")
while True:
    user_input = input("> ")
    if user_input.lower() == 'quit':
        break
    predicted_category = predict_with_few_shot(user_input)
    print(f"Predicted Category: '{predicted_category}'")