import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

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

try:
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit()

# Step 2: Prediction function
def predict_product_category(test_product, max_len=128):
    encoding = tokenizer.encode_plus(
        test_product,
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
        softmax_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label = torch.argmax(logits, dim=1).item()

    # Debugging output
    print(f"\nProduct: {test_product}")
    print(f"Tokens: {tokenizer.tokenize(test_product)}")
    print(f"Raw logits: {logits.cpu().numpy()[0]}")
    print(f"Softmax probabilities: {softmax_probs}")
    print(f"Predicted label index: {predicted_label}")
    print("Class probabilities (label: probability):")
    for idx, prob in enumerate(softmax_probs):
        category = label_encoder.inverse_transform([idx])[0]
        print(f"{category}: {prob:.4f}")

    predicted_category = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_category

# Step 3: Interactive testing
print("\nEnter a product name to classify (or 'quit' to stop):")
while True:
    user_input = input("> ")
    if user_input.lower() == 'quit':
        break
    predicted_category = predict_product_category(user_input)
    print(f"Predicted Category: '{predicted_category}'")