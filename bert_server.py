import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model, tokenizer, and label encoder
model_dir = r"C:\Users\xavie\OneDrive\Documents\Y4S1\FYP\fine_tuned_bert"
try:
    print("Loading fine-tuned BERT model and tokenizer...")
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.eval()
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

try:
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Label encoder loaded. Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    exit(1)

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "BERT Classification Server is running", "classes": list(label_encoder.classes_)})

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        product = data.get("product", "")
        if not product:
            return jsonify({"error": "No product provided"}), 400

        # Tokenize input
        inputs = tokenizer(
            product,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=1).item()
            category = label_encoder.inverse_transform([pred_idx])[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
            prob_dict = {label_encoder.inverse_transform([i])[0]: round(prob, 4) for i, prob in enumerate(probs)}

        print(f"Predicted '{product}' as '{category}'")
        return jsonify({
            "product": product,
            "category": category,
            "probabilities": prob_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    print(f"Starting BERT classification server on http://{host}:{port}...")
    app.run(host=host, port=port, debug=False)