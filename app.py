from flask import Flask, render_template, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

app = Flask(__name__)

# Load model and tokenizer
model_path = './results'  # Path to pre-trained model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)

#Create consistant behavior for the model
model.eval()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def classify_joke(joke_text):
    #Checks if Joke is appropriate or not
    inputs = tokenizer(joke_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return bool(prediction == 1)  # Assuming 1 = appropriate, 0 = inappropriate     

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/filter_joke', methods=['POST'])
def filter_joke(): 
    data = request.json
    joke = data.get('text', '')

    if not joke:
        return jsonify({"error": "No joke provided"}), 400
    
    is_appropriate = classify_joke(joke)
    return jsonify({"appropriate": is_appropriate})

if __name__ == "__main__":
    print("Templates directory contents:", os.listdir("templates"))
    app.run(debug=True)