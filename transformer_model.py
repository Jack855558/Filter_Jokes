import json
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report

# Check if a GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# Load training data
with open('labeled_jokes.json', 'r') as f: 
    train_data = json.load(f)

# Load validation data
with open('validation_jokes.json', 'r') as f: 
    validation_data = json.load(f)

# Separate text and classification labels
X_train = [joke['text'] for joke in train_data]
y_train = [joke['label'] for joke in train_data]

X_val = [joke['text'] for joke in validation_data]
y_val = [joke['label'] for joke in validation_data]

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', clean_up_tokenization_spaces=True)

# Tokenize datasets
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

# Convert to torch tensors
train_input_ids = torch.tensor(train_encodings['input_ids'])
train_attention_mask = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(y_train)

val_input_ids = torch.tensor(val_encodings['input_ids'])
val_attention_mask = torch.tensor(val_encodings['attention_mask'])
val_labels = torch.tensor(y_val)

# Define a custom Dataset class
class JokeDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Create datasets
train_dataset = JokeDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = JokeDataset(val_input_ids, val_attention_mask, val_labels)

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)  # Move the model to the GPU if available

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    save_total_limit=2,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
val_outputs = trainer.predict(val_dataset)
val_predictions = np.argmax(val_outputs.predictions, axis=1)

# Evaluate performance
accuracy = accuracy_score(y_val, val_predictions)
print(f"Accuracy on the validation set: {accuracy * 100:.2f}%")

# More detailed performance metrics
print("Classification Report:")
print(classification_report(y_val, val_predictions))


model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

# Support: Number of actual cases
# F1 score: Balance between precision and recall
# Precision: When the model predicts something, how often is it correct
# Recall: How well does the model predict the actual positives
# Macro avg: Average of precision, recall, and F1
# Weighted avg: Gives more weight to classes with more instances