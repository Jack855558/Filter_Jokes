import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#Load training data
with open('labeled_jokes.json', 'r') as f: 
    train_data = json.load(f)

#Load validation data
with open('validation_jokes.json', 'r') as f: 
    validation_data = json.load(f)

X_train = [joke['text'] for joke in train_data]
y_train = [joke['label'] for joke in train_data]

X_val = [joke['text'] for joke in validation_data]
y_val = [joke['label'] for joke in validation_data]

#Create bag of words representation for train and validation sets
#stop_words removes common words
vectorizer = CountVectorizer(stop_words='english', max_features=5000) 

#Fit vectorizer to data
X_train_bow = vectorizer.fit_transform(X_train)
X_val_bow = vectorizer.transform(X_val)

#Dont vectorize the validation so there is no data leakage

#Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_bow, y_train)

#Predict 1(appropriate) or 0(inappropriate)
val_predictions = model.predict(X_val_bow)

#Evaluate the models preformance on validation set
accuracy = accuracy_score(y_val, val_predictions)
print(f"Accuracy on the validation set: {accuracy * 100:.2f}%")

# More detailed prefomance metrics
print("Classification Report:")
print(classification_report(y_val, val_predictions))

#Support: Number of actual cases
#F1 score: balance between precesion and recall
#Precesion: When the model predicts something how often is it correct
#Recall: How well does the model predict the actual positives
#Macro avg: avg of precesion, recall, and f1
#Weigted avg: gives more weight to classes with more instances