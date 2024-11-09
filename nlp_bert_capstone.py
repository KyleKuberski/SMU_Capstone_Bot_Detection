# -*- coding: utf-8 -*-
"""NLP_BERT_Capstone.py"""

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import torch
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the data
#KAGGLE DATASET, 50k, string oriented to train model
data = pd.read_csv("C:/Users/kuber/Desktop/Capstone/Kaggle/bot_detection_data.csv")

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Apply preprocessing to the tweet column
data['Tweet'] = data['Tweet'].apply(preprocess_text)

# Encode 'Verified' column (assuming it's binary True/False)
data['Verified'] = data['Verified'].apply(lambda x: 1 if x == 'TRUE' else 0)

# Prepare labels and features
X_text = data['Tweet']
y = data['Bot Label']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=1000)
X_text_tfidf = tfidf.fit_transform(X_text)

# Convert sparse matrix to DataFrame for merging with other features
X_text_tfidf_df = pd.DataFrame(X_text_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Select other relevant features
X_other = data[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']]
X = pd.concat([X_text_tfidf_df, X_other.reset_index(drop=True)], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = logistic_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Function to predict bot probability
def predict_bot(text, retweet_count, mention_count, follower_count, verified):
    text = preprocess_text(text)
    text_tfidf = tfidf.transform([text])
    text_tfidf_df = pd.DataFrame(text_tfidf.toarray(), columns=tfidf.get_feature_names_out())

    input_features = pd.DataFrame({
        'Retweet Count': [retweet_count],
        'Mention Count': [mention_count],
        'Follower Count': [follower_count],
        'Verified': [verified]
    })

    input_data = pd.concat([text_tfidf_df, input_features], axis=1)
    bot_prob = logistic_model.predict_proba(input_data)[:, 1]
    return bot_prob[0]

# print input example usage of starting prediction function
print("Bot Probability:", predict_bot("This is a sample tweet for bot detection.", 10, 2, 300, 1))

#_________ BERT __________________
# Load the file with saved embeddings
data = pd.read_csv("C:/Users/kuber/Desktop/Capstone/BERT_Embed/x_data_with_embeddings.csv")

# Function to convert embedding string to numpy array, handling 2D format in string
def parse_embedding(embedding_str):
    # Remove the outer brackets and split by whitespace or newlines
    embedding_str = embedding_str.strip("[]").replace('\n', ' ')
    # Convert the cleaned string to a list of floats
    return np.array([float(x) for x in embedding_str.split()])

# Apply the parsing function to the embeddings column
data['parsed_embeddings'] = data['embeddings'].apply(parse_embedding)

# Combine parsed embeddings into a single 2D numpy array for processing
X_embeddings = np.vstack(data['parsed_embeddings'].values)  # Stack all parsed embedding arrays

# (Optional) PCA for Dimensionality Reduction and Plotting
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(X_embeddings)

# Plot the reduced embeddings (using example labels if available)
# Replace 'Bot Label' with the actual column name of your labels if it's different
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=data['Bot Label'])  
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA of BERT Embeddings")
plt.show()
#_________ BERT END __________________

# Print a few raw embedding strings to inspect their format
print(data['embeddings'].head(5))


#_________ MLP Model __________________
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import re

# Function to clean and parse embedding strings
def parse_embedding(embedding_str):
    # Remove newline characters, double spaces, and ensure commas between numbers
    clean_str = re.sub(r'\s+', ',', embedding_str.strip())
    clean_str = clean_str.replace('[,', '[').replace(',]', ']')  # Fix brackets after replacing spaces

    # Attempt to parse the string with literal_eval
    try:
        return np.array(ast.literal_eval(clean_str))
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing embedding: {e}")
        return None  # Return None for any problematic entries

# Convert embeddings from string to array
data['embeddings'] = data['embeddings'].apply(parse_embedding)

# Drop any rows with None in embeddings column if needed
data = data.dropna(subset=['embeddings'])


# Check how many embeddings were successfully parsed
parsed_embeddings = data['embeddings'].apply(parse_embedding)
successful_parses = parsed_embeddings.dropna()

print(f"Successfully parsed embeddings: {len(successful_parses)} / {len(data)}")


# Prepare final features
X_text = np.vstack(data['embeddings'].values)
X_other = data[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']].values
X = np.hstack((X_text, X_other))
y = data['Bot Label'].values

# Convert X and y to appropriate data types
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)  # Ensure y is an integer array for binary classification

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple MLP model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluation
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
#_________ MLP Model END __________________