import pandas as pd
import numpy as np 

"""
DATA LOADING AND PREPROCESSING
This code loads the Cleveland Heart Disease dataset, cleans it by removing missing values, and converts the target variable into a binary classification problem.
Note, we are using the Cleveland dataset from the UCI Machine Learning Repository.
The dataset contains 14 attributes
"""

filepath = "heart+disease/processed.cleveland.data"

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'target'
]

df = pd.read_csv(filepath, header=None, names=column_names)
df = df.replace('?', np.nan)  # Replace '?' with NaN
df = df.dropna()  # Drop rows with NaN values

df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

df.dropna(inplace=True)

#convert target to binary classification:
#0 = no heart disease, 1 = has heart disease (combine classes 1–4)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

#reset index (optional, just to keep things clean)
df.reset_index(drop=True, inplace=True)

#TESTING: preview the cleaned dataset
print("✅ Cleaned dataset shape:", df.shape)
print(df.head())




"""
DATA GENERATION: SYNTHETIC DOCTOR NOTES
This code generates synthetic doctor notes based on the symptoms and recommendations related to heart disease.
It uses random selection to create a variety of notes that could be used for training or testing purposes in a machine learning context.
    
"""
import random

#define possible symptoms and recommendations
symptoms = [
    "chest pain", "shortness of breath", "dizziness", "fatigue", "irregular heartbeat",
    "tightness in chest", "lightheadedness", "swelling in ankles", "nausea"
]
recommendations = [
    "recommended stress test", "ordered EKG", "suggested lifestyle changes",
    "prescribed medication", "referred to cardiologist", "scheduled follow-up"
]

#create a function to generate synthetic doctor notes
def generate_note():
    return f"Patient reports {random.choice(symptoms)}. {random.choice(recommendations)}."

#apply to each row
df['DoctorNote'] = df.apply(lambda row: generate_note(), axis=1)

#TESTING: preview the new column with synthetic doctor notes
print("✅ Doctor notes added. Sample notes:")
print(df[['DoctorNote']].head())




"""
EMBEDDING DOCTOR NOTES
This code uses the SentenceTransformer library to embed the synthetic doctor notes into a numerical format suitable for machine learning models.
uses Bert embeddings to convert text into a numerical format.
The SentenceTransformer library is a popular choice for generating embeddings from text data.
    
"""
from sentence_transformers import SentenceTransformer
import numpy as np

#load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  #fast and lightweight

#convert the list of doctor notes to embeddings
note_embeddings = model.encode(df['DoctorNote'].tolist(), show_progress_bar=True)

#convert to NumPy array
note_embeddings = np.array(note_embeddings)

#TESTING: show the shape: (num_samples, embedding_dim)
print("✅ Doctor notes embedded. Sample embedding shape:")  
print("Embeddings shape:", note_embeddings.shape)




"""
COMBINE STRUCTURED FEATURES AND EMBEDDINGS
This code combines the structured features (numerical data) with the text embeddings generated from the doctor notes. 
Generated embeddings from synthetic doctor notes using Sentence-BERT
Scaled structured features
Combined both into a single feature matrix for modeling
"""
from sklearn.preprocessing import StandardScaler

#separate features and target
structured_features = df.drop(columns=['target', 'DoctorNote'])
target = df['target'].values

#standardize numeric structured features
#(mean=0, std=1)
#Note: this is important for many ML algorithms to perform well 
scaler = StandardScaler()
structured_scaled = scaler.fit_transform(structured_features)

#combine structured + text embeddings
X_combined = np.hstack((structured_scaled, note_embeddings))
y = target

#TESTING: show the shape of the combined feature matrix and target
print("✅ Combined features created. Sample feature shape:")
print("Combined feature shape:", X_combined.shape)
print("Target shape:", y.shape)


