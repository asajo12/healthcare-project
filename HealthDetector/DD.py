import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

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

# Data analysis & Visualizations of Features
# 1. Heart Disease Frequency
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='target', data=df, palette='Set2')
legend_labels = ['No Heart Disease', 'Has Heart Disease']
handles = [plt.Rectangle((0,0),1,1, color=c) for c in sns.color_palette('Set2')[:2]]
plt.legend(handles, legend_labels, title="Condition")
plt.title('Heart Disease Frequency')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Age distribution by target
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='age', hue='target', multiple='stack', palette='Set1', bins=20)
plt.title('Age Distribution by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 3. Cholesterol vs. Age, colored by target
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age', y='chol', hue='target', palette='coolwarm')
plt.title('Cholesterol vs Age by Heart Disease Status')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.tight_layout()
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


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




""" 
TRAIN-TEST SPLIT
This code splits the combined feature matrix and target variable into training and test sets.
"""
from sklearn.model_selection import train_test_split

#split the data into training and test sets
#80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

#TESTING: show the shape of the training and test sets
print("✅ Data split into training and test sets. Sample sizes:")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

""" 
BUILDING NN MODEL
This code builds a simple feedforward neural network using Keras.

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_combined.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  #binary classification
])

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#TESTING: show the model summary
print("✅ Model summary:")
model.summary()

#training the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy (NN): {accuracy:.4f}")

""" 
VISUALIZING TRAINING HISTORY
"""
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

"""
BASE COMPARISON: LOGISTIC REGRESSION
This code builds a logistic regression model as a baseline for comparison with the neural network model.
"""

# Reuse the structured features before combining with embeddings
X_structured_only = structured_scaled  # already standardized earlier
y = target

X_train_struct, X_test_struct, y_train_struct, y_test_struct = train_test_split(
    X_structured_only, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_struct, y_train_struct)
from sklearn.metrics import accuracy_score, f1_score

y_pred_logreg = logreg.predict(X_test_struct)

acc_logreg = accuracy_score(y_test_struct, y_pred_logreg)
f1_logreg = f1_score(y_test_struct, y_pred_logreg)

print(f"📉 Logistic Regression Accuracy: {acc_logreg:.4f}")
print(f"📉 Logistic Regression F1 Score: {f1_logreg:.4f}")

loss, acc_nn = model.evaluate(X_test, y_test)

print("\n🔍 Final Model Comparison:")
print(f"✅ Neural Net (with notes) Accuracy: {acc_nn:.4f}")
print(f"📉 Logistic Regression (structured only) Accuracy: {acc_logreg:.4f}")

print(f"✅ Neural Net (with notes) F1 Score: {f1_score(y_test, model.predict(X_test) > 0.5):.4f}")
print(f"📉 Logistic Regression (structured only) F1 Score: {f1_logreg:.4f}")

# Confusion Matrices for models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Confusion Matrix: Logistic Regression:")
cm_logreg = confusion_matrix(y_test_struct, y_pred_logreg)
disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg, display_labels=['No Disease', 'Disease'])
disp_logreg.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix: Logistic Regression")
plt.grid(False)
plt.show()
y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")

print("Confusion Matrix: Neural Network")
cm = confusion_matrix(y_test, y_pred_nn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Neural Network")
plt.grid(False)
plt.show()

