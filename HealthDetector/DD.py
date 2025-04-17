import pandas as pd
import numpy as np 

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