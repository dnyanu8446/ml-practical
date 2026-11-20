import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

df.to_csv('wine.csv', index=False)

print("Wine dataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())