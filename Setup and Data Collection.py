import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets (example datasets)
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Label the data
true_df['label'] = 1  # 1 for real news
fake_df['label'] = 0  # 0 for fake news

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle
