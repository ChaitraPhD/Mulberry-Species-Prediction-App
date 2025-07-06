import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

# Title
st.set_page_config(page_title="Mulberry Species Predictor", layout="centered")
st.title("🌿 Mulberry Species Prediction App")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ChaitraPhD/Mulberry-Species-Prediction-App/master/mulberry_leafyield.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

df = load_data()

# Remove rare species (min 2 samples)
def filter_sparse(df, target="Species", min_samples=2):
    counts = df[target].value_counts()
    keep = counts[counts >= min_samples].index
    return df[df[target].isin(keep)]

df = filter_sparse(df)

# Encode species
X = df.drop("Species", axis=1)
y = df["Species"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y_enc)

# Sidebar input
st.sidebar.header("🌱 Input Features")
input_data = {
    "Internodal_distance_cm": st.sidebar.slider("Internodal Distance (cm)", 1.2, 4.5, 2.5),
    "Leaf_lamina_length _cm": st.sidebar.slider("Leaf Lamina Length (cm)", 1.2, 14.5, 8.0),
    "Leaf_lamina_width_cm_
    
