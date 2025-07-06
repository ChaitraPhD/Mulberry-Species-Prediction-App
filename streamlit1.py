import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Mulberry Species Classifier", layout="wide")
st.title('🌿 Mulberry Species Prediction App')

# Load Data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/ChaitraPhD/Mulberry-Species-Prediction-App/master/mulberry_leafyield.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

# Fix column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Filter classes with at least 2 samples
def filter_sparse_classes(df, target_column, min_samples=2):
    class_counts = df[target_column].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples].index
    return df[df[target_column].isin(classes_to_keep)]

df = filter_sparse_classes(df, 'Species', min_samples=2)

# Encode labels
X = df.drop('Species', axis=1)
y = df['Species']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# SMOTE to balance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y_encoded)

# Sidebar input
st.sidebar.header("🌱 Input Features")
input_data = {
    'Internodal_distance_cm': st.sidebar.slider('Internodal Distance (cm)', 1.2, 4.5, 2.5),
    'Leaf_lamina_length _cm': st.sidebar.slider('Leaf Lamina Length (cm)', 1.2, 14.5, 8.0),
    'Leaf_lamina_width_cm': st.sidebar.slider('Leaf Lamina Width (cm)', 5.0, 12.5, 8.0),
    'Leaf_size_sq.cm': st.sidebar.slider('Leaf Size (sq.cm)', 30, 180, 100),
    'Petiole_length_cm': st.sidebar.slider('Petiole Length (cm)', 2.0, 5.0, 3.5),
    'Petiole_width_cm': st.sidebar.slider('Petiole Width (cm)', 0.28, 0.34, 0.31)
}
input_df = pd.DataFrame([input_data])

# Model Comparison
def compare_models(X, y):
    models = {
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier()
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5)
        results[name] = scores.mean()
    return results

st.subheader("📊 Model Comparison (Cross-Validation)")
results = compare_models(X_res, y_res)
st.dataframe(pd.DataFrame(results.items(), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False))

# Final Model Training
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.write(f"### 🏆 Random Forest Accuracy: {accuracy:.2%}")

# Prediction
st.subheader("🔮 Predict Mulberry Species")
if st.button("Predict"):
    pred_encoded = rf.predict(input_df)
    prediction = label_encoder.inverse_transform(pred_encoded)
    proba = rf.predict_proba(input_df)[0]
    
    st.success(f"**Predicted Species:** {prediction[0]}")
    proba_df = pd.DataFrame(proba, index=label_encoder.classes_, columns=["Probability"])
    st.dataframe(proba_df.sort_values("Probability", ascending=False))

# Confusion Matrix
st.subheader("📈 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Feature Importance
st.subheader("📌 Feature Importance")
importances = rf.feature_importances_
feature_names = X.columns
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
st.bar_chart(feat_df.set_index('Feature').sort_values('Importance'))

# Hyperparameter Tuning
st.subheader("⚙️ Random Forest Hyperparameter Tuning")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1)
grid.fit(X_res, y_res)
best_params = grid.best_params_
best_score = grid.best_score_

st.write("**Best Parameters:**", best_params)
st.write(f"**Tuned Accuracy:** {best_score:.2%}")
