import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder  # Essential import
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Mulberry Species Data Visualisation App')

st.write('This app enables Data Visualisation of Mulberry species and their leaf measurements.')

# Load and display data

df = pd.read_csv('https://raw.githubusercontent.com/ChaitraPhD/Mulberry-Species-Prediction-App/refs/heads/master/mulberry_leafyield.csv')
    
    
def filter_sparse_classes(df, target_column, min_samples=2):
    class_counts = df[target_column].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples].index
    filtered_df = df[df[target_column].isin(classes_to_keep)]
    return filtered_df

# Load and filter data

df = filter_sparse_classes(df, 'Species', min_samples=2)

# Display data in an expander
with st.expander('📊 View Data'):
    st.subheader('Raw Data')
    st.dataframe(df)

    st.subheader('Features (X)')
    X = df.drop('Species', axis=1)
    st.dataframe(X)

    st.subheader('Target (Y)')
    y = df['Species']
    st.dataframe(y)

    # Display class distribution
    st.subheader('Class Distribution')
    class_counts = y.value_counts()
    st.bar_chart(class_counts)
   
    

# Data Visualization
with st.expander('Data Visualization'):
    st.scatter_chart(data=df, x=' Leaf_lamina_length _cm', y='Leaf_lamina_width_cm', color='Species')

# Sidebar for user input
with st.sidebar:
    st.header('Input Features')
    Internodal_distance_cm = st.slider('Internodal Distance (cm)', 1.2, 4.5, 2.5)
    Leaf_lamina_length_cm = st.slider('Leaf Lamina Length (cm)', 1.2, 14.5, 8.0)
    Leaf_lamina_width_cm = st.slider('Leaf Lamina Width (cm)', 5.0, 12.5, 8.0)
    Leaf_size_sq_cm = st.slider('Leaf_Size_sq.cm',30,180,400)
    Petiole_length_cm = st.slider('Petiole Length (cm)', 2.0, 5.0, 3.5)
    Petiole_width_cm = st.slider('Petiole Width (cm)', 0.28, 0.34, 0.31)
    



# Create input DataFrame
input_data = {
    'Internodal_distance_cm': Internodal_distance_cm,
    ' Leaf_lamina_length _cm': Leaf_lamina_length_cm,
    'Leaf_lamina_width_cm': Leaf_lamina_width_cm,
    'Leaf_size_sq.cm': Leaf_size_sq_cm,
    ' Petiole_length_cm': Petiole_length_cm,
    ' Petiole_width_cm': Petiole_width_cm
    
}

input_df = pd.DataFrame(input_data, index=[0])
input_mulberry = pd.concat([input_df, X], axis=0)
with st.expander('🔍 Input Features'):
    st.subheader('User Input')
    st.dataframe(input_df)

    st.subheader('Combined Mulberry Data')
    input_mulberry = pd.concat([input_df, X], axis=0)
    st.dataframe(input_mulberry)
def train_model(data):
    X = data.drop('Species', axis=1)
    y = data['Species']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, label_encoder, accuracy

# Train the model
model, label_encoder, accuracy = train_model(df)

# Display model accuracy in the sidebar
st.sidebar.write(f"### 🏆 Model Accuracy: {accuracy:.2%}")

# Prediction Section
st.subheader('🔮 Predict Mulberry Species')

# Make prediction based on user input
if st.button('Predict Species'):
    # Ensure the input_df has the same columns as the training data
    prediction_encoded = model.predict(input_df)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    st.success(f"**Predicted Species:** {prediction[0]}")
    prediction_proba = model.predict_proba(input_df)
    proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
    st.write('**Prediction Probabilities:**')
    st.dataframe(proba_df.T.rename(columns={0: 'Probability'}))

# Feature Importance Visualization
with st.expander('📊 Feature Importance'):
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    st.bar_chart(feature_importance_df.set_index('Feature'))

def get_confusion_matrix(data, label_encoder, model):
    X = data.drop('Species', axis=1)
    y = data['Species']
    y_encoded = label_encoder.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm

with st.expander('📈 Confusion Matrix'):
    cm = get_confusion_matrix(df, label_encoder, model)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
