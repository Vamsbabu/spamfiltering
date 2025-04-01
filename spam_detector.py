import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
import base64
import joblib
import os
import plotly.express as px

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
TEXT_VECTORIZER_PATH = os.path.join(MODEL_DIR, "text_vectorizer.joblib")
TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_model.joblib")
COMBINED_MODEL_PATH = os.path.join(MODEL_DIR, "combined_model.joblib")

# --- Functions ---
def load_data():
    """Loads and preprocesses the data."""
    url=" https://colab.research.google.com/drive/1QYayroWlYXQXpSG7VBwVmuzoyuwaE6Nq"
    data = pd.read_csv('spam.csv')
    data = data.rename(columns={'Category': 'spam', 'Message': 'text'})
    data['spam'] = data['spam'].replace(['ham', 'spam'], [0, 1])
    data = data.dropna()
    return data

def train_models(data, vectorizer_type, model_type, class_weight_value, C_value):
    """Trains and saves the models based on user-defined options."""
    X_text = data['text']
    y = data['spam']
    X_text_train, X_text_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

    if vectorizer_type == "unigram":
        text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    elif vectorizer_type == "bigram":
        text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
    else:
        raise ValueError("Invalid vectorizer type")

    X_text_train_vectorized = text_vectorizer.fit_transform(X_text_train)
    X_text_test_vectorized = text_vectorizer.transform(X_text_test)
    X_train_combined = X_text_train_vectorized
    X_test_combined = X_text_test_vectorized

    if class_weight_value == "balanced":
        class_weight = "balanced"
    elif class_weight_value == "none":
        class_weight = None
    else:
      class_weight = {0:1,1:float(class_weight_value)} # Custom weight
    
    if model_type == "logistic_regression":
        text_model = LogisticRegression(solver='liblinear', random_state=42, class_weight=class_weight, C=C_value)
        combined_model = LogisticRegression(solver='liblinear', random_state=42, class_weight=class_weight, C=C_value)
    elif model_type == "naive_bayes":
        text_model = MultinomialNB(alpha = C_value) # C_value here will represent alpha for Naive Bayes
        combined_model = MultinomialNB(alpha = C_value)
    elif model_type == "svm":
        text_model = SVC(kernel='linear', random_state=42, class_weight=class_weight, C=C_value, probability=True)
        combined_model = SVC(kernel='linear', random_state=42, class_weight=class_weight, C=C_value, probability=True)

    else:
        raise ValueError("Invalid model type")

    text_model.fit(X_text_train_vectorized, y_train)
    combined_model.fit(X_train_combined, y_train)

    # Save models and vectorizer
    joblib.dump(text_vectorizer, TEXT_VECTORIZER_PATH)
    joblib.dump(text_model, TEXT_MODEL_PATH)
    joblib.dump(combined_model, COMBINED_MODEL_PATH)

    return text_vectorizer, text_model, combined_model, X_text_test, y_test

@st.cache_resource
def load_models_and_data(vectorizer_type, model_type, class_weight_value, C_value):
    """Loads pre-trained models and vectorizer, or trains them if they don't exist."""
    data = load_data()
    if os.path.exists(TEXT_VECTORIZER_PATH) and os.path.exists(TEXT_MODEL_PATH) and os.path.exists(COMBINED_MODEL_PATH):
        text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)
        text_model = joblib.load(TEXT_MODEL_PATH)
        combined_model = joblib.load(COMBINED_MODEL_PATH)
        X_text = data['text']
        y = data['spam']
        _, X_text_test, _, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
        return text_vectorizer, text_model, combined_model, X_text_test, y_test
    else:
        st.info("Training models...")
        text_vectorizer, text_model, combined_model, X_text_test, y_test = train_models(data, vectorizer_type, model_type, class_weight_value, C_value)
        st.success("Models trained and saved!")
        return text_vectorizer, text_model, combined_model, X_text_test, y_test


def predict_spam(message, text_vectorizer, text_model, combined_model, threshold=0.5, text_weight=0.9, combined_weight=0.1):
    """Predicts if a given message is spam or not."""
    message_vectorized = text_vectorizer.transform([message])

    # Text model prediction
    text_prediction = text_model.predict_proba(message_vectorized)[:, 1]
    
    # Combined model prediction
    combined_prediction = combined_model.predict_proba(message_vectorized)[:, 1] 

    # Fused prediction
    fused_prediction = text_weight * text_prediction + combined_weight * combined_prediction
    prediction = "spam" if fused_prediction[0] > threshold else "ham"
    return prediction, fused_prediction[0], text_prediction[0], combined_prediction[0]


def highlight_features(message, text_vectorizer, text_model, top_n=5):
    """Highlights top features from text"""
    message_vectorized = text_vectorizer.transform([message])
    feature_names = np.array(text_vectorizer.get_feature_names_out())
    coefficients = text_model.coef_.flatten()
    importance = np.abs(coefficients)
    top_indices = np.argsort(importance)[::-1][:top_n]
    top_features = feature_names[top_indices]
    words = message.split()
    highlighted_words = []
    for word in words:
      if word in top_features:
        highlighted_words.append(f"<span style='background-color: yellow;'>{word}</span>")
      else:
        highlighted_words.append(word)

    return " ".join(highlighted_words)


def plot_probability_bar(probability, color):
    """Plots a horizontal bar for probability visualization"""
    fig, ax = plt.subplots(figsize=(6, 0.8))
    ax.barh(0, probability, color=color, align="center")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("Probability", fontsize=10)
    ax.invert_yaxis()  # Invert y axis to have 0 at top
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0) # Reset the pointer so that the stream starts from beginning
    return buf

def get_classification_report(X_text_test, y_test, text_vectorizer, text_model, combined_model, threshold=0.5, text_weight=0.7, combined_weight=0.3):
    """Generate classification reports for each model."""
    X_test_vectorized = text_vectorizer.transform(X_text_test)
    
    text_predictions = text_model.predict_proba(X_test_vectorized)[:, 1]
    combined_predictions = combined_model.predict_proba(X_test_vectorized)[:, 1]
    fused_predictions = (text_weight * text_predictions + combined_weight * combined_predictions)
    fused_binary_predictions = np.where(fused_predictions > threshold, 1, 0)
    
    text_report = classification_report(y_test, (text_predictions > threshold).astype(int), output_dict=True)
    combined_report = classification_report(y_test, (combined_predictions > threshold).astype(int), output_dict=True)
    fused_report = classification_report(y_test, fused_binary_predictions, output_dict=True)
    return text_report, combined_report,fused_report, fused_binary_predictions

def display_classification_report(report, title):
    st.markdown(f"#### {title}")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

def plot_confusion_matrix(y_true, y_pred):
    """Plots a confusion matrix."""
    conf_mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0) # Reset the pointer so that the stream starts from beginning
    return buf


def display_feature_importance(text_vectorizer, text_model):
        """Displays feature importance along with their coefficients"""
        st.markdown("### Feature Importance")
        feature_names = np.array(text_vectorizer.get_feature_names_out())
        coefficients = text_model.coef_.flatten()
        importance = np.abs(coefficients)
        
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Absolute Coefficient': importance
        })
        
        st.dataframe(feature_df.sort_values(by='Absolute Coefficient', ascending=False))

def display_example_predictions(X_text_test, y_test, fused_binary_predictions, num_examples=5):
        """Displays example predictions."""
        st.markdown("### Example Predictions (Fused Model)")
        for i in range(num_examples):
            st.write(f"**Email:** {X_text_test.iloc[i][:50]}...")
            st.write(f"**Actual:** {y_test.iloc[i]}, **Predicted:** {fused_binary_predictions[i]}")
            st.markdown("---")

# --- Streamlit App ---
def prediction_page(text_vectorizer, text_model, combined_model):
    st.title("Email Spam Detector")

    # Configuration Options
    st.sidebar.header("Model Configurations")
    threshold = st.sidebar.slider("Threshold:", min_value=0.0, max_value=1.0, value=0.5)
    text_weight = st.sidebar.slider("Text Model Weight:", min_value=0.0, max_value=1.0, value=0.7)
    combined_weight = st.sidebar.slider("Combined Model Weight:", min_value=0.0, max_value=1.0, value=0.3)
    
    # Tabs
    tab1, tab2= st.tabs(["Prediction", "Details"])

    with tab1:
        # Input text area
        user_input = st.text_area("Enter Email Message:", height=150)

        if user_input:
            prediction, fused_probability, text_probability, combined_probability = predict_spam(user_input, text_vectorizer, text_model, combined_model,threshold,text_weight,combined_weight)

            st.markdown("### Fused Model Prediction")
            st.write(f"**Prediction:**  <span style='color: {'red' if prediction == 'spam' else 'green'}; font-weight: bold;  font-size: 1.5em;'>{prediction.upper()}</span>",unsafe_allow_html=True)
            st.image(plot_probability_bar(fused_probability, "green" if prediction == "ham" else "red"), width=500)

            # Displaying probabilities for both models using columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Text Model Output")
                st.image(plot_probability_bar(text_probability, "green" if text_probability<0.5 else "red"), width=350)
            with col2:
                st.markdown("### Combined Model Output")
                st.image(plot_probability_bar(combined_probability, "green" if combined_probability<0.5 else "red"), width=350)

    with tab2:
        if user_input:
            st.markdown("### Feature Analysis")
            highlighted_text = highlight_features(user_input, text_vectorizer, text_model)
            st.markdown(f"<p style='font-size: 1.1em;'>{highlighted_text}</p>", unsafe_allow_html=True)
            display_feature_importance(text_vectorizer,text_model)
            
def overview_page(data):
    st.title("Email Spam Detector Overview")
    st.write("This app uses a combination of text-based features and a model to classify emails as either 'spam' or 'ham'.")
    st.markdown("""
    ### How it works:
    1. **Text Processing**: The text of the emails is converted into a numerical format using TF-IDF, highlighting the important words for classification.
    2. **Model Combination**: We use two models: one trained purely on text and another trained on combination of features (which is the same as the first here).
    3. **Weighted Fused Prediction**: The models output a probability, and these probabilities are combined with weights to provide a final spam score. The weights and the classification threshold can be adjusted on the prediction page.
    4. **Feature Highlighting**: The important words in the email are highlighted to show what the model is looking for.

    ### Use
    - To try out the model, please navigate to the "Prediction" page using the sidebar.
    - Feel free to adjust the model parameters to achieve the best prediction.
    """)
    st.markdown("### Dataset Sample:")
    st.dataframe(data.head(5))

    # Interactive visualization
    st.markdown("### Interactive Data Visualization")
    fig = px.pie(data, names='spam', title='Distribution of Ham vs Spam Emails')
    st.plotly_chart(fig)


def evaluation_page(X_text_test, y_test, text_vectorizer, text_model, combined_model, threshold=0.5, text_weight=0.7, combined_weight=0.3):
    st.title("Model Evaluation")
    st.write("This section displays the performance of the text-based and combined models on the held-out test set.")

    text_report, combined_report, fused_report, fused_binary_predictions = get_classification_report(X_text_test, y_test, text_vectorizer, text_model, combined_model, threshold, text_weight, combined_weight)
    display_classification_report(text_report, "Text-Based Model Report")
    display_classification_report(combined_report, "Combined Model Report")
    display_classification_report(fused_report, "Fused Model Report")

    st.markdown("### Confusion Matrix (Fused Model)")
    st.image(plot_confusion_matrix(y_test, fused_binary_predictions))
    display_feature_importance(text_vectorizer, text_model)
    display_example_predictions(X_text_test, y_test, fused_binary_predictions)


# --- Main ---
def main():
    st.set_page_config(page_title="Email Spam Detector", layout="wide")
    data = load_data() # Load the data here to display it on overview page
    
    # Sidebar for navigation and model configurations
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Prediction", "Evaluation"])

    st.sidebar.header("Model Training Configurations")
    vectorizer_type = st.sidebar.selectbox("Vectorizer Type:", ["unigram", "bigram"])
    model_type = st.sidebar.selectbox("Model Type:", ["logistic_regression", "naive_bayes", "svm"])
    class_weight_value = st.sidebar.selectbox("Class Weight:", ["none", "balanced", 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]) # class weight as "balanced", "none" or as number

    if model_type == "logistic_regression" or model_type == "svm": # Add hyperparameter C only for these models
      C_value = st.sidebar.slider("C (Regularization):", min_value=0.01, max_value=2.0, value=1.0, step = 0.01)
    elif model_type == "naive_bayes":
        C_value = st.sidebar.slider("Alpha (Smoothing Parameter):", min_value=0.01, max_value=2.0, value=1.0, step = 0.01)

    text_vectorizer, text_model, combined_model, X_text_test, y_test = load_models_and_data(vectorizer_type, model_type, class_weight_value, C_value)
    
    # Configuration options for evaluation page
    if page == "Evaluation":
         threshold = st.sidebar.slider("Threshold:", min_value=0.0, max_value=1.0, value=0.5)
         text_weight = st.sidebar.slider("Text Model Weight:", min_value=0.0, max_value=1.0, value=0.7)
         combined_weight = st.sidebar.slider("Combined Model Weight:", min_value=0.0, max_value=1.0, value=0.3)
         evaluation_page(X_text_test, y_test, text_vectorizer, text_model, combined_model, threshold, text_weight, combined_weight)

    if page == "Overview":
        overview_page(data)
    elif page == "Prediction":
         prediction_page(text_vectorizer, text_model, combined_model)


if __name__ == "__main__":
    main()