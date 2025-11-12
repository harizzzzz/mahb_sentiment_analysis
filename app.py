import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

#import all related libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords #remove stopwords
from nltk.stem import WordNetLemmatizer #to format word back to root word
from nltk.sentiment import SentimentIntensityAnalyzer #to help score the sentiment of text

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download("omw-1.4", quiet=True)
nltk.download("vader_lexicon", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia= SentimentIntensityAnalyzer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Load pretrained model
@st.cache_resource
def load_model():
    return joblib.load("v2.pkl")

model = load_model()

# Streamlit UI
st.set_page_config(page_title="Sentiment Prediction App", layout="wide")

st.title("📊 Sentiment Analysis App with WordCloud")
st.markdown("Upload a CSV file with a `TEXT` column to predict sentiment using the trained SVM model.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'TEXT' not in df.columns:
        st.error("❌ The uploaded CSV must contain a 'TEXT' column.")
    else:
        # Preprocess text
        st.info("Preprocessing and predicting sentiments...")
        df['cleaned_text'] = df['TEXT'].apply(preprocess_text)
        df['text_length'] = df['TEXT'].apply(len)
        df['word_count'] = df['TEXT'].apply(lambda x: len(str(x).split()))
        df['vader_score'] = df['TEXT'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
# --- Select same features as training ---
        X_test = df[['cleaned_text', 'text_length', 'word_count', 'vader_score']]
        
        # Predict
        predictions = model.predict(X_test)
        df['Predicted_Sentiment'] = predictions
        
        # Show results
        st.subheader("🧾 Preview of Predictions")
        st.dataframe(df.head())
        
        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download predictions as CSV",
            data=csv,
            file_name='predicted_sentiment.csv',
            mime='text/csv'
        )
        
        # Generate WordClouds
        st.subheader("☁️ WordCloud by Sentiment")
        sentiments = df['Predicted_Sentiment'].unique()
        
        for sentiment in sentiments:
            text = ' '.join(df[df['Predicted_Sentiment'] == sentiment]['clean_text'])
            if text.strip() == "":
                continue
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=200
            ).generate(text)
            
            st.markdown(f"### {sentiment} Reviews")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
