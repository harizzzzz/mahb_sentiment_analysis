import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
        df['clean_text'] = df['TEXT'].apply(preprocess_text)
        
        # Predict
        predictions = model.predict(df['TEXT'])
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
