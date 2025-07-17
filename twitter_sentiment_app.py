
import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Twitter Sentiment Analysis", layout="wide")
st.title("💬 Twitter Sentiment Analysis & Visualization")
st.markdown("Analyze and visualize sentiment patterns in Twitter data to understand public opinion.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|\#","", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    tokens = text.split()
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

uploaded_file = st.file_uploader("📤 Upload Twitter CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = ['id', 'entity', 'sentiment', 'text']
    df['cleaned_text'] = df['text'].apply(clean_text)

    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df['sentiment'].value_counts())

    st.subheader("☁️ WordClouds by Sentiment")
    sentiments = df['sentiment'].unique()
    for sentiment in sentiments:
        text = " ".join(df[df['sentiment'] == sentiment]['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.markdown(f"**{sentiment}**")
        st.image(wordcloud.to_array())

    st.subheader("🧠 Train Sentiment Classifier")
    X = df['cleaned_text']
    y = df['sentiment']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)
    st.success("✅ Model Trained Successfully")

    # Optional prediction
    st.subheader("🔎 Predict Sentiment of a Tweet")
    user_input = st.text_input("Type a tweet here...")
    if user_input:
        cleaned_input = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vec_input)[0]
        st.write(f"**Predicted Sentiment:** `{prediction}`")
