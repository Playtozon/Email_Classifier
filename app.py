import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import regex as re
from nltk.corpus import stopwords


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

tokenizer = nltk.word_tokenize
def PreProcessedText(text):
    # lowercase + remove URLs first
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    tokens = tokenizer(text)

    result = []
    append = result.append

    for tok in tokens:
        if tok.isalnum() and tok not in stop_words:
            append(ps.stem(tok))

    return " ".join(result)

st.title("Email Classifier")
input_text = st.text_area("Email")

if st.button("Predict"):

# 1preprocess
    transform_text = PreProcessedText(input_text)
# 2vectorize
    vector_input = tfidf.transform([transform_text])
# 3model
    result = model.predict(vector_input)[0]
# 4display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



