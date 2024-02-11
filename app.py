import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

tfidf = pickle.load(open('D:\Pycharm Projects\sms-spam-classification\\vectorizer.pkl', 'rb'))
model = pickle.load(open('D:\Pycharm Projects\sms-spam-classification\model.pkl', 'rb'))
def transform_text(text):
    text = text.lower()

    words = word_tokenize(text)

    stopwords_set = set(stopwords.words('english'))

    words = [word for word in words if word not in stopwords_set and word not in string.punctuation]

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word) for word in words]

    ps = PorterStemmer()

    words = [ps.stem(word) for word in words]

    cleaned_text = ' '.join(words)

    return cleaned_text

st.title('Email/SMS Spam Classifier')
input_sms = st.text_area('Enter the message')

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)
    print(transformed_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    print(result)
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')






