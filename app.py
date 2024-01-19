import streamlit as st 
import pandas as pd
import pickle
import sklearn
from nltk.corpus import stopwords
import nltk 
import string 
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import nltk
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
from PIL import Image
nltk.download('punkt')
nltk.download('stopwords')

#st.image(image, caption='EMAIL')

# Function to transform text.
def transform_text(text):
    text = text.lower()
    tokenized_text = word_tokenize(text)   
    filtered_text = [stemmer.stem(word) for word in tokenized_text if word not in stopwords.words('english') and word.isalnum()]
    text = ' '.join(filtered_text)
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_words)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('Email Spam Classifier')

input_sms = st.text_input('Enter the Message ')

classifier_name = st.selectbox("Select your classifier :-", ['KNeighborsClassifier', 'SVC', 'RandomForestClassifier', 'LogisticRegression', 'GradientBoostingClassifier', 'GaussianNB', 'MultinomialNB', 'BernoulliNB', 'DecisionTreeClassifier'])
model = pickle.load(open(classifier_name+'.pkl', 'rb'))

if st.checkbox("Check me"):
    st.write("You are not Reda")

if st.button('Click to Predict'):
    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])

    # Check if the model is SVC and convert to dense array if necessary
    if classifier_name == 'SVC':
        vector_input = vector_input.toarray()

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')