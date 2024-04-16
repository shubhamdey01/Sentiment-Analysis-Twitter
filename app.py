import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import streamlit as st
import re
import nltk
nltk.download('stopwords')

# loading trained model
model = pickle.load(open('twitterModel.pkl', 'rb'))
vectorizer = pickle.load(open('tweetVectorizer.pkl', 'rb'))

def stemVectorize(text):
    engStopwords = stopwords.words('english')
    porterStemmer = PorterStemmer()
    newText = re.sub('[^a-zA-Z]', ' ', text)
    newText = newText.lower()
    newText = newText.split()
    newText = [porterStemmer.stem(word) for word in newText if not word in engStopwords]
    newText = ' '.join(newText)

    newText = vectorizer.transform([newText])

    return newText

def sentimentAnalysis(inputData):
    prediction = model.predict(inputData)
    return prediction[0]
    
def main():
    st.title('Tweet Sentiment Analysis')
    tweet = st.text_input('Tweet to analyze')
    tweet = stemVectorize(tweet)
    if st.button('Analyze'):
        flag = sentimentAnalysis(tweet)
        if flag:
            st.success('Positive Tweet')
        else:
            st.success('Negative Tweet')

if __name__ == '__main__':
    main()