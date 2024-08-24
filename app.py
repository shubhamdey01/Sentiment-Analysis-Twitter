import pickle
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import streamlit as st
import nltk

# loading trained model
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
model = pickle.load(open('models/model.pkl', 'rb'))

# Pre-Processing Tweets
# creating stemmer
stemmer = PorterStemmer()

# defining regex patterns
url = r'((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)'
user = r'@[^\s]+'
alpha = r'[^a-zA-Z]'
seq = r'(.)\1\1+'
seqReplace = r'\1\1'

# Cache the download of NLTK resources
@st.cache_resource
def downloadResources():
    nltk.download('punkt_tab')

def preprocessData(text):
    # lowercasing
    text = text.lower()

    # removing URL's
    text = re.sub(url, ' ', text)

    # removing mentions @USER
    text = re.sub(user, ' ', text)

    # remove all non alphabets
    text = re.sub(alpha, ' ', text)

    # replace more than 2 consecutive letters by 2 letters
    text = re.sub(seq, seqReplace, text)

    # remove extra whitespaces
    text = text.strip()
    text = re.sub(r' +', ' ', text)

    tokens = []

    for word in word_tokenize(text):
        # if word not in stopwordEng:
        tokens.append(stemmer.stem(word))   # stemming

    return ' '.join(tokens)


def main():
    # page configuration
    st.set_page_config(
        page_title = 'Sentiment Analysis',
        page_icon = 'images/pageLogo.jpeg',
        layout = 'centered'
    )

    # page title
    cols = st.columns([7, 1])
    with cols[0]:
        st.title('Sentiment Analysis of Tweets')
    with cols[1]:
        st.image('images/twitterSentiment.jpg', width=100)

    # input
    tweet = st.text_area('Tweet to analyze', height=200)

    # making prediction
    downloadResources()
    tweet = preprocessData(tweet)
    tweet = vectorizer.transform([tweet])

    if st.button('Analyze'):
        flag = model.predict(tweet)[0]
        if flag:
            st.success('Positive Tweet')
        else:
            st.error('Negative Tweet')


if __name__ == '__main__':
    main()