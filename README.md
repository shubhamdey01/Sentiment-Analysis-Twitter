# Sentiment Analysis of Tweets

<img src="/images/sentimentAnalysis.png">
<!-- <img src="/images/twitterSentiment.jpg"> -->

## Introduction

Natural Language Processing (NLP) is key in data science, with sentiment analysis being a major application. This project focuses on classifying tweets as positive or negative using machine learning. We begin by preprocessing and cleaning the text, then convert it into numerical vectors for analysis. Machine learning models are used to predict tweet sentiments, showcasing the efficiency and practical application of NLP in handling large text datasets.

## Problem Statement

Let's first clarify the problem statement, as understanding the objective is crucial before working with the dataset. The task is to identify hate speech in tweets. For the purpose of this project, a tweet is classified as hate speech if it contains harmful or offensive language. The objective is to accurately distinguish between tweets that contain such language and those that do not. Given a labeled dataset, where a label of ‘0’ indicates the presence of hate speech and ‘4’ indicates its absence, our goal is to predict the correct labels for a test dataset. The performance of the model will be evaluated using the F1-Score, which balances precision and recall.

<!-- ![images](Twitter-sentiment-analysis-1.jpg) -->


## Vectorizers 

Vectorizers are crucial for machine learning as they convert text from strings into numerical vectors, which are used for prediction. Below are the vectorizers employed to transform text into a mathematical representation.

* [__Count Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [__Tfidf Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## Machine Learning Models

Machine learning models are essential for analyzing numerical data and making predictions. Once the text is converted into vectors, these models use the data to classify or predict outcomes. Below are the machine learning models applied to interpret the vectorized text and perform sentiment analysis.

* [__BernoulliNB__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
* [__MultinomialNB__](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
* [__LogisticRegression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [__LinearSVC__](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

## Dataset

Access the dataset here: [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

## Text Preprocessing and Cleaning

Text preprocessing and cleaning are crucial steps in preparing text data for analysis. This process ensures that the text is formatted correctly for machine learning models. Key tasks include:

- **Removing Noise**: Eliminating irrelevant characters, symbols, extra whitespace, mentions and links.
- **Tokenization**: Breaking text into individual words or tokens.
- **Lowercasing**: Converting all text to lowercase for consistency.
- **Removing Stopwords**: Filtering out common words (e.g., "the," "and") that do not add significant meaning.
- **Stemming/Lemmatization**: Reducing words to their base or root form (e.g., "running" to "run").

Additional cleaning involves:
- **Handling Special Characters**: Removing or replacing special characters and punctuation.
- **Correcting Spelling Errors**: Fixing typos to improve text consistency.
- **Removing Duplicates**: Eliminating duplicate entries to avoid redundancy.

These steps transform raw text into a structured format, improving the quality and accuracy of machine learning models.


## Exploratory Data Analysis (EDA)

After performing __EDA__, it can be observed that the dataset contains an equal number of positive and negative tweets.

<img src="/images/dataDistribution.png">
<br><br>

Using __Word Clouds__, it was observed that some words are common in both positive and negative tweets. However, words such as 'love,' 'awesome,' and 'right' were most frequently used in positive tweets. In contrast, the negative word cloud showed that words such as 'bad,' 'sad,' and 'miss' were most frequently used

<img src="/images/positive.png"> <br>
<img src="/images/negative.png">

## Results 

| ID      | Text Extraction | ML Model           | Accuracy | f1-score (class 0) | f1-score (class 1) | Training Time |
|---------|-----------------|--------------------|----------|--------------------|--------------------|---------------|
| Model-1 | CountVectorizer | BernoulliNB        | 0.79     | 0.79               | 0.79               | 0.93 sec      |
| Model-2 | CountVectorizer | MultinomialNB      | 0.8      | 0.8                | 0.79               | 0.87 sec      |
| Model-3 | CountVectorizer | LogisticRegression | 0.8      | 0.8                | 0.81               | 531.09 sec    |
| Model-4 | CountVectorizer | LinearSVC          | 0.78     | 0.78               | 0.78               | 680.62 sec    |
| Model-5 | TfidfVectorizer | BernoulliNB        | 0.79     | 0.79               | 0.79               | 1.07 sec      |
| Model-6 | TfidfVectorizer | MultinomialNB      | 0.8      | 0.8                | 0.79               | 0.71 sec      |
| Model-7 | TfidfVectorizer | LogisticRegression | 0.82     | 0.82               | 0.82               | 22.81sec    |
| Model-8 | TfidfVectorizer | LinearSVC          | 0.81     | 0.81               | 0.81               | 52.5 sec      |


Logistic Regression combined with the TF-IDF Vectorizer has shown the best performance for our dataset. This combination effectively captures the relevance of terms in the text and accurately classifies sentiments. Therefore, we will use this model setup for deployment, as it provides a robust solution for sentiment analysis.

<img src='/images/model7.png'>

## Getting Started

Follow these simple steps to set up and start working on the project:

1. __Clone the Repository__:
   ```bash
   git clone https://github.com/shubhamdey01/Sentiment-Analysis-Twitter.git
   ```
   
2. __Navigate to the Project Directory__:
   ```bash
   cd Twitter-Sentiment-Analysis-with-Python
   ```
   
3. __Check Python Version__: Ensure that you have Python 3.12 installed. You can find the required packages in the `requirements.txt` file.

4. __Create a Virtual Environment__ (recommended for project isolation):
   ```bash
   python3 -m venv venv
   ```
   
5. __Activate the Virtual Environment__:

   - For macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   
   - For Windows:
     ```bash
     venv\Scripts\activate
     ```

6. __Install Dependencies__ from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

7. __Run the program__:
   Start the streamlit server:
   ```bash
   streamlit run app.py
   ```

## Future Work

Possible enhancements for the project include:

- Fine-tuning a transformer model like BERT for better accuracy.
- Implementing real-time sentiment analysis using live Twitter data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


