import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

class TextPreprocessor:
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')  # Tokenizer to split text into words
        self.lemmatizer = WordNetLemmatizer()  # Lemmatizer to convert words to their base form
        self.stop_words = set(stopwords.words('english'))  # Set of English stop words

    # Method to preprocess text by cleaning, tokenization, and lemmatization
    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove non-alphanumeric characters
        tokens = self.tokenizer.tokenize(text)  # Tokenize the cleaned text
        # Lemmatize tokens and remove stopwords
        lemmatized = [self.lemmatizer.lemmatize(token.lower())
                      for token in tokens if token.lower() not in self.stop_words]
        return lemmatized  # Return the preprocessed tokens