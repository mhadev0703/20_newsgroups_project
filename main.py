from sklearn.datasets import fetch_20newsgroups
from preprocessor import TextPreprocessor
from indexer import Indexer, TfidfIndexer
from visualization import visualize_bar_chart, visualize_word_cloud
from gensim.models import Word2Vec
from collections import Counter

import ssl
import numpy as np

# Bypass SSL Certificate Verification
ssl._create_default_https_context = ssl._create_unverified_context

class TextSearchEngine:
    def __init__(self):
        self.indexer = Indexer()  # Indexer to handle document indexing
        self.preprocessor = TextPreprocessor()  # TextPreprocessor to preprocess the documents
        self.documents = []  # list to store preprocessed documents for Word2Vec
        self.model = None  # Placeholder for the Word2Vec model
        self.tfidf_indexer = TfidfIndexer()  # Tfidf indexer

    # Method to add and index a document after preprocessing
    def add_document(self, document, doc_id):
        preprocessed_text = self.preprocessor.preprocess(document)
        self.indexer.index_document(preprocessed_text, doc_id)
        self.documents.append(" ".join(preprocessed_text))  # for tf-idf

    def train_word_vectors(self, corpus):
        self.model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, workers=4)

    def search(self, query):
        preprocessed_query = self.preprocessor.preprocess(query)
        # Filter out words that are not in the Word2Vec model's vocabulary
        query_words_in_vocab = [word for word in preprocessed_query if word in self.model.wv]

        if not query_words_in_vocab:
            raise ValueError("None of the query words are in the Word2Vec vocabulary.")

        # Use vector averaging for multi-word queries
        query_vector = np.mean([self.model.wv[word] for word in query_words_in_vocab], axis=0)

        # If the query_vector is empty or all zeros, return no results
        if np.all(query_vector == 0):
            return []

        similar_words = self.model.wv.similar_by_vector(query_vector, topn=10)
        results = Counter()
        for word, _ in similar_words:
            if word in self.indexer.tok2idx:
                doc_ids = self.indexer.postings_lists[self.indexer.tok2idx[word]]
                for doc_id in doc_ids:
                    results[doc_id] += 1  # Boost scores for semantic matches

        return results.most_common(10)

    def fetch_document_details(self, doc_ids, num_words=50):
        documents = []
        for doc_id in doc_ids:
            doc_content = newsgroups.data[doc_id]  # Fetch the full document text
            doc_summary = ' '.join(doc_content.split()[:num_words])  # Create a summary from the first 50 words
            documents.append((doc_id, doc_summary))
        return documents

if __name__ == '__main__':
    try:
        # Load the dataset, removing headers, footers, and quotes
        newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        engine = TextSearchEngine()

        # Preprocess the documents and train the Word2Vec model
        preprocessed_corpus = [engine.preprocessor.preprocess(doc) for doc in newsgroups.data]
        engine.train_word_vectors(preprocessed_corpus)

        # Index each document
        for idx, document in enumerate(newsgroups.data):
            engine.add_document(document, idx)

        # Get a query from the user and perform a search
        query = input("Please enter your search query: ")
        search_results = engine.search(query)

        # Extract the documents associated with the search results
        top_documents = [newsgroups.data[doc_id] for doc_id, _ in search_results]

        # Extract keywords from the top documents and visualize
        top_keywords = engine.tfidf_indexer.extract_keywords(top_documents)
        keywords, scores = zip(*top_keywords)
        visualize_bar_chart(keywords, scores)
        visualize_word_cloud(keywords, scores)

        # Display summaries of the top related documents
        print("\nTop 10 Related Documents:\n")
        for idx, (doc_id, summary) in enumerate(engine.fetch_document_details([doc_id for doc_id, _ in search_results]),
                                                1):
            print(f"Doc {idx} (ID: {doc_id}): {summary}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
