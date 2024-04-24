from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class Indexer:
    def __init__(self):
        self.tok2idx = {}  # Mapping from token to a unique index
        self.idx2tok = {}  # Reverse mapping from index to token
        self.postings_lists = defaultdict(list)  # Inverted index data structure

    # Method to add a document to the index
    def index_document(self, document, doc_id):
        for term in document:
            if term not in self.tok2idx:  # If term is not in index, add it
                self.tok2idx[term] = len(self.tok2idx)
                self.idx2tok[self.tok2idx[term]] = term
            # Append the document ID to the postings list of the term
            self.postings_lists[self.tok2idx[term]].append(doc_id)

    # Method to search for documents that contain any of the query terms
    def search(self, query_terms):
        results = Counter()  # Counter to keep track of document frequencies
        for term in query_terms:
            term_id = self.tok2idx.get(term, None)  # Get the index of the term
            if term_id is not None:
                docs_with_term = self.postings_lists[term_id]  # Retrieve documents containing the term
                results.update(docs_with_term)  # Update the results counter
        return results  # Return the results as a Counter object


class TfidfIndexer:
    def __init__(self, max_features=10000, max_df=0.5, min_df=2, stop_words='english'):
        self.vectorizer = TfidfVectorizer(max_features=max_features, max_df=max_df, min_df=min_df, stop_words=stop_words)
        self.tfidf_matrix = None
        self.feature_names = None

    def build_index(self, documents):
        # Fit the vectorizer to the documents to build the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def extract_keywords(self, documents, top_n=10):
        # Build the index if it has not been done already
        if not self.tfidf_matrix:
            self.build_index(documents)
        sums = self.tfidf_matrix.sum(axis=0)
        data = [(term, sums[0, col]) for col, term in enumerate(self.feature_names)]
        # Sort the terms by their score and return the top_n terms
        return sorted(data, key=lambda x: x[1], reverse=True)[:top_n]
