# 20_newsgroups_project

## Overview
This project is a text search engine developed in Python that showcases a blend of Information Retrieval (IR) and Natural Language Processing (NLP) techniques. It utilizes a collection of documents from the 20 Newsgroups dataset, enabling users to search for topics and view related documents and keywords.

## Project Structure
- `preprocessor.py`: Handles the cleaning and preparation of text data.
- `indexer.py`: Manages the indexing of documents for quick retrieval.
- `visualization.py`: Includes functions to visualize data as bar chart and word cloud.
- `main.py`: The main executable script that integrates all components.

## Features
- Text preprocessing with tokenization, stop-word removal, and lemmatization using the NLTK library.
- Inverted index and TF-IDF based indexing for efficient document retrieval.
- Semantic search capabilities using the Gensim Word2Vec model.
- Interactive search with user input via the command line.
- Data visualization of search results, including keyword significance and document summaries.

## Output Examples
The system provides interactive visualizations for the user to analyze the search results effectively. Below are examples based on the search query "NASA space mission".

### Bar Chart of Top Keywords by TF-IDF
<img width="400" alt="barchart" src="https://github.com/mhadev0703/20_newsgroups_project/assets/145727959/30f8e837-7b69-4ada-b463-f641c10fb99b">

The bar chart visualizes the top keywords extracted from the documents most relevant to the search query "NASA space mission," indicating their TF-IDF scores.

### Word Cloud Visualization
<img width="400" alt="wordcloud" src="https://github.com/mhadev0703/20_newsgroups_project/assets/145727959/b114e346-3228-43e0-8b87-c1803f656dc2">

The word cloud represents the prominence of keywords within the documents related to the query "NASA space mission," highlighting the most significant terms.

### Top Related Documents
<img width="600" alt="textresult" src="https://github.com/mhadev0703/20_newsgroups_project/assets/145727959/a30e5b49-f519-4b9e-be6e-27f168b9809c">

Displayed are the top 10 documents retrieved in response to the query "NASA space mission".
## Installation and Usage
- Install required Python libraries: `pip install -r requirements.txt` (ensure you have a `requirements.txt` file with all necessary libraries).
- Run the main script: `python main.py`.
- Follow the command line prompts to enter a search query.

## Python Libraries Used
- scikit-learn
- gensim
- nltk
- numpy
- matplotlib
- wordcloud
