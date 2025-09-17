# prepare_models.py
import pyarrow.dataset as ds
import pandas as pd
import pickle
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

# --- CONFIGURATION ---
parquet_path = r'H:\syntorion\pharmabot\pharma_new\final\finalmedication_embeddings_final.parquet'
batch_size = 500

# --- HELPER FUNCTIONS ---
# Ensure you have downloaded the necessary NLTK data first
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize(text):
    """Tokenizes text, removes stopwords, and converts to lowercase."""
    return [
        word for word in word_tokenize(str(text).lower())
        if word.isalpha() and word not in stop_words
    ]

# --- SCRIPT START ---
print("🚀 Starting Part 1: Model Preparation (CPU-Intensive Task)")
start_time = time.time()

# === Step 1: Build Dictionary ===
print("\n⚙️  Building corpus and dictionary from Parquet file...")
scanner = ds.dataset(parquet_path, format='parquet').scanner(batch_size=batch_size)
token_iterator = (tokenize(text) for batch in scanner.to_batches() for text in batch.to_pandas()['text'])

dictionary = Dictionary(token_iterator)
dictionary.save('tfidf_dictionary.gensim')
print(f"✅ Vocabulary dictionary saved to tfidf_dictionary.gensim")

# === Step 2: Train TF-IDF Model ===
print("\n⚙️  Training Gensim TF-IDF model...")
corpus_scanner = ds.dataset(parquet_path, format='parquet').scanner(batch_size=batch_size)
corpus_iterator = (dictionary.doc2bow(tokenize(text)) for batch in corpus_scanner.to_batches() for text in batch.to_pandas()['text'])
tfidf_model = TfidfModel(corpus_iterator)
tfidf_model.save('tfidf_model.gensim')

# === Step 3: Save Models for Ingestion Script ===
with open('sparse_model.pkl', 'wb') as f:
    pickle.dump({
        'dictionary': dictionary,
        'tfidf_model': tfidf_model
    }, f)
print("💾 Gensim TF-IDF model and dictionary saved to sparse_model.pkl")

end_time = time.time()
print(f"\n🎉 Model preparation complete! Time taken: {(end_time - start_time) / 60:.2f} minutes.")