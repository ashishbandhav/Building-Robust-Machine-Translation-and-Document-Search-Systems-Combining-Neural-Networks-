import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the translation model
@st.cache_resource
def load_translation_model():
    model_name = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Function to translate text
def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Load the document embedding model
@st.cache_resource
def load_search_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Function to find the most relevant document
def search_documents(query, documents, model):
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)
    best_match_index = np.argmax(cosine_scores.numpy())
    return documents[best_match_index], float(cosine_scores[0, best_match_index])

# Load models
tokenizer, translation_model = load_translation_model()
search_model = load_search_model()

# Streamlit app layout
st.title("Machine Translation and Document Search App")
st.header("Translate English Text to French")
input_text = st.text_area("Enter English text for translation:")

if st.button("Translate"):
    if input_text:
        translated_text = translate_text(input_text, tokenizer, translation_model)
        st.success(f"Translated Text: {translated_text}")
    else:
        st.warning("Please enter text to translate.")

st.header("Search for Relevant Documents")
query_text = st.text_input("Enter search query:")
documents = [
    "Machine translation is a sub-field of computational linguistics.",
    "Neural networks are key to advancing artificial intelligence.",
    "Document search systems retrieve relevant information efficiently.",
    "Combining vector space models with neural methods improves search accuracy."
]

if st.button("Search"):
    if query_text:
        best_doc, score = search_documents(query_text, documents, search_model)
        st.success(f"Best Matched Document: {best_doc}\nScore: {score:.2f}")
    else:
        st.warning("Please enter a query to search.")

st.sidebar.header("App Info")
st.sidebar.write("This app combines neural machine translation and document search.")
