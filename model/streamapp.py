import streamlit as st
import joblib
import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from nltk import download
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import fitz  # PyMuPDF for PDF extraction

# Ensure necessary NLTK data is downloaded
download('punkt')
download('stopwords')
download('wordnet')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load models and vectorizer
tfidf = joblib.load('tfidf_vectorizer.joblib')
loaded_model = joblib.load('random_forest_model.joblib')
loaded_classes = joblib.load('class_labels.joblib')  # Loaded classes, e.g., ["ACCOUNTANT", "IT", "HR"]
word2vec_model = gensim.models.Word2Vec.load("resume_word2vec.model")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

# Function to calculate the similarity score
def calculate_similarity(job_description, resume_text):
    job_tokens = word_tokenize(job_description.lower())
    job_vectors = [word2vec_model.wv[token] for token in job_tokens if token in word2vec_model.wv]
    job_vector = np.mean(job_vectors, axis=0) if job_vectors else np.zeros(100)

    resume_tokens = word_tokenize(resume_text.lower())
    resume_vectors = [word2vec_model.wv[token] for token in resume_tokens if token in word2vec_model.wv]
    resume_vector = np.mean(resume_vectors, axis=0) if resume_vectors else np.zeros(100)

    if np.any(job_vector) and np.any(resume_vector):
        similarity_score = cosine_similarity([job_vector], [resume_vector])[0][0]
    else:
        similarity_score = 0.0

    return similarity_score

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text()
    return text

# Function to check CV match with job description
def check_cv_match(pdf_text, provided_job_category, job_description):
    processed_resume = preprocess_text(pdf_text)
    resume_vectorized = tfidf.transform([processed_resume])

    # Get prediction
    predicted_category_index = loaded_model.predict(resume_vectorized)[0]

    # Check if the predicted category matches the provided job category
    if predicted_category_index == provided_job_category:
        similarity_score = calculate_similarity(job_description, pdf_text)
        return True, float(similarity_score)
    else:
        return False, 0.0

# Streamlit app
st.title("CV Matching with Job Description")

st.write("Upload your CV (PDF format), select a job category, and provide the job description to check the match.")

# File upload
uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

# Job categories (replace with your actual class labels)
job_categories = {i: label for i, label in enumerate(loaded_classes)}

# Input fields
job_category_index = st.selectbox("Select Job Category", options=list(job_categories.keys()), format_func=lambda x: job_categories[x])
job_description = st.text_area("Enter Job Description")

# Submit button
if st.button("Check CV Match"):
    if uploaded_file and job_description:
        # Extract text from the uploaded file
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Check CV match
        category_match, similarity_score = check_cv_match(pdf_text, job_category_index, job_description)

        st.write(f"**Category Match:** {category_match}")
        st.write(f"**Similarity Score:** {similarity_score:.2f}")
    else:
        st.error("Please upload a resume and provide all required inputs.")
