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
from io import BytesIO
import pandas as pd

# Ensure necessary NLTK data is downloaded
download('punkt')
download('stopwords')
download('wordnet')
download('punkt_tab')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load saved models and vectorizer - removing cache to ensure no interference
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    loaded_model = joblib.load('random_forest_model.joblib')
    loaded_classes = joblib.load('class_labels.joblib')
    word2vec_model = gensim.models.Word2Vec.load("resume_word2vec.model")
    return tfidf, loaded_model, loaded_classes, word2vec_model

tfidf, loaded_model, loaded_classes, word2vec_model = load_models()

# Define job categories - ensure these exactly match your model's training categories
JOB_CATEGORIES = {
    0: "ACCOUNTANT",
    1: "ADVOCATE",
    2: "AGRICULTURE",
    3: "APPAREL",
    4: "ARTS",
    5: "AUTOMOBILE",
    6: "AVIATION",
    7: "BANKING",
    8: "BPO",
    9: "BUSINESS-DEVELOPMENT",
    10: "CHEF",
    11: "CONSTRUCTION",
    12: "CONSULTANT",
    13: "DESIGNER",
    14: "DIGITAL-MEDIA",
    15: "ENGINEERING",
    16: "FINANCE",
    17: "FITNESS",
    18: "HEALTHCARE",
    19: "HR",
    20: "INFORMATION-TECHNOLOGY",
    21: "PUBLIC-RELATIONS",
    22: "SALES",
    23: "TEACHER"
}

# Debug function to show category mapping
@st.cache_data
def get_category_mapping():
    return pd.DataFrame({
        'Category Code': JOB_CATEGORIES.keys(),
        'Category Name': JOB_CATEGORIES.values()
    })

# Function to preprocess text - identical to Flask version
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text

# Function to calculate the similarity score - identical to Flask version
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

# Function to extract text from PDF - modified to match Flask's text extraction
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Function to check CV match - enhanced debugging
def check_cv_match(pdf_text, provided_job_category, job_description):
    # Debug: Show first 100 chars of processed text
    st.session_state['debug_processed_text'] = preprocess_text(pdf_text)[:100] + "..."
    
    processed_resume = preprocess_text(pdf_text)
    resume_vectorized = tfidf.transform([processed_resume])
    
    # Debug: Show feature names and some values
    if 'debug_features' not in st.session_state:
        feature_names = tfidf.get_feature_names_out()
        st.session_state['debug_features'] = feature_names
        st.session_state['debug_feature_values'] = resume_vectorized.toarray()[0][:20]  # First 20 features
    
    # Get prediction
    predicted_category_index = loaded_model.predict(resume_vectorized)[0]
    predicted_category_name = JOB_CATEGORIES.get(predicted_category_index, f"Unknown ({predicted_category_index})")
    
    # Debug: Show prediction probabilities
    if hasattr(loaded_model, 'predict_proba'):
        probabilities = loaded_model.predict_proba(resume_vectorized)[0]
        st.session_state['debug_probabilities'] = list(zip(range(len(probabilities)), probabilities))
    
    # Check category match
    if predicted_category_index == provided_job_category:
        similarity_score = calculate_similarity(job_description, pdf_text)
        return True, float(similarity_score), predicted_category_name
    else:
        return False, 0.0, predicted_category_name

# Streamlit UI
st.title("CV-Job Description Suitability Analyzer")
st.write("Upload your CV and provide job details to check suitability")

# Debug mode toggle
debug_mode = st.checkbox("Enable debug mode")

if debug_mode:
    st.subheader("Debug Information")
    st.write("Category Mapping:")
    st.dataframe(get_category_mapping())
    
    if 'debug_features' in st.session_state:
        st.write("First 20 feature names and values:")
        debug_df = pd.DataFrame({
            'Feature': st.session_state['debug_features'][:20],
            'Value': st.session_state['debug_feature_values']
        })
        st.dataframe(debug_df)
    
    if 'debug_probabilities' in st.session_state:
        st.write("Prediction probabilities:")
        prob_df = pd.DataFrame(
            st.session_state['debug_probabilities'],
            columns=['Category Code', 'Probability']
        )
        prob_df['Category Name'] = prob_df['Category Code'].map(JOB_CATEGORIES)
        st.dataframe(prob_df.sort_values('Probability', ascending=False))

# File upload
uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

# Job category dropdown
selected_category_name = st.selectbox(
    "Select Job Category",
    options=list(JOB_CATEGORIES.values()),
    index=0
)

# Get numeric code
selected_category_code = [k for k, v in JOB_CATEGORIES.items() if v == selected_category_name][0]

# Job description input
job_description = st.text_area("Paste the Job Description", height=200)

if st.button("Analyze Suitability"):
    if uploaded_file is not None and job_description.strip() != "":
        with st.spinner("Analyzing your CV..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                category_match, similarity_score, predicted_category = check_cv_match(
                    pdf_text, selected_category_code, job_description
                )
                
                # Display results
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Selected Job Category:**")
                    st.info(selected_category_name)
                
                with col2:
                    st.write("**Predicted CV Category:**")
                    if category_match:
                        st.success(predicted_category)
                    else:
                        st.error(predicted_category)
                
                if debug_mode and 'debug_processed_text' in st.session_state:
                    st.write("Processed text sample:", st.session_state['debug_processed_text'])
                
                if category_match:
                    st.metric("Similarity Score", f"{similarity_score:.2%}")
                    
                    if similarity_score > 0.7:
                        st.success("Excellent match!")
                    elif similarity_score > 0.5:
                        st.warning("Good match")
                    else:
                        st.info("Fair match")
                else:
                    st.error("Category Mismatch")
                    st.markdown(f"""
                    **Suggestions:**
                    - Tailor your CV to highlight {selected_category_name} skills
                    - Include keywords from the job description
                    - Review the predicted category for insights
                    """)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                if debug_mode:
                    st.exception(e)
    else:
        st.error("Please upload a PDF file and provide a job description")