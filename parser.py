# pip install streamlit google-generativeai pdfminer.six docx2txt spacy nltk scikit-learn
# login to google ai studio
# get your API key
import streamlit as st
import google.generativeai as genai
import pdfminer.high_level
import docx2txt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Configure Google Gemini API
genai.configure(api_key="xxxxxxxxxxxxxxxxxx")

# Function to extract text from resumes
def extract_text(file_path):
    if file_path.name.endswith(".pdf"):
        return pdfminer.high_level.extract_text(file_path)
    elif file_path.name.endswith(".docx"):
        return docx2txt.process(file_path)
    return None

# Function to extract keywords from text
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    return " ".join(set(keywords))  # Remove duplicates

# Function to extract resume details using Google Gemini API
def extract_resume_details(text):
    prompt = f"""
    Extract key details from the following resume:
    {text}
    Identify Name, Email, Phone, Skills, and Work Experience as JSON.
    """
    response = genai.generate_text(prompt)
    return response.text  # Parsed structured resume details

# Function to calculate similarity between job description and resumes
def calculate_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# Streamlit UI
st.set_page_config(layout="wide")  # Set layout to wide mode

st.title("📄 AI-Powered Resume Screening & Ranking")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Job Description")
    job_desc = st.text_area("Enter Job Description Here", height=300)

with col2:
    st.subheader("📑 Ranked Resumes")

# Resume upload section
uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_desc:
        st.error("Please enter a job description before ranking resumes.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        results = []
        
        with col2:
            for file in uploaded_files:
                resume_text = extract_text(file)
                resume_keywords = extract_keywords(resume_text)  # Extract resume keywords
                job_keywords = extract_keywords(job_desc)  # Extract job description keywords
                
                match_score = calculate_similarity(resume_keywords, job_keywords)  # Compute similarity
                
                results.append((file.name, match_score, resume_keywords))
            
            # Sort resumes by match score (highest first)
            results.sort(key=lambda x: x[1], reverse=True)

            st.subheader("📊 Ranked Resumes")
            for rank, (name, score, keywords) in enumerate(results, 1):
                st.markdown(f"**{rank}. {name}** - Match Score: `{score:.2f}`")
                with st.expander(f"🔍 View Extracted Details ({name})"):
                    st.write("**Extracted Keywords & Skills:**")
                    st.write(keywords)
