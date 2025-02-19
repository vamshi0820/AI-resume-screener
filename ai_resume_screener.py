import streamlit as st
import PyPDF2
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def preprocess_text(text):
    """Tokenizes and lemmatizes the text."""
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def extract_skills(text):
    """Extracts key skills from the given text using NLP."""
    skill_keywords = {"python", "java", "sql", "machine learning", "deep learning", "data analysis",
                      "nlp", "cloud computing", "docker", "aws", "react", "angular", "django", "flask"}
    text_tokens = set(text.lower().split())
    matched_skills = skill_keywords.intersection(text_tokens)
    return list(matched_skills)

def extract_experience(text):
    """Extracts years of experience from the resume."""
    match = re.search(r'(\d+)\+?\s*(years|year)', text.lower())
    if match:
        return int(match.group(1))
    return 0

def compute_similarity(resume_text, job_desc_text):
    """Computes similarity between resume and job description using TF-IDF."""
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity = cosine_similarity(vectors)[0][1]
    return round(similarity * 100, 2)

def main():
    st.title("🔍 AI-Powered Resume Screener")
    st.subheader("📄 Upload Resume & Paste Job Description")

    job_desc = st.text_area("📌 Paste the Job Description")
    resume_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])

    if resume_file and job_desc:
        resume_text = extract_text_from_pdf(resume_file)
        resume_processed = preprocess_text(resume_text)
        job_desc_processed = preprocess_text(job_desc)

        score = compute_similarity(resume_processed, job_desc_processed)
        resume_skills = extract_skills(resume_text)
        job_desc_skills = extract_skills(job_desc)
        matching_skills = list(set(resume_skills) & set(job_desc_skills))

        experience_years = extract_experience(resume_text)

        # Display Match Score
        st.subheader("✅ Match Score")
        st.write(f"**Resume matches {score}% with the job description.**")

        # Skill Matching
        st.subheader("🎯 Skill Matching")
        st.write(f"**Skills in Resume:** {', '.join(resume_skills) if resume_skills else 'No skills detected'}")
        st.write(f"**Skills in Job Description:** {', '.join(job_desc_skills) if job_desc_skills else 'No skills detected'}")
        st.write(f"**Matching Skills:** {', '.join(matching_skills) if matching_skills else 'No matching skills found'}")

        # Experience Analysis
        st.subheader("📊 Experience Analysis")
        if experience_years:
            st.write(f"**Experience in Resume:** {experience_years} years")
        else:
            st.write("**No experience details detected in the resume.**")

        # Overall Recommendation
        st.subheader("📌 Final Recommendation")
        if score > 70 and matching_skills:
            st.success("🚀 Great match! This candidate is a strong fit.")
        elif score > 50:
            st.warning("⚠️ Moderate match. Requires further evaluation.")
        else:
            st.error("❌ Low match. Consider a better fit.")

if __name__ == "__main__":
    main()

