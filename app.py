import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import extract_pdf, extract_docx
from skills import SKILLS

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.big-title {
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}
.card {
    padding:15px;
    border-radius:10px;
    background-color:#1e1e1e;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<p class="big-title">🚀 AI Resume Screening System</p>', unsafe_allow_html=True)

# -------------------- INPUT --------------------
uploaded_files = st.file_uploader("📄 Upload Multiple Resumes", accept_multiple_files=True)
job_desc = st.text_area("🧾 Paste Job Description")

# -------------------- SKILL FUNCTION --------------------
def extract_skills(text):
    found = []
    for skill in SKILLS:
        if skill in text:
            found.append(skill)
    return found

# -------------------- MAIN LOGIC --------------------
if uploaded_files and job_desc:

    results = []

    for file in uploaded_files:

        # Extract text
        if file.type == "application/pdf":
            text = extract_pdf(file)
        else:
            text = extract_docx(file)

        text = text.lower()
        job_desc_lower = job_desc.lower()

        # TF-IDF similarity
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text, job_desc_lower])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

        # Skills
        skills_found = extract_skills(text)
        job_skills = extract_skills(job_desc_lower)

        missing_skills = list(set(job_skills) - set(skills_found))

        results.append({
            "name": file.name,
            "score": round(score * 100, 2),
            "skills": skills_found,
            "missing": missing_skills
        })

    # -------------------- SORTING --------------------
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # -------------------- TABLE --------------------
    st.subheader("📊 Candidate Ranking Table")
    df = pd.DataFrame(results)
    st.dataframe(df)

    # -------------------- DETAILED OUTPUT --------------------
    st.subheader("📄 Detailed Analysis")

    for r in results:
        st.markdown(f"### 📄 {r['name']}")

        # Progress Bar
        st.progress(int(r["score"]))

        st.write(f"✅ Score: {r['score']} %")
        st.write(f"🧠 Skills: {r['skills']}")
        st.write(f"❌ Missing Skills: {r['missing']}")

        # Suggestions
        if r["missing"]:
            st.info(f"💡 Improve by adding: {', '.join(r['missing'])}")
        else:
            st.success("🎉 Perfect Skill Match!")

        # Selection
        if r["score"] > 70:
            st.success("🎯 Selected")
        else:
            st.error("❌ Not Selected")

        st.write("---")

    # -------------------- GRAPH --------------------
    st.subheader("📈 Resume Ranking Graph")

    names = [r["name"] for r in results]
    scores = [r["score"] for r in results]

    plt.figure()
    plt.bar(names, scores)
    plt.xlabel("Candidates")
    plt.ylabel("Score")
    plt.title("Resume Ranking")

    st.pyplot(plt)

else:
    st.warning("⚠️ Please upload resume and enter job description")