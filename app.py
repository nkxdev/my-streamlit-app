import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import extract_pdf, extract_docx
from skills import SKILLS

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# -------------------- UI DESIGN --------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get skill insights 🚀")

# -------------------- GRAPH FUNCTION --------------------
def plot_scores(names, scores):
    fig, ax = plt.subplots(figsize=(10,5))

    ax.barh(names, scores)
    ax.set_xlabel("Score")
    ax.set_title("Resume Ranking")

    # Show values on bars
    for i, v in enumerate(scores):
        ax.text(v + 0.5, i, str(v), va='center')

    st.pyplot(fig)

# -------------------- INPUT --------------------
uploaded_files = st.file_uploader("📄 Upload Multiple Resumes", accept_multiple_files=True)
job_desc = st.text_area("🧾 Paste Job Description")

# -------------------- SKILL FUNCTION --------------------
def extract_skills(text):
    found = []
    for skill in SKILLS:
        if skill.lower() in text:
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

        st.progress(int(r["score"]))
        st.write(f"✅ Score: {r['score']} %")
        st.write(f"🧠 Skills: {r['skills']}")
        st.write(f"❌ Missing Skills: {r['missing']}")

        if r["missing"]:
            st.info(f"💡 Improve by adding: {', '.join(r['missing'])}")
        else:
            st.success("🎉 Perfect Skill Match!")

        if r["score"] > 70:
            st.success("🎯 Selected")
        else:
            st.error("❌ Not Selected")

        st.write("---")

    # -------------------- GRAPH --------------------
    st.subheader("📈 Resume Ranking Graph")

    names = [r["name"] for r in results]
    scores = [r["score"] for r in results]

    # Sort for better visualization
    sorted_data = sorted(zip(scores, names), reverse=True)
    scores, names = zip(*sorted_data)

    plot_scores(names, scores)

else:
    st.warning("⚠️ Please upload resume and enter job description")
