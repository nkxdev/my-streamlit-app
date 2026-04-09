import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from wordcloud import WordCloud

from utils import extract_pdf, extract_docx
from skills import SKILLS

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Resume Analyzer Pro", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- ADVANCED CSS STYLING --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: #0f0f23;
    }
    
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .missing-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .candidate-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .candidate-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .progress-container {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }
    
    .selected {
        border-left: 4px solid #00ff88;
    }
    
    .rejected {
        border-left: 4px solid #ff416c;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.05);
    }
    
    .upload-box {
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-item {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .stat-label {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR CONFIGURATION --------------------
with st.sidebar:
    st.markdown("### ⚙️ Analysis Settings")
    
    similarity_threshold = st.slider(
        "Match Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=70,
        help="Minimum score required for candidate selection"
    )
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Standard", "Strict", "Lenient"],
        help="Strict: Higher standards, Lenient: Lower standards"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Export Options")
    
    export_format = st.selectbox(
        "Export Format",
        ["Excel", "CSV", "PDF Report"]
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Advanced Features")
    enable_wordcloud = st.toggle("Generate Word Clouds", value=True)
    enable_semantic = st.toggle("Semantic Analysis", value=True)
    enable_ats = st.toggle("ATS Compatibility Check", value=True)

# -------------------- HEADER --------------------
st.markdown("<h1>🤖 AI Resume Analyzer Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Next-Generation Resume Screening & Candidate Ranking System</p>", unsafe_allow_html=True)

# -------------------- ADVANCED FUNCTIONS --------------------
def advanced_skill_extraction(text):
    """Enhanced skill extraction with context awareness"""
    found_skills = []
    skill_scores = {}
    
    text_lower = text.lower()
    
    # Exact matches
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found_skills.append(skill)
            skill_scores[skill] = len(matches) * 10  # Weight by frequency
    
    # Related skills detection
    skill_synonyms = {
        "python": ["py", "django", "flask", "pandas", "numpy"],
        "javascript": ["js", "node.js", "react", "vue", "angular"],
        "machine learning": ["ml", "deep learning", "ai", "tensorflow", "pytorch"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes"]
    }
    
    for main_skill, synonyms in skill_synonyms.items():
        for syn in synonyms:
            if syn in text_lower and main_skill not in found_skills:
                found_skills.append(main_skill)
                skill_scores[main_skill] = 5
    
    return found_skills, skill_scores

def calculate_experience_score(text):
    """Calculate experience score based on years mentioned"""
    years_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience'
    matches = re.findall(years_pattern, text.lower())
    
    if matches:
        years = [int(m) for m in matches]
        avg_years = sum(years) / len(years)
        return min(avg_years * 10, 100)  # Cap at 100
    return 0

def calculate_education_score(text):
    """Score based on education level"""
    education_scores = {
        "phd": 100, "doctorate": 100, "ph.d": 100,
        "masters": 80, "mba": 80, "m.s": 80, "m.sc": 80,
        "bachelor": 60, "b.s": 60, "b.sc": 60, "b.tech": 60, "be": 60,
        "associate": 40
    }
    
    text_lower = text.lower()
    max_score = 0
    
    for edu, score in education_scores.items():
        if edu in text_lower:
            max_score = max(max_score, score)
    
    return max_score

def semantic_similarity(text1, text2):
    """Enhanced similarity with multiple algorithms"""
    # TF-IDF Similarity
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    
    try:
        vectors = vectorizer.fit_transform([text1, text2])
        tfidf_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except:
        tfidf_sim = 0
    
    # Keyword overlap
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    jaccard = len(words1 & words2) / len(words1 | words2) if (words1 | words2) else 0
    
    # Weighted combination
    final_score = (tfidf_sim * 0.7) + (jaccard * 0.3)
    return final_score

def generate_wordcloud(text, title):
    """Generate word cloud visualization"""
    wordcloud = WordCloud(
        width=400, 
        height=200, 
        background_color='rgba(255,255,255,0)',
        colormap='cool',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=14, pad=20)
    return fig

def create_radar_chart(scores_dict):
    """Create radar chart for candidate profile"""
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    values += values[:1]  # Complete the circle
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#00d4ff')
    ax.fill(angles, values, alpha=0.25, color='#00d4ff')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='white')
    ax.set_ylim(0, 100)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    
    return fig

def ats_check(text):
    """Check ATS compatibility"""
    issues = []
    score = 100
    
    # Check for common ATS issues
    if len(text) < 200:
        issues.append("Resume too short")
        score -= 20
    
    if re.search(r'[^\x00-\x7F]+', text):
        issues.append("Special characters detected")
        score -= 10
    
    if text.count('\n\n\n') > 5:
        issues.append("Excessive blank lines")
        score -= 5
    
    # Check for standard sections
    sections = ["experience", "education", "skills"]
    for section in sections:
        if section not in text.lower():
            issues.append(f"Missing '{section}' section")
            score -= 10
    
    return max(score, 0), issues

# -------------------- MAIN INPUT SECTION --------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Resumes")
    with st.container():
        uploaded_files = st.file_uploader(
            "Drop resumes here or click to upload", 
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'txt'],
            help="Supported formats: PDF, DOCX, DOC, TXT"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

with col2:
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area(
        "Paste detailed job description...",
        height=200,
        placeholder="Enter job requirements, skills needed, experience level..."
    )
    
    # Quick skills input
    st.markdown("### 🎯 Required Skills (Optional)")
    custom_skills = st.text_input(
        "Add specific skills (comma-separated)",
        placeholder="e.g., React, Node.js, AWS..."
    )

# -------------------- ANALYSIS LOGIC --------------------
if uploaded_files and job_desc:
    
    # Parse custom skills
    if custom_skills:
        additional_skills = [s.strip() for s in custom_skills.split(',')]
        SKILLS.extend(additional_skills)
    
    results = []
    all_texts = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Analyzing {file.name}... ({idx+1}/{len(uploaded_files)})")
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Extract text
        try:
            if file.type == "application/pdf":
                text = extract_pdf(file)
            elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                              "application/msword"]:
                text = extract_docx(file)
            else:
                text = file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            continue
        
        all_texts.append(text)
        text_lower = text.lower()
        job_desc_lower = job_desc.lower()
        
        # Calculate scores
        semantic_score = semantic_similarity(text_lower, job_desc_lower) * 100
        
        # Skills analysis
        resume_skills, skill_scores = advanced_skill_extraction(text_lower)
        job_skills, _ = advanced_skill_extraction(job_desc_lower)
        
        missing_skills = list(set(job_skills) - set(resume_skills))
        extra_skills = list(set(resume_skills) - set(job_skills))
        matching_skills = list(set(resume_skills) & set(job_skills))
        
        skill_match_ratio = len(matching_skills) / len(job_skills) if job_skills else 0
        skill_score = skill_match_ratio * 100
        
        # Experience & Education
        exp_score = calculate_experience_score(text_lower)
        edu_score = calculate_education_score(text_lower)
        
        # ATS Score
        ats_score, ats_issues = ats_check(text) if enable_ats else (100, [])
        
        # Weighted final score
        weights = {
            "semantic": 0.4,
            "skills": 0.3,
            "experience": 0.15,
            "education": 0.1,
            "ats": 0.05
        }
        
        if analysis_mode == "Strict":
            weights["semantic"] = 0.5
            weights["skills"] = 0.35
        elif analysis_mode == "Lenient":
            weights["semantic"] = 0.3
            weights["skills"] = 0.25
        
        final_score = (
            semantic_score * weights["semantic"] +
            skill_score * weights["skills"] +
            exp_score * weights["experience"] +
            edu_score * weights["education"] +
            ats_score * weights["ats"]
        )
        
        results.append({
            "name": file.name,
            "final_score": round(final_score, 2),
            "semantic_score": round(semantic_score, 2),
            "skill_score": round(skill_score, 2),
            "experience_score": round(exp_score, 2),
            "education_score": round(edu_score, 2),
            "ats_score": round(ats_score, 2),
            "skills_found": resume_skills,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "extra_skills": extra_skills,
            "ats_issues": ats_issues,
            "text": text,
            "skill_details": skill_scores
        })
    
    status_text.empty()
    progress_bar.empty()
    
    # Sort results
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    # -------------------- DASHBOARD STATS --------------------
    st.markdown("---")
    st.markdown("### 📊 Analysis Dashboard")
    
    cols = st.columns(4)
    metrics = [
        ("Total Candidates", len(results), "👥"),
        ("Average Score", f"{sum(r['final_score'] for r in results)/len(results):.1f}%", "📈"),
        ("Selected", sum(1 for r in results if r["final_score"] >= similarity_threshold), "✅"),
        ("Top Match", f"{results[0]['final_score']:.1f}%" if results else "0%", "🎯")
    ]
    
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-value">{icon} {value}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # -------------------- RANKING TABLE --------------------
    st.markdown("---")
    st.markdown("### 🏆 Candidate Rankings")
    
    df_data = []
    for r in results:
        status = "✅ Selected" if r["final_score"] >= similarity_threshold else "❌ Not Selected"
        df_data.append({
            "Rank": 0,
            "Candidate": r["name"],
            "Overall": f"{r['final_score']}%",
            "Semantic": f"{r['semantic_score']}%",
            "Skills": f"{r['skill_score']}%",
            "Exp": f"{r['experience_score']}%",
            "Edu": f"{r['education_score']}%",
            "ATS": f"{r['ats_score']}%",
            "Status": status
        })
    
    for i, row in enumerate(df_data):
        row["Rank"] = i + 1
    
    df = pd.DataFrame(df_data)
    
    # Color coding
    def color_score(val):
        try:
            num = float(val.replace('%', ''))
            if num >= 80: return 'background-color: rgba(0,255,136,0.3); color: #00ff88; font-weight: bold'
            elif num >= 60: return 'background-color: rgba(255,193,7,0.3); color: #ffc107; font-weight: bold'
            else: return 'background-color: rgba(255,65,108,0.3); color: #ff416c; font-weight: bold'
        except:
            return ''
    
    styled_df = df.style.applymap(color_score, subset=["Overall", "Semantic", "Skills", "Exp", "Edu", "ATS"])
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    # -------------------- DETAILED ANALYSIS --------------------
    st.markdown("---")
    st.markdown("### 🔍 Detailed Candidate Analysis")
    
    # Create tabs for each candidate
    tabs = st.tabs([f"#{i+1} {r['name'][:20]}..." for i, r in enumerate(results)])
    
    for tab, result in zip(tabs, results):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Candidate Header
                is_selected = result["final_score"] >= similarity_threshold
                status_class = "selected" if is_selected else "rejected"
                
                st.markdown(f"""
                    <div class="candidate-card {status_class}">
                        <h3 style="color: #00d4ff; margin-bottom: 1rem;">📄 {result['name']}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Overall Score with animated progress
                st.markdown(f"### Overall Match: {result['final_score']}%")
                
                # Color based on score
                bar_color = "#00ff88" if result["final_score"] >= 80 else "#ffc107" if result["final_score"] >= 60 else "#ff416c"
                
                st.markdown(f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {result['final_score']}%; background: {bar_color};"></div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detailed scores
                score_cols = st.columns(5)
                score_data = [
                    ("Semantic", result["semantic_score"], "🎯"),
                    ("Skills", result["skill_score"], "🧠"),
                    ("Experience", result["experience_score"], "💼"),
                    ("Education", result["education_score"], "🎓"),
                    ("ATS", result["ats_score"], "🤖")
                ]
                
                for col, (label, score, icon) in zip(score_cols, score_data):
                    col.metric(f"{icon} {label}", f"{score}%")
                
                # Skills Section
                st.markdown("#### 🎯 Skills Analysis")
                
                if result["matching_skills"]:
                    st.markdown("**✅ Matching Skills:**")
                    st.markdown(" ".join([f"<span class='skill-badge'>{s}</span>" for s in result["matching_skills"]]), unsafe_allow_html=True)
                
                if result["missing_skills"]:
                    st.markdown("<br>**❌ Missing Skills:**", unsafe_allow_html=True)
                    st.markdown(" ".join([f"<span class='missing-badge'>{s}</span>" for s in result["missing_skills"]]), unsafe_allow_html=True)
                
                if result["extra_skills"]:
                    st.markdown(f"<br>**💡 Bonus Skills:** {', '.join(result['extra_skills'])}", unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                if result["missing_skills"]:
                    st.info(f"**Improvement:** Add experience with {', '.join(result['missing_skills'][:3])} to increase match by ~{len(result['missing_skills'])*10}%")
                
                if result["ats_issues"]:
                    st.warning(f"**ATS Issues:** {', '.join(result['ats_issues'])}")
                
                # Selection Status
                if is_selected:
                    st.success("🎉 **RECOMMENDED FOR INTERVIEW**")
                else:
                    st.error("⚠️ **DOES NOT MEET REQUIREMENTS**")
            
            with col2:
                # Radar Chart
                radar_data = {
                    "Semantic": result["semantic_score"],
                    "Skills": result["skill_score"],
                    "Experience": result["experience_score"],
                    "Education": result["education_score"],
                    "ATS": result["ats_score"]
                }
                
                fig = create_radar_chart(radar_data)
                st.pyplot(fig)
                
                # Word Cloud
                if enable_wordcloud:
                    st.markdown("#### ☁️ Keyword Cloud")
                    wc_fig = generate_wordcloud(result["text"], "Resume Keywords")
                    st.pyplot(wc_fig)
    
    # -------------------- COMPARATIVE ANALYSIS --------------------
    st.markdown("---")
    st.markdown("### 📈 Comparative Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(4, len(results)*0.5)))
        
        names = [r["name"][:15] + "..." if len(r["name"]) > 15 else r["name"] for r in results]
        scores = [r["final_score"] for r in results]
        colors = ["#00ff88" if s >= similarity_threshold else "#ff416c" for s in scores]
        
        bars = ax.barh(names, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 1, i, f"{score:.1f}%", va='center', color='white', fontweight='bold')
        
        ax.set_xlabel("Match Score (%)", color='white', fontsize=12)
        ax.set_title("Candidate Ranking", color='white', fontsize=14, pad=20)
        ax.set_facecolor('none')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.patch.set_alpha(0)
        
        # Add threshold line
        ax.axvline(x=similarity_threshold, color='#ffc107', linestyle='--', linewidth=2, label=f'Threshold ({similarity_threshold}%)')
        ax.legend(loc='lower right', facecolor='none', edgecolor='white', labelcolor='white')
        
        st.pyplot(fig)
    
    with col2:
        # Skills distribution
        all_skills = {}
        for r in results:
            for skill in r["skills_found"]:
                all_skills[skill] = all_skills.get(skill, 0) + 1
        
        if all_skills:
            top_skills = dict(sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            skills = list(top_skills.keys())
            counts = list(top_skills.values())
            
            bars = ax.bar(skills, counts, color='cyan', alpha=0.7, edgecolor='white')
            ax.set_ylabel("Frequency", color='white')
            ax.set_title("Most Common Skills Across Candidates", color='white', pad=20)
            ax.set_facecolor('none')
            ax.tick_params(colors='white', rotation=45)
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.patch.set_alpha(0)
            
            st.pyplot(fig)
    
    # -------------------- EXPORT --------------------
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    if st.button("📥 Generate Report"):
        # Create export dataframe
        export_df = pd.DataFrame([{
            "Candidate": r["name"],
            "Final Score": r["final_score"],
            "Semantic Match": r["semantic_score"],
            "Skill Match": r["skill_score"],
            "Experience Score": r["experience_score"],
            "Education Score": r["education_score"],
            "ATS Score": r["ats_score"],
            "Skills Found": ", ".join(r["skills_found"]),
            "Missing Skills": ", ".join(r["missing_skills"]),
            "Recommendation": "Selected" if r["final_score"] >= similarity_threshold else "Rejected"
        } for r in results])
        
        if export_format == "Excel":
            buffer = BytesIO()
            export_df.to_excel(buffer, index=False, engine='openpyxl')
            buffer.seek(0)
            st.download_button(
                "Download Excel Report",
                buffer,
                "resume_analysis_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "CSV":
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download CSV Report",
                csv,
                "resume_analysis_report.csv",
                "text/csv"
            )
        else:
            st.info("PDF generation requires additional libraries (reportlab)")

else:
    # Empty state
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; opacity: 0.6;">
            <h2 style="color: #667eea; font-size: 4rem;">📄</h2>
            <h3 style="color: #a0a0a0;">Ready to Analyze</h3>
            <p style="color: #666;">Upload resumes and enter a job description to get started</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    features = [
        ("🎯 Multi-Algorithm Scoring", "TF-IDF + Semantic + Jaccard similarity"),
        ("🧠 Smart Skill Extraction", "Context-aware skill detection with synonyms"),
        ("📊 Visual Analytics", "Radar charts, word clouds, and comparative graphs"),
        ("🤖 ATS Compatibility", "Automated ATS formatting checks"),
        ("⚙️ Customizable Thresholds", "Adjustable scoring weights and criteria"),
        ("💾 Export Reports", "Excel, CSV, and PDF report generation")
    ]
    
    feat_cols = st.columns(3)
    for i, (icon_title, desc) in enumerate(features):
        with feat_cols[i % 3]:
            st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 15px; margin: 0.5rem 0; border: 1px solid rgba(102, 126, 234, 0.3);">
                    <h4 style="color: #00d4ff; margin-bottom: 0.5rem;">{icon_title}</h4>
                    <p style="color: #a0a0a0; font-size: 0.9rem;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)
