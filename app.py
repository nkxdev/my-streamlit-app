# app.py (Updated sections)
import os

import streamlit as st
import pandas as pd
from chain_coordinator import ChainCoordinator
from utils import extract_pdf, extract_docx
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="🤖 AI Resume Analyzer - Agentic Edition", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS remains the same as before...
st.markdown("""
    <style>
    # [Keep your existing CSS here]
    </style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Agent Settings")

    use_agentic = st.toggle("🤖 Enable Agentic Analysis", value=True)

    api_key = ""
    if use_agentic:
        api_key = st.text_input("Enter Claude API Key (optional)", type="password")
        real_key_present = bool(
            api_key and api_key.strip() and api_key != "your_api_key_here"
        ) or bool(
            os.getenv("ANTHROPIC_API_KEY", "").strip()
            and os.getenv("ANTHROPIC_API_KEY") != "your_api_key_here"
        )
        if real_key_present:
            st.success("🔑 Real Claude API will be used")
        else:
            st.info("🤖 Mock AI (offline mode) — no API key needed")
    
    st.markdown("---")
    st.markdown("### Analysis Mode")
    analysis_mode = st.selectbox(
        "Select Analysis Type",
        ["Quick Analysis", "Deep Analysis", "Detailed Report"]
    )

# ==================== HEADER ====================
st.markdown("<h1>🤖 AI Resume Analyzer Pro - Agentic Edition</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Next-Generation Resume Screening with Chain Prompting</p>", 
            unsafe_allow_html=True)

# ==================== INPUT SECTION ====================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop resumes here", 
        accept_multiple_files=True,
        type=['pdf', 'docx', 'doc', 'txt']
    )

with col2:
    st.markdown("### 📝 Job Description")
    job_desc = st.text_area(
        "Paste job requirements",
        height=200,
        placeholder="Enter detailed job description..."
    )

# ==================== ANALYSIS LOGIC ====================
if uploaded_files and job_desc:
    if st.button("🚀 Start Agentic Analysis"):
        
        # Extract texts
        resume_texts = []
        for file in uploaded_files:
            try:
                if file.type == "application/pdf":
                    text = extract_pdf(file)
                elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                                  "application/msword"]:
                    text = extract_docx(file)
                else:
                    text = file.read().decode('utf-8')
                resume_texts.append(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
        
        if resume_texts:
            if use_agentic:
                # ==================== RUN AGENTIC CHAIN ====================
                st.markdown("---")
                st.markdown("## 🔄 Running Agentic Analysis Chain...")
                
                coordinator = ChainCoordinator(api_key=api_key or None)
                results = coordinator.run_analysis_chain(job_desc, resume_texts)
                
                # ==================== DISPLAY RESULTS ====================
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Create summary table
                summary_data = []
                for result in results:
                    try:
                        score_data = result["overall_score"]
                        score = score_data.get("Final Score", score_data.get("raw", "N/A"))
                        recommendation = score_data.get("Hiring Recommendation", "Pending")
                        
                        summary_data.append({
                            "Candidate #": result["candidate_idx"],
                            "Score": str(score),
                            "Recommendation": recommendation,
                            "Skills Match": result["skill_matching"].get("Overall Skill Match Percentage", "N/A"),
                            "Experience Match": result["experience_scoring"].get("Overall Score", "N/A")
                        })
                    except:
                        summary_data.append({
                            "Candidate #": result["candidate_idx"],
                            "Score": "Error",
                            "Recommendation": "Review",
                            "Skills Match": "N/A",
                            "Experience Match": "N/A"
                        })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # ==================== DETAILED ANALYSIS TABS ====================
                tabs = st.tabs([f"Candidate #{i+1}" for i in range(len(results))])
                
                for tab, result in zip(tabs, results):
                    with tab:
                        # Job Analysis
                        with st.expander("🎯 Job Requirements Analysis", expanded=False):
                            st.json(result["job_analysis"])
                        
                        # Resume Analysis
                        with st.expander("📄 Resume Analysis", expanded=True):
                            st.json(result["resume_analysis"])
                        
                        # Skill Matching
                        with st.expander("🧠 Skill Matching Results", expanded=True):
                            st.json(result["skill_matching"])
                        
                        # Experience Scoring
                        with st.expander("💼 Experience Evaluation", expanded=False):
                            st.json(result["experience_scoring"])
                        
                        # Overall Score
                        with st.expander("🏆 Overall Match Score", expanded=True):
                            score_info = result["overall_score"]
                            if isinstance(score_info, dict):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Score", score_info.get("Final Score", "N/A"))
                                col2.metric("Confidence", score_info.get("Confidence Level", "N/A"))
                                col3.metric("Recommendation", score_info.get("Hiring Recommendation", "N/A"))
                            st.json(score_info)
                        
                        # Recommendations
                        with st.expander("💡 Personalized Recommendations", expanded=True):
                            rec_info = result["recommendations"]
                            if isinstance(rec_info, dict):
                                st.markdown("**Hiring Recommendation:** " + 
                                          rec_info.get("Hiring Recommendation", "Pending"))
                                
                                if "Strengths" in rec_info or "Top 3 Strengths" in rec_info:
                                    strengths = rec_info.get("Top 3 Strengths", rec_info.get("Strengths", []))
                                    st.markdown("**Strengths:**")
                                    for strength in strengths:
                                        st.write(f"✅ {strength}")
                                
                                if "Areas to Improve" in rec_info or "Top 3 Areas to Improve" in rec_info:
                                    improvements = rec_info.get("Top 3 Areas to Improve", rec_info.get("Areas to Improve", []))
                                    st.markdown("**Areas to Improve:**")
                                    for improvement in improvements:
                                        st.write(f"��� {improvement}")
                                
                                if "Interview Questions" in rec_info or "Suggested Interview Questions" in rec_info:
                                    questions = rec_info.get("Suggested Interview Questions", rec_info.get("Interview Questions", []))
                                    st.markdown("**Suggested Interview Questions:**")
                                    for q in questions:
                                        st.write(f"❓ {q}")
                            
                            st.json(rec_info)
                
                # ==================== EXPORT ====================
                st.markdown("---")
                st.markdown("## 💾 Export Results")
                
                if st.button("📥 Generate Comprehensive Report"):
                    export_data = []
                    for result in results:
                        export_data.append({
                            "Candidate": f"Resume {result['candidate_idx']}",
                            "Overall Score": str(result["overall_score"].get("Final Score", "N/A")),
                            "Job Analysis": json.dumps(result["job_analysis"]),
                            "Resume Analysis": json.dumps(result["resume_analysis"]),
                            "Skill Matching": json.dumps(result["skill_matching"]),
                            "Experience Scoring": json.dumps(result["experience_scoring"]),
                            "Recommendations": json.dumps(result["recommendations"])
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Report (CSV)",
                        data=csv,
                        file_name="agentic_analysis_report.csv",
                        mime="text/csv"
                    )
            else:
                # Traditional analysis (keep existing code)
                st.info("Using traditional analysis mode")

else:
    st.info("👆 Upload resumes and enter job description to begin")
