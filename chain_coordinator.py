# chain_coordinator.py
import streamlit as st
from agents import ResumeAnalysisAgents, AgentResponse
import time

class ChainCoordinator:
    """Coordinates the chain of agents"""
    
    def __init__(self, api_key: str = None):
        try:
            self.agents = ResumeAnalysisAgents(api_key=api_key)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.stop()
        
        self.chain_history = []
    
    def run_analysis_chain(self, job_description: str, resume_text: str, resume_name: str = "Resume"):
        """Run the complete analysis chain"""
        
        results = {
            "resume_name": resume_name,
            "analyses": []
        }
        
        try:
            # STEP 1: Analyze Job
            st.write("📋 **Step 1: Analyzing Job Requirements...**")
            job_analysis = self.agents.analyze_job_requirements(job_description)
            results["analyses"].append(job_analysis)
            st.write("✅ Job requirements analyzed")
            time.sleep(0.5)
            
            # STEP 2: Analyze Resume
            st.write("📄 **Step 2: Analyzing Resume...**")
            resume_analysis = self.agents.analyze_resume(resume_text)
            results["analyses"].append(resume_analysis)
            st.write("✅ Resume analyzed")
            time.sleep(0.5)
            
            # STEP 3: Match Skills
            st.write("🧠 **Step 3: Matching Skills...**")
            skill_matching = self.agents.find_skill_matches(
                job_analysis.response,
                resume_analysis.response
            )
            results["analyses"].append(skill_matching)
            st.write("✅ Skills matched")
            time.sleep(0.5)
            
            # STEP 4: Score Experience
            st.write("💼 **Step 4: Scoring Experience...**")
            exp_scoring = self.agents.score_experience(
                job_analysis.response,
                resume_analysis.response
            )
            results["analyses"].append(exp_scoring)
            st.write("✅ Experience scored")
            time.sleep(0.5)
            
            # STEP 5: Calculate Overall Score
            st.write("🏆 **Step 5: Calculating Overall Score...**")
            overall_score = self.agents.calculate_overall_score(results["analyses"])
            results["analyses"].append(overall_score)
            st.write("✅ Overall score calculated")
            time.sleep(0.5)
            
            # STEP 6: Generate Recommendations
            st.write("💡 **Step 6: Generating Recommendations...**")
            recommendations = self.agents.generate_recommendations(results["analyses"])
            results["analyses"].append(recommendations)
            st.write("✅ Recommendations generated")
            
            results["success"] = True
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def run_batch_analysis(self, job_description: str, resume_texts: list) -> list:
        """Run analysis for multiple resumes"""
        all_results = []
        
        for idx, (resume_text, resume_name) in enumerate(resume_texts):
            st.write(f"\n{'='*60}")
            st.write(f"📊 Processing Resume {idx + 1}/{len(resume_texts)}")
            st.write(f"{'='*60}\n")
            
            result = self.run_analysis_chain(job_description, resume_text, resume_name)
            all_results.append(result)
        
        return all_results
