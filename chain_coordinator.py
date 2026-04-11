# chain_coordinator.py
import json
from typing import List, Dict, Any
from agents import ResumeAnalysisAgents, AgentResponse
import streamlit as st

class ChainCoordinator:
    """Coordinates the chain of agents"""
    
    def __init__(self):
        self.agents = ResumeAnalysisAgents()
        self.chain_history = []
    
    def parse_json_response(self, response_text: str) -> dict:
        """Safely parse JSON from agent response"""
        try:
            # Try to find JSON block in response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "{" in response_text:
                json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
            else:
                return {"raw": response_text}
            
            return json.loads(json_str)
        except:
            return {"raw": response_text}
    
    def run_analysis_chain(self, job_description: str, resume_texts: List[str]) -> List[Dict]:
        """Run complete analysis chain for all resumes"""
        
        results = []
        
        # STEP 1: Analyze Job Requirements (Once for all candidates)
        st.write("🔍 **Step 1:** Analyzing Job Requirements...")
        job_analysis = self.agents.analyze_job_requirements(job_description)
        self.chain_history.append(job_analysis)
        job_context = job_analysis.response
        st.success("✅ Job requirements analyzed")
        
        # STEP 2-6: Process each resume through the chain
        for idx, resume_text in enumerate(resume_texts):
            st.write(f"\n📄 **Processing Resume {idx + 1}**")
            
            # STEP 2: Analyze Resume
            st.write("   ├─ Analyzing resume content...")
            resume_analysis = self.agents.analyze_resume(
                resume_text, 
                context=job_context
            )
            self.chain_history.append(resume_analysis)
            
            # STEP 3: Find Skill Matches
            st.write("   ├─ Matching skills...")
            skill_matching = self.agents.find_skill_matches(
                job_context,
                resume_analysis.response,
                context=f"{job_analysis.response}\n{resume_analysis.response}"
            )
            self.chain_history.append(skill_matching)
            
            # STEP 4: Score Experience
            st.write("   ├─ Evaluating experience...")
            exp_scoring = self.agents.score_experience(
                job_context,
                resume_analysis.response,
                context=f"{skill_matching.response}"
            )
            self.chain_history.append(exp_scoring)
            
            # STEP 5: Calculate Overall Score
            st.write("   ├─ Computing overall match score...")
            all_analyses = [job_analysis, resume_analysis, skill_matching, exp_scoring]
            overall_score = self.agents.calculate_overall_score(
                all_analyses,
                job_description,
                context="Combining all analyses..."
            )
            self.chain_history.append(overall_score)
            
            # STEP 6: Generate Recommendations
            st.write("   └─ Generating recommendations...")
            candidate_name = resume_text.split('\n')[0][:30]  # Extract name from resume
            recommendations = self.agents.generate_recommendations(
                candidate_name,
                float(self._extract_score(overall_score.response)),
                all_analyses,
                context=overall_score.response
            )
            self.chain_history.append(recommendations)
            
            # Compile result
            result = {
                "candidate_idx": idx + 1,
                "resume_text": resume_text,
                "job_analysis": self.parse_json_response(job_analysis.response),
                "resume_analysis": self.parse_json_response(resume_analysis.response),
                "skill_matching": self.parse_json_response(skill_matching.response),
                "experience_scoring": self.parse_json_response(exp_scoring.response),
                "overall_score": self.parse_json_response(overall_score.response),
                "recommendations": self.parse_json_response(recommendations.response),
            }
            results.append(result)
            st.success(f"✅ Resume {idx + 1} completed")
        
        return results
    
    def _extract_score(self, response: str) -> float:
        """Extract numeric score from response"""
        import re
        match = re.search(r'\d+(?:\.\d+)?', response)
        return float(match.group()) if match else 50.0
