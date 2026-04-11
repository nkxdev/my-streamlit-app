# chain_coordinator.py
from agents import (
    job_analyzer,
    resume_analyzer,
    skill_matcher,
    experience_scorer,
    final_scorer,
    recommendation_generator
)

class ChainRunner:
    """Runs all agents in sequence (like a manager)"""
    
    def run_full_chain(self, job_description, resume_text):
        """
        Run all agents one after another
        Each one receives info from previous ones
        """
        
        print("🔄 Starting Chain Analysis...\n")
        
        # ========== STEP 1 ==========
        print("Step 1: Analyzing Job Requirements...")
        job_info = job_analyzer(job_description)
        print(f"✅ Done!\n")
        print(job_info)
        print("\n" + "="*50 + "\n")
        
        # ========== STEP 2 ==========
        print("Step 2: Analyzing Resume...")
        resume_info = resume_analyzer(resume_text)
        print(f"✅ Done!\n")
        print(resume_info)
        print("\n" + "="*50 + "\n")
        
        # ========== STEP 3 ==========
        print("Step 3: Matching Skills (uses info from Step 1 & 2)...")
        skill_info = skill_matcher(
            job_description,
            resume_text,
            job_info,           # ← From step 1
            resume_info         # ← From step 2
        )
        print(f"✅ Done!\n")
        print(skill_info)
        print("\n" + "="*50 + "\n")
        
        # ========== STEP 4 ==========
        print("Step 4: Scoring Experience (uses info from all above)...")
        experience_info = experience_scorer(
            job_info,           # ← From step 1
            resume_info,        # ← From step 2
            skill_info          # ← From step 3
        )
        print(f"✅ Done!\n")
        print(experience_info)
        print("\n" + "="*50 + "\n")
        
        # ========== STEP 5 ==========
        print("Step 5: Calculating Final Score (combines everything)...")
        combined_info = f"""
        Job Analysis: {job_info}
        Resume Analysis: {resume_info}
        Skills Match: {skill_info}
        Experience Score: {experience_info}
        """
        
        final_info = final_scorer(combined_info)
        print(f"✅ Done!\n")
        print(final_info)
        print("\n" + "="*50 + "\n")
        
        # ========== STEP 6 ==========
        print("Step 6: Generating Recommendations (final step)...")
        recommendations = recommendation_generator(
            resume_text,
            final_info,         # ← From step 5
            combined_info       # ← All previous info
        )
        print(f"✅ Done!\n")
        print(recommendations)
        
        # Return everything
        return {
            "step1_job_analysis": job_info,
            "step2_resume_analysis": resume_info,
            "step3_skill_matching": skill_info,
            "step4_experience_scoring": experience_info,
            "step5_final_score": final_info,
            "step6_recommendations": recommendations
        }


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    chain = ChainRunner()
    
    job_desc = """
    We need a Python Developer with 3+ years experience
    Skills needed: Python, Django, PostgreSQL, Docker
    Education: Bachelor's in CS
    """
    
    resume = """
    Name: John Doe
    Skills: Python, Flask, MySQL, Git
    Experience: 5 years as Python developer
    Education: BS in Computer Science
    """
    
    results = chain.run_full_chain(job_desc, resume)
