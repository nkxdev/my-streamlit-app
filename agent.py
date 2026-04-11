# agents.py
import anthropic
from typing import Optional
from dataclasses import dataclass

@dataclass
class AgentResponse:
    agent_name: str
    response: str
    data: dict = None

class ResumeAnalysisAgents:
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def _call_agent(self, agent_name: str, prompt: str, context: str = "") -> AgentResponse:
        """Generic agent call with chain prompting"""
        full_prompt = f"""
You are a {agent_name} specialized in resume analysis.
Previous Context: {context if context else 'None'}

{prompt}

Respond in a clear, structured JSON format when possible.
"""
        
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        )
        
        return AgentResponse(
            agent_name=agent_name,
            response=message.content[0].text
        )
    
    # ==================== AGENT 1: ANALYSIS AGENT ====================
    def analyze_job_requirements(self, job_description: str) -> AgentResponse:
        """Extract and analyze job requirements"""
        prompt = f"""
Analyze this job description and extract:
1. Key Skills Required (ranked by importance)
2. Experience Level Required
3. Education Requirements
4. Nice-to-Have Skills
5. Seniority Level
6. Key Responsibilities

Job Description:
{job_description}

Provide structured output with clear categorization.
"""
        return self._call_agent("Job Requirements Analyzer", prompt)
    
    # ==================== AGENT 2: RESUME ANALYZER ====================
    def analyze_resume(self, resume_text: str, context: str = "") -> AgentResponse:
        """Deep analysis of resume content"""
        prompt = f"""
Thoroughly analyze this resume and extract:
1. Key Skills (categorized by type: Programming, Tools, Soft Skills)
2. Experience (years, roles, achievements)
3. Education (degree, institution, graduation year)
4. Certifications and Achievements
5. Strengths
6. Potential Gaps

Resume Text:
{resume_text}

Be thorough and precise.
"""
        return self._call_agent("Resume Analyzer", prompt, context)
    
    # ==================== AGENT 3: MATCHING AGENT ====================
    def find_skill_matches(self, job_skills: str, resume_skills: str, context: str = "") -> AgentResponse:
        """Find and score skill matches"""
        prompt = f"""
Compare these two skill sets and provide:
1. Exact Matches (skill is directly present)
2. Related Matches (similar technologies/concepts)
3. Missing Skills (critical ones needed)
4. Bonus Skills (candidate has but not required)
5. Match Score (0-100) for each skill
6. Overall Skill Match Percentage

Required Skills from Job:
{job_skills}

Resume Skills:
{resume_skills}

Provide reasoning for each match/mismatch.
"""
        return self._call_agent("Skill Matching Agent", prompt, context)
    
    # ==================== AGENT 4: EXPERIENCE SCORER ====================
    def score_experience(self, job_requirements: str, resume_exp: str, context: str = "") -> AgentResponse:
        """Score experience alignment"""
        prompt = f"""
Score the candidate's experience against requirements:
1. Years of Experience Match
2. Relevant Experience in Similar Roles
3. Industry Experience Alignment
4. Project Complexity Match
5. Leadership/Seniority Match
6. Technical Depth Assessment

Job Requirements:
{job_requirements}

Candidate Experience:
{resume_exp}

Provide scores out of 100 for each category with justification.
"""
        return self._call_agent("Experience Scoring Agent", prompt, context)
    
    # ==================== AGENT 5: HOLISTIC SCORER ====================
    def calculate_overall_score(self, all_analyses: list, job_desc: str, context: str = "") -> AgentResponse:
        """Calculate comprehensive match score"""
        prompt = f"""
Based on all the previous analyses provided, calculate:
1. Skill Match Weight: 35%
2. Experience Match Weight: 30%
3. Education Match Weight: 15%
4. Certification/Achievements Weight: 10%
5. Fit & Culture Potential Weight: 10%

Previous Analyses:
{chr(10).join([f"- {a.agent_name}: {a.response[:200]}" for a in all_analyses])}

Job Description Context:
{job_desc}

Provide:
- Final Score (0-100)
- Confidence Level
- Hiring Recommendation (Strong Yes/Yes/Maybe/No)
- Risk Assessment
"""
        return self._call_agent("Holistic Scoring Agent", prompt, context)
    
    # ==================== AGENT 6: RECOMMENDATION AGENT ====================
    def generate_recommendations(self, candidate_name: str, final_score: float, 
                                all_analyses: list, context: str = "") -> AgentResponse:
        """Generate personalized recommendations"""
        prompt = f"""
Generate comprehensive recommendations for:
Candidate: {candidate_name}
Final Score: {final_score}/100

Based on Analysis Summary:
{chr(10).join([f"- {a.agent_name}: {a.response[:150]}" for a in all_analyses])}

Provide:
1. Hiring Recommendation (Accept/Further Discussion/Reject)
2. Top 3 Strengths
3. Top 3 Areas to Improve
4. Skills to Develop Next 6 Months
5. Suggested Interview Questions
6. Potential in Company (Long-term growth)
7. Department/Role Best Fit
"""
        return self._call_agent("Recommendation Agent", prompt, context)
