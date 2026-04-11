# agents.py
import anthropic
import os
from typing import Optional

class AgentResponse:
    """Stores response from an agent"""
    def __init__(self, agent_name: str, response: str, data: dict = None):
        self.agent_name = agent_name
        self.response = response
        self.data = data or {}

class ResumeAnalysisAgents:
    """AI Agents for resume analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("❌ API Key not found! Set ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def _call_agent(self, agent_name: str, prompt: str) -> AgentResponse:
        """Call Claude AI and get response"""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response_text = message.content[0].text
            return AgentResponse(agent_name=agent_name, response=response_text)
        
        except Exception as e:
            error_msg = f"Error in {agent_name}: {str(e)}"
            return AgentResponse(agent_name=agent_name, response=error_msg)
    
    # ============ AGENT 1: JOB ANALYZER ============
    def analyze_job_requirements(self, job_description: str) -> AgentResponse:
        """Extract job requirements"""
        prompt = f"""
You are a Job Requirements Analyzer.

Analyze this job description and extract:
1. Key Skills Required (list them)
2. Experience Level Required (in years)
3. Education Requirements
4. Nice-to-Have Skills
5. Seniority Level (Junior/Mid/Senior)

Job Description:
{job_description}

Provide clear, structured output.
"""
        return self._call_agent("Job Requirements Analyzer", prompt)
    
    # ============ AGENT 2: RESUME ANALYZER ============
    def analyze_resume(self, resume_text: str) -> AgentResponse:
        """Extract resume information"""
        prompt = f"""
You are a Resume Analyzer.

Analyze this resume and extract:
1. Key Skills (categorized by type)
2. Years of Experience
3. Education Level and Institution
4. Certifications
5. Work Experience (job titles and companies)
6. Key Achievements

Resume:
{resume_text}

Provide clear, structured output.
"""
        return self._call_agent("Resume Analyzer", prompt)
    
    # ============ AGENT 3: SKILL MATCHER ============
    def find_skill_matches(self, job_analysis: str, resume_analysis: str) -> AgentResponse:
        """Match skills between job and resume"""
        prompt = f"""
You are a Skill Matching Agent.

Compare the job requirements with resume skills:

JOB REQUIREMENTS:
{job_analysis}

RESUME ANALYSIS:
{resume_analysis}

Provide:
1. Exact Skill Matches (skills present in both)
2. Missing Skills (required but not in resume)
3. Bonus Skills (in resume but not required)
4. Match Percentage (0-100)

Be specific and list each skill.
"""
        return self._call_agent("Skill Matching Agent", prompt)
    
    # ============ AGENT 4: EXPERIENCE SCORER ============
    def score_experience(self, job_analysis: str, resume_analysis: str) -> AgentResponse:
        """Score experience level"""
        prompt = f"""
You are an Experience Scoring Agent.

JOB REQUIREMENTS:
{job_analysis}

CANDIDATE EXPERIENCE:
{resume_analysis}

Score (0-100) the following:
1. Years of Experience Match
2. Relevant Experience
3. Industry Experience
4. Overall Experience Score

Provide reasoning for each score.
"""
        return self._call_agent("Experience Scoring Agent", prompt)
    
    # ============ AGENT 5: FINAL SCORER ============
    def calculate_overall_score(self, all_analyses: list) -> AgentResponse:
        """Calculate final overall score"""
        analyses_text = "\n\n".join([
            f"{a.agent_name}:\n{a.response}" for a in all_analyses
        ])
        
        prompt = f"""
You are a Final Scoring Agent.

Based on all these analyses:

{analyses_text}

Calculate:
1. Final Match Score (0-100)
2. Hiring Recommendation (Strong Yes / Yes / Maybe / No)
3. Confidence Level (High / Medium / Low)
4. Risk Assessment

Provide a clear overall score.
"""
        return self._call_agent("Final Scoring Agent", prompt)
    
    # ============ AGENT 6: RECOMMENDATION GENERATOR ============
    def generate_recommendations(self, all_analyses: list) -> AgentResponse:
        """Generate recommendations"""
        analyses_text = "\n\n".join([
            f"{a.agent_name}:\n{a.response}" for a in all_analyses
        ])
        
        prompt = f"""
You are a Recommendation Agent.

Based on all these analyses:

{analyses_text}

Provide:
1. Hiring Recommendation (Accept/Discuss/Reject)
2. Top 3 Strengths
3. Top 3 Areas to Improve
4. Skills to Develop
5. Suggested Interview Questions (3 questions)
6. Department Best Fit

Be specific and actionable.
"""
        return self._call_agent("Recommendation Agent", prompt)
