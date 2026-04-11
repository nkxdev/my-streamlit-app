# agents.py
import anthropic

class SimpleAgent:
    """A single AI agent that answers questions"""
    
    def __init__(self):
        # Connect to Claude AI
        self.client = anthropic.Anthropic(api_key="your-key-here")
    
    def ask_question(self, question, previous_info=""):
        """
        Ask AI a question
        
        previous_info = Information from previous agents
        """
        
        # Build the message
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": f"""
{previous_info}

Now answer this question:
{question}
"""
                }
            ]
        )
        
        # Get the answer
        answer = message.content[0].text
        return answer

# ============= CREATE SPECIFIC AGENTS =============

def job_analyzer(job_description):
    """Agent 1: Analyze what the job needs"""
    agent = SimpleAgent()
    
    question = f"""
    Read this job description and tell me:
    1. What skills are MUST HAVE?
    2. What skills are NICE TO HAVE?
    3. How many years experience needed?
    4. What education level?
    
    Job Description:
    {job_description}
    """
    
    answer = agent.ask_question(question)
    return answer


def resume_analyzer(resume_text):
    """Agent 2: Analyze what resume has"""
    agent = SimpleAgent()
    
    question = f"""
    Read this resume and tell me:
    1. What skills does this person have?
    2. How many years of experience?
    3. What education do they have?
    4. What are their strengths?
    
    Resume:
    {resume_text}
    """
    
    answer = agent.ask_question(question)
    return answer


def skill_matcher(job_skills, resume_skills, job_info, resume_info):
    """Agent 3: Compare skills"""
    agent = SimpleAgent()
    
    previous = f"""
    What the job needs:
    {job_info}
    
    What the candidate has:
    {resume_info}
    """
    
    question = f"""
    Compare these two:
    
    Job Skills Needed: {job_skills}
    Resume Skills: {resume_skills}
    
    Tell me:
    1. Which skills match perfectly?
    2. Which skills are missing?
    3. Which extra skills does resume have?
    4. What's the match percentage? (0-100)
    """
    
    answer = agent.ask_question(question, previous)
    return answer


def experience_scorer(job_requirements, resume_experience, previous_context):
    """Agent 4: Score experience"""
    agent = SimpleAgent()
    
    question = f"""
    {previous_context}
    
    Now evaluate experience:
    
    Job Requirements: {job_requirements}
    Resume Experience: {resume_experience}
    
    Score out of 100:
    1. Years of experience match?
    2. Similar job experience?
    3. Level of responsibility match?
    
    Give final experience score (0-100)
    """
    
    answer = agent.ask_question(question, previous_context)
    return answer


def final_scorer(all_info):
    """Agent 5: Calculate final score"""
    agent = SimpleAgent()
    
    question = f"""
    Based on ALL this information:
    {all_info}
    
    Give me:
    1. Final Score (0-100)
    2. Should we hire? (Yes/Maybe/No)
    3. Why?
    """
    
    answer = agent.ask_question(question, all_info)
    return answer


def recommendation_generator(candidate_info, final_score, previous_info):
    """Agent 6: Give recommendations"""
    agent = SimpleAgent()
    
    question = f"""
    Based on analysis score of {final_score}:
    {previous_info}
    
    Give me:
    1. Top 3 strengths
    2. Top 3 weaknesses
    3. What should they learn?
    4. Questions to ask in interview?
    5. Is this a good hire?
    """
    
    answer = agent.ask_question(question, previous_info)
    return answer
