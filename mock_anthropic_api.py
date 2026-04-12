"""
Mock Anthropic API Server for the Agentic Resume Checker
=========================================================
Provides a /v1/messages endpoint that mimics the Anthropic Claude API,
enabling the resume analysis pipeline to run fully offline without a
real API key.

Run as a standalone server
---------------------------
    python mock_anthropic_api.py
    # or
    uvicorn mock_anthropic_api:app --host 0.0.0.0 --port 8000

Use with the official anthropic Python client
----------------------------------------------
    import anthropic
    client = anthropic.Anthropic(
        api_key="any-non-empty-string",
        base_url="http://localhost:8000",
    )
"""

import json
import os
import re
import sys
import uuid
from typing import List

from fastapi import FastAPI, HTTPException, Request

# Ensure the module's directory is on the path so skills.py is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skills import SKILLS

app = FastAPI(
    title="Mock Anthropic API",
    description=(
        "A local mock of the Anthropic Claude API tailored for the "
        "Agentic Resume Checker project."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

def extract_skills_from_text(text: str) -> List[str]:
    """Return known skills found in *text* (preserving discovery order)."""
    text_lower = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z\-])" + re.escape(skill.lower()) + r"(?![a-z\-])"
        if re.search(pattern, text_lower):
            found.append(skill)
    # Deduplicate while preserving order
    return list(dict.fromkeys(found))


def extract_years_of_experience(text: str) -> float:
    """Return the maximum years-of-experience figure found in *text*."""
    text_lower = text.lower()
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?\s+(?:of\s+)?experience",
        r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"over\s+(\d+)\s+years?",
    ]
    years: List[float] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text_lower):
            try:
                years.append(float(m.group(1)))
            except (ValueError, IndexError):
                pass
    return max(years) if years else 0.0


def extract_education_level(text: str) -> str:
    """Return the highest education level found in *text*."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["ph.d", "phd", "doctorate"]):
        return "PhD"
    if any(kw in text_lower for kw in ["master's", "masters", "m.s.", "m.tech", "mba", "m.sc", "msc", "m.e."]):
        return "Master's Degree"
    if any(kw in text_lower for kw in ["bachelor's", "bachelors", "b.s.", "b.tech", "b.sc", "bsc", "b.e.", "b.a.", "undergraduate"]):
        return "Bachelor's Degree"
    if any(kw in text_lower for kw in ["associate", "diploma", "polytechnic"]):
        return "Associate/Diploma"
    return "Not Specified"


def determine_seniority(text: str, required_years: float) -> str:
    """Infer seniority level from keywords and required years of experience."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["senior", "lead", "principal", "staff", "head of"]):
        return "Senior"
    if any(kw in text_lower for kw in ["junior", "entry", "fresher", "graduate", "trainee", "associate"]):
        return "Junior"
    if required_years >= 5:
        return "Senior"
    if required_years <= 1:
        return "Junior"
    return "Mid-level"


# ---------------------------------------------------------------------------
# Skill category sets (used for categorisation in handle_resume_analyzer)
# ---------------------------------------------------------------------------

_SKILL_PROG = frozenset({
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "rust", "kotlin", "swift", "php", "ruby", "scala", "r", "matlab",
})
_SKILL_WEB = frozenset({
    "html", "css", "react", "angular", "vue", "next.js", "node",
    "express", "django", "flask", "spring", "spring boot", "sass",
    "tailwind", "bootstrap",
})
_SKILL_DATA = frozenset({
    "machine learning", "deep learning", "data science", "nlp",
    "tensorflow", "pytorch", "keras", "pandas", "numpy",
    "scikit-learn", "xgboost",
})
_SKILL_CLOUD = frozenset({
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd",
})
_SKILL_DB = frozenset({
    "sql", "mysql", "postgresql", "mongodb", "firebase",
    "redis", "oracle", "sqlite",
})


# ---------------------------------------------------------------------------
# Agent type detection
# ---------------------------------------------------------------------------

_AGENT_MARKERS = {
    "job_analyzer": "You are a Job Requirements Analyzer",
    "resume_analyzer": "You are a Resume Analyzer",
    "skill_matcher": "You are a Skill Matching Agent",
    "experience_scorer": "You are an Experience Scoring Agent",
    "final_scorer": "You are a Final Scoring Agent",
    "recommendation": "You are a Recommendation Agent",
}


def detect_agent_type(prompt: str) -> str:
    """Identify which resume-analysis agent sent *prompt*."""
    for agent_type, marker in _AGENT_MARKERS.items():
        if marker in prompt:
            return agent_type
    return "unknown"


# ---------------------------------------------------------------------------
# Per-agent response handlers
# ---------------------------------------------------------------------------

def handle_job_analyzer(prompt: str) -> dict:
    """Extract structured job requirements from the prompt."""
    jd_match = re.search(r"Job Description:\s*\n([\s\S]+?)(?:\n\nProvide|$)", prompt)
    jd_text = jd_match.group(1).strip() if jd_match else prompt

    skills = extract_skills_from_text(jd_text)
    required_years = extract_years_of_experience(jd_text)
    education = extract_education_level(jd_text)
    seniority = determine_seniority(jd_text, required_years)

    # Nice-to-have: skills mentioned after "preferred / bonus / optional" keywords
    nice_match = re.search(
        r"(?:preferred|nice[\s\-]to[\s\-]have|bonus|plus|optional):?([\s\S]{0,400})",
        jd_text.lower(),
    )
    nice_skills: List[str] = []
    if nice_match:
        nice_skills = [s for s in extract_skills_from_text(nice_match.group(1)) if s not in skills]

    return {
        "Key Skills Required": skills[:15] or ["Communication", "Problem Solving", "Teamwork"],
        "Experience Level Required": f"{int(required_years)}+ years" if required_years else "Not specified",
        "Education Requirements": education,
        "Nice-to-Have Skills": nice_skills[:5],
        "Seniority Level": seniority,
    }


def handle_resume_analyzer(prompt: str) -> dict:
    """Extract structured candidate information from the prompt."""
    resume_match = re.search(r"Resume:\s*\n([\s\S]+?)(?:\n\nProvide|$)", prompt)
    resume_text = resume_match.group(1).strip() if resume_match else prompt

    skills = extract_skills_from_text(resume_text)
    years_exp = extract_years_of_experience(resume_text)
    education = extract_education_level(resume_text)

    categorized = {
        "Programming Languages": [s for s in skills if s in _SKILL_PROG],
        "Web Development": [s for s in skills if s in _SKILL_WEB],
        "Data/AI/ML": [s for s in skills if s in _SKILL_DATA],
        "Cloud/DevOps": [s for s in skills if s in _SKILL_CLOUD],
        "Databases": [s for s in skills if s in _SKILL_DB],
    }
    known = {s for cat in categorized.values() for s in cat}
    categorized["Other"] = [s for s in skills if s not in known][:10]

    return {
        "Key Skills": categorized,
        "Years of Experience": years_exp if years_exp > 0 else "Not clearly stated",
        "Education Level": education,
        "Certifications": [],
        "Work Experience": ["Extracted from resume text"],
        "Key Achievements": ["See resume for details"],
    }


def handle_skill_matcher(prompt: str) -> dict:
    """Compare job requirements with resume skills."""
    job_section = re.search(r"JOB REQUIREMENTS:\s*\n([\s\S]+?)(?:RESUME ANALYSIS:|$)", prompt)
    resume_section = re.search(r"RESUME ANALYSIS:\s*\n([\s\S]+?)(?:Provide:|$)", prompt)

    job_text = job_section.group(1) if job_section else ""
    resume_text = resume_section.group(1) if resume_section else ""

    job_skills = set(extract_skills_from_text(job_text))
    resume_skills = set(extract_skills_from_text(resume_text))

    exact_matches = sorted(job_skills & resume_skills)
    missing = sorted(job_skills - resume_skills)
    bonus = sorted(resume_skills - job_skills)

    match_pct = round(len(exact_matches) / len(job_skills) * 100) if job_skills else 50

    return {
        "Exact Skill Matches": exact_matches,
        "Missing Skills": missing,
        "Bonus Skills": bonus[:10],
        "Match Percentage": match_pct,
        "Overall Skill Match Percentage": f"{match_pct}%",
        "Total Job Skills": len(job_skills),
        "Total Resume Skills": len(resume_skills),
        "Matched Count": len(exact_matches),
    }


def handle_experience_scorer(prompt: str) -> dict:
    """Score candidate experience against job requirements."""
    job_section = re.search(r"JOB REQUIREMENTS:\s*\n([\s\S]+?)(?:CANDIDATE EXPERIENCE:|$)", prompt)
    candidate_section = re.search(r"CANDIDATE EXPERIENCE:\s*\n([\s\S]+?)(?:Score|$)", prompt)

    job_text = job_section.group(1) if job_section else ""
    candidate_text = candidate_section.group(1) if candidate_section else ""

    required_years = extract_years_of_experience(job_text) or 2.0
    actual_years = extract_years_of_experience(candidate_text)

    ratio = actual_years / required_years if required_years > 0 else 0.5
    if ratio >= 1.0:
        years_score = 90
    elif ratio >= 0.75:
        years_score = 75
    elif ratio >= 0.5:
        years_score = 55
    else:
        years_score = 35

    job_skills = extract_skills_from_text(job_text)
    cand_skills = extract_skills_from_text(candidate_text)
    overlap = len(set(job_skills) & set(cand_skills))
    relevant_score = min(90, 40 + overlap * 5) if job_skills else 60

    industry_score = 65
    overall = round(years_score * 0.4 + relevant_score * 0.4 + industry_score * 0.2)

    return {
        "Years of Experience Match": years_score,
        "Relevant Experience": relevant_score,
        "Industry Experience": industry_score,
        "Overall Experience Score": overall,
        "Overall Score": f"{overall}/100",
        "Required Years": f"{int(required_years)} years",
        "Actual Years": f"{actual_years:.1f} years" if actual_years > 0 else "Not stated",
        "Reasoning": (
            f"Candidate has {actual_years:.1f} years vs {int(required_years)} required. "
            f"Skill overlap covers {overlap} of {len(job_skills)} required skills."
        ),
    }


def handle_final_scorer(prompt: str) -> dict:
    """Derive an overall match score from the prior agent outputs."""
    skill_match = re.search(r"Match Percentage[\s\":]+(\d+)", prompt)
    exp_match = re.search(r"Overall(?:\s+Experience)?\s+Score[\s\":]+(\d+)", prompt)

    skill_score = int(skill_match.group(1)) if skill_match else 60
    exp_score = int(exp_match.group(1)) if exp_match else 65

    final = round(skill_score * 0.55 + exp_score * 0.45)

    if final >= 80:
        recommendation, confidence, risk = "Strong Yes", "High", "Low Risk"
    elif final >= 65:
        recommendation, confidence, risk = "Yes", "Medium", "Medium Risk"
    elif final >= 50:
        recommendation, confidence, risk = "Maybe", "Medium", "Medium-High Risk"
    else:
        recommendation, confidence, risk = "No", "High", "High Risk"

    return {
        "Final Score": final,
        "Final Match Score": f"{final}/100",
        "Hiring Recommendation": recommendation,
        "Confidence Level": confidence,
        "Risk Assessment": risk,
        "Skill Match Component": f"{skill_score}/100",
        "Experience Component": f"{exp_score}/100",
    }


def handle_recommendation(prompt: str) -> dict:
    """Generate actionable hiring recommendations."""
    score_match = re.search(r"Final Score[\s\":]+(\d+)", prompt)
    final_score = int(score_match.group(1)) if score_match else 65

    missing_match = re.search(r"Missing Skills[\s\":]+\[([^\]]*)\]", prompt)
    missing_skills: List[str] = []
    if missing_match:
        for s in missing_match.group(1).split(","):
            s = s.strip().strip("\"'")
            if s:
                missing_skills.append(s)

    if final_score >= 75:
        hiring_rec = "Accept"
        strengths = [
            "Strong technical skill alignment with the role",
            "Relevant hands-on experience in required domains",
            "Educational background meets or exceeds requirements",
        ]
        areas_to_improve = [
            "Could deepen expertise in emerging technologies",
            "Benefit from broadening cross-functional collaboration",
            "Strengthen system design and architecture knowledge",
        ]
    elif final_score >= 55:
        hiring_rec = "Discuss"
        strengths = [
            "Solid foundational technical skills",
            "Demonstrated learning agility",
            "Relevant educational background",
        ]
        areas_to_improve = [
            "Gap in some key required technical skills",
            "Limited experience in industry-specific tools",
            "Would benefit from additional senior mentorship",
        ]
    else:
        hiring_rec = "Reject"
        strengths = [
            "Shows enthusiasm and foundational knowledge",
            "Educational qualifications present",
            "Potential for growth with development",
        ]
        areas_to_improve = [
            "Significant skill gaps relative to job requirements",
            "Insufficient years of relevant experience",
            "Needs substantial upskilling before being a strong fit",
        ]

    skills_to_develop = (
        missing_skills[:3] if missing_skills else ["Cloud Architecture", "System Design", "Advanced Algorithms"]
    )
    interview_questions = [
        "Describe a challenging project that required you to learn a new technology quickly. How did you approach it?",
        "How do you handle technical disagreements within a team?",
        "Walk us through how you would design a scalable system for a high-traffic application.",
    ]

    return {
        "Hiring Recommendation": hiring_rec,
        "Top 3 Strengths": strengths,
        "Top 3 Areas to Improve": areas_to_improve,
        "Skills to Develop": skills_to_develop,
        "Suggested Interview Questions": interview_questions,
        "Department Best Fit": "Engineering" if final_score >= 65 else "Junior Engineering / Internship",
    }


# ---------------------------------------------------------------------------
# Central dispatcher
# ---------------------------------------------------------------------------

def analyze_prompt(prompt: str) -> dict:
    """Route *prompt* to the appropriate agent handler and return a dict."""
    handlers = {
        "job_analyzer": handle_job_analyzer,
        "resume_analyzer": handle_resume_analyzer,
        "skill_matcher": handle_skill_matcher,
        "experience_scorer": handle_experience_scorer,
        "final_scorer": handle_final_scorer,
        "recommendation": handle_recommendation,
    }
    agent_type = detect_agent_type(prompt)
    handler = handlers.get(agent_type)
    if handler:
        return handler(prompt)
    return {"message": "Analysis complete", "agent_type": agent_type}


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "Mock Anthropic API",
        "version": "1.0.0",
        "description": "Local mock of the Anthropic Claude API for resume analysis",
        "endpoints": ["GET /health", "POST /v1/messages"],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Mock Anthropic API"}


@app.post("/v1/messages")
async def create_message(request: Request):
    """Mimic Anthropic's POST /v1/messages endpoint."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    model = body.get("model", "claude-3-5-sonnet-20241022")

    # Extract the latest user message
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_prompt = content
            elif isinstance(content, list):
                user_prompt = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            break

    result = analyze_prompt(user_prompt)
    response_text = json.dumps(result, indent=2, ensure_ascii=False)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": response_text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(1, len(user_prompt) // 4),
            "output_tokens": max(1, len(response_text) // 4),
        },
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Mock Anthropic API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    print(f"🚀 Starting Mock Anthropic API on http://{args.host}:{args.port}")
    print(f"   GET  /health")
    print(f"   POST /v1/messages")
    print(
        f"\n💡 Use with the anthropic client:\n"
        f"   anthropic.Anthropic(api_key='any-key', base_url='http://localhost:{args.port}')"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
