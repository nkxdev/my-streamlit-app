# chain_coordinator.py
import json
import re
import time

import streamlit as st

from agents import ResumeAnalysisAgents, AgentResponse


def _parse_json_response(response_text: str) -> dict:
    """Best-effort JSON extraction from an agent response string.

    Tries (in order):
    1. Direct JSON parse
    2. JSON inside a fenced code block
    3. Returns ``{"raw": response_text}`` as a fallback
    """
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Look for ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response_text)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    return {"raw": response_text}


class ChainCoordinator:
    """Coordinates the six-agent resume-analysis chain."""

    _RESULT_KEYS = (
        "job_analysis", "resume_analysis", "skill_matching",
        "experience_scoring", "overall_score", "recommendations",
    )

    def __init__(self, api_key: str = None):
        self.agents = ResumeAnalysisAgents(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_analysis_chain(self, job_description: str, resume_texts: list) -> list:
        """Analyse every resume in *resume_texts* against *job_description*.

        Returns a list of result dicts, one per resume, each containing:
        ``candidate_idx``, ``job_analysis``, ``resume_analysis``,
        ``skill_matching``, ``experience_scoring``, ``overall_score``,
        ``recommendations``, and ``success``.
        """
        results = []
        for idx, resume_text in enumerate(resume_texts, start=1):
            st.markdown("---")
            st.markdown(f"### 📊 Processing Resume {idx} of {len(resume_texts)}")
            result = self._analyse_single_resume(job_description, resume_text, idx)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_single_resume(
        self, job_description: str, resume_text: str, idx: int
    ) -> dict:
        """Run the full six-step chain for one resume and return a result dict."""
        result: dict = {"candidate_idx": idx, "success": False}
        for key in self._RESULT_KEYS:
            result[key] = {}

        try:
            # Step 1 — Job analysis
            with st.spinner("📋 Analysing job requirements…"):
                job_resp = self.agents.analyze_job_requirements(job_description)
            result["job_analysis"] = _parse_json_response(job_resp.response)
            st.write("✅ Job requirements analysed")
            time.sleep(0.2)

            # Step 2 — Resume analysis
            with st.spinner("📄 Analysing resume…"):
                resume_resp = self.agents.analyze_resume(resume_text)
            result["resume_analysis"] = _parse_json_response(resume_resp.response)
            st.write("✅ Resume analysed")
            time.sleep(0.2)

            # Step 3 — Skill matching
            with st.spinner("🧠 Matching skills…"):
                skill_resp = self.agents.find_skill_matches(
                    job_resp.response, resume_resp.response
                )
            result["skill_matching"] = _parse_json_response(skill_resp.response)
            st.write("✅ Skills matched")
            time.sleep(0.2)

            # Step 4 — Experience scoring
            with st.spinner("💼 Scoring experience…"):
                exp_resp = self.agents.score_experience(
                    job_resp.response, resume_resp.response
                )
            result["experience_scoring"] = _parse_json_response(exp_resp.response)
            st.write("✅ Experience scored")
            time.sleep(0.2)

            # Step 5 — Overall score
            prior_analyses = [job_resp, resume_resp, skill_resp, exp_resp]
            with st.spinner("🏆 Calculating overall score…"):
                score_resp = self.agents.calculate_overall_score(prior_analyses)
            result["overall_score"] = _parse_json_response(score_resp.response)
            st.write("✅ Overall score calculated")
            time.sleep(0.2)

            # Step 6 — Recommendations
            with st.spinner("💡 Generating recommendations…"):
                rec_resp = self.agents.generate_recommendations(
                    prior_analyses + [score_resp]
                )
            result["recommendations"] = _parse_json_response(rec_resp.response)
            st.write("✅ Recommendations generated")

            result["success"] = True

        except Exception as exc:
            st.error(f"❌ Error analysing resume {idx}: {exc}")
            result["success"] = False
            result["error"] = str(exc)

        return result
