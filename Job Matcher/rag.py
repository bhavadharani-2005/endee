import json
import os
from typing import Any

from openai import OpenAI

from embed import INDEX_NAME, get_embedder, get_endee_client, load_jobs


def extract_resume_skills(resume_text: str, jobs: list[dict[str, Any]]) -> list[str]:
    resume_lower = resume_text.lower()
    unique_skills: list[str] = []
    seen: set[str] = set()

    for job in jobs:
        for skill in job.get("skills", []):
            normalized = skill.lower()
            if normalized in resume_lower and normalized not in seen:
                unique_skills.append(skill)
                seen.add(normalized)
    return unique_skills


def semantic_search(resume_text: str, top_k: int = 3) -> list[dict[str, Any]]:
    embedder = get_embedder()
    client = get_endee_client()
    index = client.get_index(name=INDEX_NAME)

    vector = embedder.encode(resume_text, normalize_embeddings=True).tolist()
    results = index.query(vector=vector, top_k=top_k)

    cleaned_results = []
    for item in results or []:
        cleaned_results.append(
            {
                "id": item.get("id"),
                "similarity": item.get("similarity", 0.0),
                "meta": item.get("meta", {}),
            }
        )
    return cleaned_results


def rule_based_insights(
    resume_text: str,
    matches: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not matches:
        return {
            "best_suited_role": "No suitable match found",
            "matching_skills": [],
            "missing_skills": [],
            "suggestions": [
                "Add more project details, measurable achievements, and relevant technical skills.",
                "Include a professional summary that clearly states your target role.",
            ],
        }

    top_match = matches[0]
    top_meta = top_match.get("meta", {})
    top_skills = top_meta.get("skills", [])
    resume_lower = resume_text.lower()
    resume_skills = extract_resume_skills(resume_text, jobs)

    matched = [skill for skill in top_skills if skill.lower() in resume_lower]
    missing = [skill for skill in top_skills if skill.lower() not in resume_lower]

    suggestions: list[str] = []
    if missing:
        suggestions.append(f"Add evidence of experience with: {', '.join(missing[:5])}.")
    suggestions.append("Highlight quantifiable achievements for each role or project.")
    suggestions.append("Tailor the summary section to the target position and domain.")

    return {
        "best_suited_role": top_meta.get("title", "Unknown Role"),
        "matching_skills": matched or resume_skills,
        "missing_skills": missing,
        "suggestions": suggestions,
    }


def generate_openai_insights(
    resume_text: str,
    matches: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)
    retrieved_context = []
    for match in matches:
        meta = match.get("meta", {})
        retrieved_context.append(
            {
                "title": meta.get("title", ""),
                "skills": meta.get("skills", []),
                "description": meta.get("description", ""),
                "similarity": match.get("similarity", 0.0),
            }
        )

    prompt = {
        "resume_text": resume_text,
        "retrieved_jobs": retrieved_context,
        "required_output": {
            "best_suited_role": "string",
            "matching_skills": ["string"],
            "missing_skills": ["string"],
            "suggestions": ["string"],
        },
    }

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a resume analysis assistant. Use only the provided resume text and "
                    "retrieved job data. Return strict JSON with keys: best_suited_role, "
                    "matching_skills, missing_skills, suggestions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=True),
            },
        ],
    )

    content = response.output_text
    return json.loads(content)


def analyze_resume(resume_text: str) -> dict[str, Any]:
    jobs = load_jobs()
    matches = semantic_search(resume_text, top_k=3)

    try:
        insights = generate_openai_insights(resume_text, matches, jobs)
    except Exception:
        insights = rule_based_insights(resume_text, matches, jobs)

    return {
        "matches": matches,
        "insights": insights,
    }


__all__ = [
    "analyze_resume",
    "extract_resume_skills",
    "rule_based_insights",
    "semantic_search",
]
