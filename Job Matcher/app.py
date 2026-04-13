import streamlit as st

from embed import ensure_jobs_index
from rag import analyze_resume


st.set_page_config(
    page_title="AI Resume Matcher",
    layout="wide",
)


def render_match_card(match: dict, rank: int) -> None:
    meta = match.get("meta", {})
    title = meta.get("title", "Unknown Role")
    skills = meta.get("skills", [])
    score = match.get("score") or match.get("similarity")

    with st.container(border=True):
        st.subheader(f"{rank}. {title}")

        if score is not None:
            st.caption(f"Similarity Score: {score:.4f}")

        st.write("**Key Skills:**")
        st.write(", ".join(skills) if skills else "No skills metadata available.")


def main() -> None:
    st.title("AI Resume Matcher")
    st.write(
        "Paste a resume below to find relevant jobs using semantic search and AI insights."
    )

    resume_text = st.text_area(
        "Resume Text",
        height=280,
        placeholder="Paste resume text here...",
    )

    if st.button("Analyze Resume", type="primary", use_container_width=True):

        if not resume_text.strip():
            st.warning("Please enter resume text.")
            return

        try:
            with st.spinner("Preparing job index..."):
                ensure_jobs_index()

            with st.spinner("Analyzing resume..."):
                result = analyze_resume(resume_text)

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.info("Make sure Endee server is running on http://127.0.0.1:8080")
            return

        matches = result.get("matches", [])
        insights = result.get("insights", {})

        if not matches:
            st.info("No matches found. Try improving your resume content.")
            return

        left, right = st.columns([1.2, 1])

        # LEFT SIDE
        with left:
            st.header("Top Job Matches")
            for idx, match in enumerate(matches, start=1):
                render_match_card(match, idx)

        # RIGHT SIDE
        with right:
            st.header("AI Insights")

            st.write(f"**Best Role:** {insights.get('best_suited_role', 'N/A')}")

            st.write("**Matching Skills:**")
            st.write(", ".join(insights.get("matching_skills", [])) or "None")

            st.write("**Missing Skills:**")
            st.write(", ".join(insights.get("missing_skills", [])) or "None")

            st.write("**Suggestions:**")
            suggestions = insights.get("suggestions", [])
            if suggestions:
                for s in suggestions:
                    st.write(f"- {s}")
            else:
                st.write("No suggestions needed.")


if __name__ == "__main__":
    main()
