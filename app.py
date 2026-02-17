import streamlit as st
import fitz
import json
from openai import OpenAI
from rag_utils import create_vector_store, retrieve
from skills import extract_skills

st.set_page_config(layout = "wide")
st.title("RAG-Powered Resume Intelligence System")

client = OpenAI(api_key = st.secrets["OPENAI_API_KEY"])

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

uploaded_file = st.file_uploader("Upload Resume(PDF)", type=["pdf"])
job_description = st.text_area("Enter job description")

if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    # combined_text = resume_text + "\n" + job_description
    # Create vector store ONLY from resume
    index, chunks = create_vector_store(resume_text)

# Use job description as retrieval query
    retrieved_chunks = retrieve(
        job_description,
        index,
        chunks
    )


    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are an expert AI career advisor.

Job Description:
{job_description}

Relevant Resume Sections:
{context}

Compare the resume against the job description.

Return ONLY a valid JSON object with:

- alignment_summary
- missing_skills_explanation
- bullet_improvements (list)
- impact_suggestions (list)
"""

    
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
    response_format={"type": "json_object"}
)
    output_text = response.choices[0].message.content

    st.subheader("AI Analysis")

    try:
        output_json  = json.loads(output_text)

        st.write("**Alignment Summary:**")
        st.write(output_json["alignment_summary"])

        st.write("**Missing Skills Explanation:**")
        st.write(output_json["missing_skills_explanation"])

        st.write("**Bullet Improvements:**")
        for item in output_json["bullet_improvements"]:
            st.write(f"- {item}")

        st.write("**Impact Suggestions:**")
        for item in output_json["impact_suggestions"]:
            st.write(f"- {item}")

    except Exception as e:
        st.write("Error processing output:")
        st.write(output_text)
        st.write(f"Exception: {e}")
            
