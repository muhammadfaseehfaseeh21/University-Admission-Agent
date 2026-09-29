import os

import streamlit as st

from crew import run_admission_system


# --------------------------------
# Page setup
# --------------------------------

st.set_page_config(
    page_title="University Admission AI",
    page_icon="🎓",
)


st.title("🎓 University Admission AI")

st.write(
    "Multi-Agent University Admission System"
)

st.write(
    "The system checks requirements, evaluates "
    "eligibility, and recommends programs."
)


# --------------------------------
# API Key
# --------------------------------

if "GROQ_API_KEY" in st.secrets:

    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# --------------------------------
# Student Information
# --------------------------------

st.subheader("👨‍🎓 Student Information")


name = st.text_input(
    "Student Name"
)


matric_marks = st.number_input(
    "Matric / O-Level Percentage",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
)


intermediate_marks = st.number_input(
    "Intermediate / A-Level Percentage",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
)


academic_group = st.selectbox(
    "Academic Group",
    [
        "Pre-Engineering",
        "Pre-Medical",
        "ICS / Computer Science",
        "General Science",
        "Commerce",
        "Humanities",
    ],
)


program = st.selectbox(
    "Desired Program",
    [
        "BS Computer Science",
        "BS Software Engineering",
        "BS Artificial Intelligence",
        "BS Data Science",
        "BS Cyber Security",
        "BS Business Administration",
    ],
)


interests = st.text_area(
    "Student Interests",
    placeholder=(
        "Example: Programming, AI, Data Science, "
        "Cyber Security..."
    ),
)


# --------------------------------
# Run system
# --------------------------------

if st.button(
    "🚀 Analyze Admission",
    type="primary",
):

    if not name:

        st.warning(
            "Please enter the student name."
        )

    elif not interests:

        st.warning(
            "Please enter the student's interests."
        )

    elif not os.environ.get("GROQ_API_KEY"):

        st.error(
            "GROQ_API_KEY is missing. "
            "Please add it to Streamlit Secrets."
        )

    else:

        student = f"""
        Student Name: {name}

        Matric / O-Level:
        {matric_marks}%

        Intermediate / A-Level:
        {intermediate_marks}%

        Academic Group:
        {academic_group}

        Desired Program:
        {program}

        Interests:
        {interests}
        """

        try:

            with st.spinner(
                "🤖 AI Agents are analyzing the student..."
            ):

                result = run_admission_system(student)

            st.success(
                "Admission analysis completed!"
            )

            st.subheader(
                "📋 Admission Report"
            )

            st.markdown(result)

        except Exception as error:

            st.error(
                "An error occurred while running "
                "the admission system."
            )

            st.write(str(error))
