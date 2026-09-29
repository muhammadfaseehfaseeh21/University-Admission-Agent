import os

from crewai import Crew, LLM, Process, Task

from requirements_agent import create_requirements_agent
from eligibility_agent import create_eligibility_agent
from recommendation_agent import create_recommendation_agent


def run_admission_system(student):

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    # Groq LLM
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
    )

    # Create agents
    requirements_agent = create_requirements_agent(llm)
    eligibility_agent = create_eligibility_agent(llm)
    recommendation_agent = create_recommendation_agent(llm)

    # ------------------------------------------------
    # TASK 1: Admission Requirements
    # ------------------------------------------------

    requirements_task = Task(
        description=f"""
        Check the admission requirements for this student.

        Student Information:
        {student}

        Check:

        1. Academic marks
        2. Academic group
        3. Required subjects
        4. Basic documents
        5. Program requirements

        Use these demonstration requirements:

        BS Computer Science:
        Minimum 50% and Mathematics expected.

        BS Software Engineering:
        Minimum 50% and Mathematics expected.

        BS Artificial Intelligence:
        Minimum 50% and Mathematics expected.

        BS Data Science:
        Minimum 50% and Mathematics expected.

        BS Cyber Security:
        Minimum 50% and Mathematics expected.

        BS Business Administration:
        Minimum 45%.

        Important:
        These are demonstration requirements only.
        Actual university requirements may be different.
        """,

        expected_output="""
        A clear admission requirements report containing:

        1. Academic requirements
        2. Subject requirements
        3. Required documents
        4. Program requirements
        5. Important notes
        """,

        agent=requirements_agent,
    )

    # ------------------------------------------------
    # TASK 2: Eligibility
    # ------------------------------------------------

    eligibility_task = Task(
        description=f"""
        Evaluate the apparent eligibility of this student.

        Student Information:
        {student}

        Use the Requirements Agent's report.

        Explain:

        1. Requirements satisfied
        2. Requirements not satisfied
        3. Missing information
        4. Potential concerns
        5. Overall eligibility assessment

        Do not guarantee admission.
        """,

        expected_output="""
        A clear eligibility report containing:

        - Satisfied requirements
        - Unsatisfied requirements
        - Missing information
        - Potential concerns
        - Overall eligibility assessment
        """,

        agent=eligibility_agent,

        context=[requirements_task],
    )

    # ------------------------------------------------
    # TASK 3: Program Recommendation
    # ------------------------------------------------

    recommendation_task = Task(
        description=f"""
        Recommend suitable university programs for this student.

        Student Information:
        {student}

        Consider:

        1. Academic marks
        2. Academic group
        3. Student interests
        4. Eligibility assessment

        Available programs:

        - BS Computer Science
        - BS Software Engineering
        - BS Artificial Intelligence
        - BS Data Science
        - BS Cyber Security
        - BS Business Administration

        Explain why each suitable program may match
        the student's academic background and interests.

        Do not guarantee admission.
        """,

        expected_output="""
        A clear program recommendation report containing:

        1. Suitable programs
        2. Reason for each recommendation
        3. Eligibility considerations
        4. Important verification notes
        """,

        agent=recommendation_agent,

        context=[
            requirements_task,
            eligibility_task,
        ],
    )

    # ------------------------------------------------
    # CREATE CREW
    # ------------------------------------------------

    crew = Crew(
        agents=[
            requirements_agent,
            eligibility_agent,
            recommendation_agent,
        ],

        tasks=[
            requirements_task,
            eligibility_task,
            recommendation_task,
        ],

        process=Process.sequential,

        verbose=False,
    )

    # Run the multi-agent system
    result = crew.kickoff()

    return result.raw
