import os

from crewai import Agent, Crew, LLM, Process, Task

from agents.requirements_agent import create_requirements_agent
from agents.eligibility_agent import create_eligibility_agent
from agents.recommendation_agent import create_recommendation_agent


def run_admission_system(student):

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    # Groq LLM
    llm = LLM(
        model="openai/gpt-oss-120b",
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        temperature=0.2,
    )

    # Create agents
    requirements_agent = create_requirements_agent(llm)
    eligibility_agent = create_eligibility_agent(llm)
    recommendation_agent = create_recommendation_agent(llm)

    # -------------------------------
    # Task 1
    # -------------------------------

    requirements_task = Task(
        description=f"""
        Check the admission requirements for this student:

        {student}

        Check:
        - Minimum marks
        - Required subjects
        - Basic documents
        - Program requirements

        For this demonstration system use:

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

        Clearly mention that these are demonstration
        requirements and actual university requirements
        may be different.
        """,

        expected_output="""
        A clear report containing:
        1. Academic requirements
        2. Subject requirements
        3. Required documents
        4. Important notes
        """,

        agent=requirements_agent,
    )

    # -------------------------------
    # Task 2
    # -------------------------------

    eligibility_task = Task(
        description=f"""
        Evaluate the student's eligibility.

        Student information:

        {student}

        Use the admission requirements produced
        by the Requirements Agent.

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
        - Overall assessment
        """,

        agent=eligibility_agent,

        context=[requirements_task],
    )

    # -------------------------------
    # Task 3
    # -------------------------------

    recommendation_task = Task(
        description=f"""
        Recommend suitable programs for this student:

        {student}

        Consider:
        - Academic marks
        - Academic group
        - Student interests
        - Eligibility result

        Available programs:

        1. BS Computer Science
        2. BS Software Engineering
        3. BS Artificial Intelligence
        4. BS Data Science
        5. BS Cyber Security
        6. BS Business Administration

        Explain why each recommended program
        may fit the student's profile.

        Do not guarantee admission.
        """,

        expected_output="""
        A program recommendation report containing:
        - Suitable programs
        - Reason for recommendation
        - Eligibility considerations
        - Important verification notes
        """,

        agent=recommendation_agent,

        context=[
            requirements_task,
            eligibility_task,
        ],
    )

    # -------------------------------
    # Crew
    # -------------------------------

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

    result = crew.kickoff()

    return result.raw
