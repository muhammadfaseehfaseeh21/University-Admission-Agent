import os

from crewai import Crew, LLM, Process, Task

from requirements_agent import create_requirements_agent
from eligibility_agent import create_eligibility_agent
from recommendation_agent import create_recommendation_agent


def run_admission_system(student):

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    # Groq LLM through CrewAI
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2,
    )

    # Create agents
    requirements_agent = create_requirements_agent(llm)
    eligibility_agent = create_eligibility_agent(llm)
    recommendation_agent = create_recommendation_agent(llm)

    # Task 1: Requirements
    requirements_task = Task(
        description=f"""
        Check the admission requirements for this student:

        {student}

        Check:
        - Minimum academic marks
        - Required subjects
        - Basic documents
        - Program requirements

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

        Clearly state that these are demonstration
        requirements and actual university requirements
        may be different.
        """,
        expected_output="""
        A clear admission requirements report containing:

        1. Academic requirements
        2. Subject requirements
        3. Required documents
        4. Important notes
        """,
        agent=requirements_agent,
    )

    # Task 2: Eligibility
    eligibility_task = Task(
        description=f"""
        Evaluate this student's eligibility:

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

    # Task 3: Program Recommendation
    recommendation_task = Task(
        description=f"""
        Recommend suitable university programs for:

        {student}

        Consider:

        - Academic marks
        - Academic group
        - Student interests
        - Eligibility assessment

        Available programs:

        1. BS Computer Science
        2. BS Software Engineering
        3. BS Artificial Intelligence
        4. BS Data Science
        5. BS Cyber Security
        6. BS Business Administration

        Explain why each recommended program
        may be suitable.

        Do not guarantee admission.
        """,
        expected_output="""
        A clear program recommendation report containing:

        - Suitable programs
        - Reason for each recommendation
        - Eligibility considerations
        - Important verification notes
        """,
        agent=recommendation_agent,
        context=[
            requirements_task,
            eligibility_task,
        ],
    )

    # Create Crew
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

    # Run Crew
    result = crew.kickoff()

    return result.raw
