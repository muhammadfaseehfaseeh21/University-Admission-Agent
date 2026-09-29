from crewai import Agent


def create_requirements_agent(llm):

    return Agent(
        role="Admission Requirements Agent",

        goal="Check the admission requirements for the student's program.",

        backstory=(
            "You are a university admission specialist. "
            "You check academic marks, subjects, and "
            "basic admission requirements."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
