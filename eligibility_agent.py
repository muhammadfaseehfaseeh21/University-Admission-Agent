from crewai import Agent


def create_eligibility_agent(llm):

    return Agent(
        role="Eligibility Agent",

        goal="Evaluate whether the student appears eligible.",

        backstory=(
            "You are a university eligibility expert. "
            "You compare the student's academic information "
            "with admission requirements."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
