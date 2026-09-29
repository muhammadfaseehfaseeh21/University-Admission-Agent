from crewai import Agent


def create_recommendation_agent(llm):

    return Agent(
        role="Program Recommendation Agent",

        goal=(
            "Recommend suitable university programs based "
            "on the student's academic profile and interests."
        ),

        backstory=(
            "You are a university academic advisor. "
            "You analyze academic background, interests, "
            "and eligibility to recommend suitable programs."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
