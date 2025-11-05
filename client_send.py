from src.open_deep_research.client import OpenDeepResearchClient

client = OpenDeepResearchClient("http://localhost:8000")
print(client.health_check())

response = client.research(
    query="what is tesla latest stock price?",
    allow_clarification=False
)

print(
    "Job status {status}\n\nResearch: {research_brief}\n\nFinal report {final_report}...".format(
    status=response.status,
    research_brief=response.research_brief,
    final_report=response.final_report[:256]
    )
)
